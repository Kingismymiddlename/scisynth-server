import httpx
import xml.etree.ElementTree as ET
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SEMANTIC_BASE = "https://api.semanticscholar.org/graph/v1"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/search")
async def search(query: str, max_results: int = 15):
    async with httpx.AsyncClient(timeout=30) as client:

        search_resp = await client.get(
            f"{PUBMED_BASE}/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "sort": "relevance"}
        )
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"papers": [], "error": "No papers found"}

        fetch_resp = await client.get(
            f"{PUBMED_BASE}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract"}
        )
        root = ET.fromstring(fetch_resp.text)
        papers = []

        for article in root.findall(".//PubmedArticle"):
            pmid     = article.findtext(".//PMID") or ""
            title    = article.findtext(".//ArticleTitle") or "No title"
            abstract = article.findtext(".//AbstractText") or "No abstract available"
            year     = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate") or "Unknown"
            journal  = article.findtext(".//Journal/Title") or "Unknown Journal"
            author_list = article.findall(".//Author")
            authors = []
            for a in author_list[:3]:
                ln = a.findtext("LastName") or ""
                fn = a.findtext("ForeName") or ""
                if ln:
                    authors.append(f"{ln} {fn}".strip())
            author_str = ", ".join(authors) + (" et al." if len(author_list) > 3 else "")
            papers.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": journal,
                "authors": author_str,
                "citations": None
            })

        try:
            pmids = [p["pmid"] for p in papers if p["pmid"]]
            sem_resp = await client.post(
                f"{SEMANTIC_BASE}/paper/batch",
                params={"fields": "citationCount"},
                json={"ids": [f"PMID:{pid}" for pid in pmids]},
                timeout=10
            )
            citation_map = {}
            for item in sem_resp.json():
                if item and "externalIds" in item:
                    pid = str(item["externalIds"].get("PubMed", ""))
                    citation_map[pid] = item.get("citationCount", 0)
            for p in papers:
                p["citations"] = citation_map.get(p["pmid"])
        except Exception:
            pass

        return {"papers": papers}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
