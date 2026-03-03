bash

rm /mnt/user-data/outputs/main.py && cat > /mnt/user-data/outputs/main.py << 'PYEOF'
import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

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
                "pmid": pmid, "title": title, "abstract": abstract,
                "year": year, "journal": journal, "authors": author_str, "citations": None
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


class SynthesizeRequest(BaseModel):
    hypothesis: str
    papers: list

@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    papers_text = "\n\n---\n\n".join([
        f"[{i+1}] {p['title']} ({p['authors']}, {p['year']}, {p['journal']})\nCitations: {p.get('citations') or 'N/A'}\n{p['abstract']}"
        for i, p in enumerate(req.papers)
    ])

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1500,
                "system": """You are an expert biomedical literature reviewer. Synthesize the provided real PubMed abstracts.
Return ONLY a JSON object (no markdown) with:
{"summary":"2-3 paragraphs","keyFindings":["f1","f2","f3","f4"],"consensus":"...","gaps":"...","verdict":"Supported OR Partially Supported OR Insufficient Evidence OR Contradicted","confidence":"Low OR Medium OR High"}""",
                "messages": [{"role": "user", "content": f"Hypothesis: \"{req.hypothesis}\"\n\n{papers_text}"}]
            }
        )
        data = resp.json()
        if "error" in data:
            return {"error": data["error"]["message"]}

        text = "".join(b["text"] for b in data.get("content", []) if b["type"] == "text")
        import re, json
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            return {"error": "Could not parse synthesis"}
        return json.loads(match.group())


app.mount("/", StaticFiles(directory="static", html=True), name="static")
PYEOF
echo "done"
