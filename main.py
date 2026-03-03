import os
import re
import json
import httpx
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


@app.get("/health")
def health():
    return {"status": "ok", "ai": "groq/llama-3.3-70b"}


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
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured on server."}

    papers_text = "\n\n---\n\n".join([
        f"[{i+1}] {p['title']} ({p['authors']}, {p['year']}, {p['journal']})\nCitations: {p.get('citations') or 'N/A'}\n{p['abstract']}"
        for i, p in enumerate(req.papers)
    ])

    prompt = f"""You are an expert biomedical literature reviewer. Synthesize these PubMed abstracts.

Hypothesis: "{req.hypothesis}"

Papers:
{papers_text}

Return ONLY a JSON object (no markdown, no backticks) with exactly these keys:
{{"summary":"2-3 paragraph executive summary","keyFindings":["finding 1","finding 2","finding 3","finding 4"],"consensus":"what the literature agrees on","gaps":"key research gaps and contradictions","verdict":"Supported OR Partially Supported OR Insufficient Evidence OR Contradicted","confidence":"Low OR Medium OR High"}}"""

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            GROQ_BASE,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1500
            }
        )
        data = resp.json()
        if "error" in data:
            return {"error": data["error"].get("message", "Groq API error")}

        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            return {"error": "Could not parse synthesis response"}
        return json.loads(match.group())


app.mount("/", StaticFiles(directory="static", html=True), name="static")
