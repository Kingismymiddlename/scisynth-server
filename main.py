import os
import re
import json
import httpx
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


load_dotenv()

app = FastAPI(title="Biomedical Literature Synthesizer")

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

# Correct replacement for deprecated llama-3.3-70b-versatile
GROQ_MODEL = "openai/gpt-oss-120b"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": "groq",
        "ai": GROQ_MODEL,
    }


def get_all_text(element: Optional[ET.Element], default: str = "") -> str:
    """Safely extract text including nested XML tags."""
    if element is None:
        return default

    text = " ".join(" ".join(element.itertext()).split())
    return text if text else default


def get_first_text(article: ET.Element, path: str, default: str = "") -> str:
    """Find first matching XML element and return clean text."""
    return get_all_text(article.find(path), default)


def extract_abstract(article: ET.Element) -> str:
    """PubMed abstracts may have multiple AbstractText sections."""
    abstract_parts = []

    for node in article.findall(".//Abstract/AbstractText"):
        label = node.attrib.get("Label")
        text = get_all_text(node)

        if text:
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)

    return " ".join(abstract_parts) if abstract_parts else "No abstract available"


def extract_year(article: ET.Element) -> str:
    year = article.findtext(".//PubDate/Year")
    if year:
        return year

    medline_date = article.findtext(".//PubDate/MedlineDate")
    if medline_date:
        match = re.search(r"\d{4}", medline_date)
        if match:
            return match.group(0)
        return medline_date

    return "Unknown"


def extract_authors(article: ET.Element, max_authors: int = 3) -> str:
    author_list = article.findall(".//Author")
    authors = []

    for author in author_list[:max_authors]:
        last_name = author.findtext("LastName") or ""
        fore_name = author.findtext("ForeName") or ""
        collective_name = author.findtext("CollectiveName") or ""

        if last_name:
            authors.append(f"{last_name} {fore_name}".strip())
        elif collective_name:
            authors.append(collective_name.strip())

    if not authors:
        return "Unknown authors"

    author_str = ", ".join(authors)
    if len(author_list) > max_authors:
        author_str += " et al."

    return author_str


def parse_model_json(text: str) -> Dict[str, Any]:
    """Parse JSON from model response, with fallback for accidental extra text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("Could not parse synthesis response")
        return json.loads(match.group())


@app.get("/search")
async def search(
    query: str,
    max_results: int = Query(default=15, ge=1, le=50),
):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            search_resp = await client.get(
                f"{PUBMED_BASE}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json",
                    "sort": "relevance",
                },
            )
            search_resp.raise_for_status()

            ids = search_resp.json().get("esearchresult", {}).get("idlist", [])

            if not ids:
                return {"papers": [], "error": "No papers found"}

            fetch_resp = await client.get(
                f"{PUBMED_BASE}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "xml",
                    "rettype": "abstract",
                },
            )
            fetch_resp.raise_for_status()

            root = ET.fromstring(fetch_resp.text)
            papers = []

            for article in root.findall(".//PubmedArticle"):
                pmid = article.findtext(".//PMID") or ""
                title = get_first_text(article, ".//ArticleTitle", "No title")
                abstract = extract_abstract(article)
                year = extract_year(article)
                journal = get_first_text(article, ".//Journal/Title", "Unknown Journal")
                authors = extract_authors(article)

                papers.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "year": year,
                        "journal": journal,
                        "authors": authors,
                        "citations": None,
                    }
                )

            # Optional citation enrichment from Semantic Scholar
            try:
                pmids = [p["pmid"] for p in papers if p.get("pmid")]

                if pmids:
                    sem_resp = await client.post(
                        f"{SEMANTIC_BASE}/paper/batch",
                        params={"fields": "citationCount,externalIds"},
                        json={"ids": [f"PMID:{pid}" for pid in pmids]},
                        timeout=10,
                    )
                    sem_resp.raise_for_status()

                    citation_map = {}

                    for item in sem_resp.json():
                        if not item:
                            continue

                        external_ids = item.get("externalIds") or {}
                        pubmed_id = str(external_ids.get("PubMed", ""))

                        if pubmed_id:
                            citation_map[pubmed_id] = item.get("citationCount", 0)

                    for paper in papers:
                        paper["citations"] = citation_map.get(paper["pmid"])

            except Exception:
                # Citation enrichment is optional, so search should still succeed.
                pass

            return {"papers": papers}

    except ET.ParseError:
        return {"papers": [], "error": "Failed to parse PubMed XML response"}
    except httpx.HTTPStatusError as e:
        return {
            "papers": [],
            "error": f"External API HTTP error: {e.response.status_code}",
            "details": e.response.text,
        }
    except Exception as e:
        return {"papers": [], "error": f"Search failed: {str(e)}"}


class Paper(BaseModel):
    pmid: Optional[str] = ""
    title: str = "No title"
    abstract: str = "No abstract available"
    year: Optional[str] = "Unknown"
    journal: Optional[str] = "Unknown Journal"
    authors: Optional[str] = "Unknown authors"
    citations: Optional[int] = None


class SynthesizeRequest(BaseModel):
    hypothesis: str = Field(..., min_length=1)
    papers: List[Paper]


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured on server."}

    if not req.papers:
        return {"error": "No papers provided for synthesis."}

    papers_text = "\n\n---\n\n".join(
        [
            (
                f"[{i + 1}] {paper.title} "
                f"({paper.authors}, {paper.year}, {paper.journal})\n"
                f"PMID: {paper.pmid or 'N/A'}\n"
                f"Citations: {paper.citations if paper.citations is not None else 'N/A'}\n"
                f"{paper.abstract}"
            )
            for i, paper in enumerate(req.papers)
        ]
    )

    prompt = f"""
You are an expert biomedical literature reviewer. Synthesize these PubMed abstracts.

Hypothesis: "{req.hypothesis}"

Papers:
{papers_text}

Return ONLY a valid JSON object with exactly these keys:
{{
  "summary": "2-3 paragraph executive summary",
  "keyFindings": ["finding 1", "finding 2", "finding 3", "finding 4"],
  "consensus": "what the literature agrees on",
  "gaps": "key research gaps and contradictions",
  "verdict": "Supported OR Partially Supported OR Insufficient Evidence OR Contradicted",
  "confidence": "Low OR Medium OR High"
}}

Do not include markdown.
Do not include backticks.
Do not include any text before or after the JSON object.
""".strip()

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                GROQ_BASE,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1500,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()

            data = resp.json()

            if "error" in data:
                return {
                    "error": data["error"].get("message", "Groq API error")
                }

            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            if not text:
                return {"error": "Empty synthesis response from model."}

            return parse_model_json(text)

    except httpx.HTTPStatusError as e:
        return {
            "error": f"Groq API HTTP error: {e.response.status_code}",
            "details": e.response.text,
        }
    except json.JSONDecodeError:
        return {"error": "Model returned invalid JSON."}
    except Exception as e:
        return {"error": f"Synthesis failed: {str(e)}"}


# Mount static frontend only if the folder exists.
# This prevents deployment crash when static/ is missing.
static_dir = Path("static")
if static_dir.exists() and static_dir.is_dir():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
