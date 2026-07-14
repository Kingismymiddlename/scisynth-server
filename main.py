import os
import re
import json
import httpx
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


load_dotenv()

app = FastAPI(title="SciSynth - Biomedical Literature Synthesizer")

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
GROQ_MODEL = "openai/gpt-oss-120b"


@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SciSynth - Biomedical Literature Synthesizer</title>
  <style>
    :root {
      --bg: #070b08;
      --surface: #101711;
      --surface2: #162016;
      --border: rgba(74, 180, 100, 0.22);
      --green: #4ab464;
      --green2: #66d37b;
      --text: #e8f0e8;
      --muted: #91a591;
      --muted2: #5d6c5d;
      --amber: #d4a843;
      --red: #c45c5c;
      --blue: #5b9bd4;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 20% 0%, rgba(74, 180, 100, 0.12), transparent 32%),
        radial-gradient(circle at 80% 10%, rgba(91, 155, 212, 0.09), transparent 28%),
        var(--bg);
      color: var(--text);
      font-family: var(--sans);
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(74,180,100,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(74,180,100,.035) 1px, transparent 1px);
      background-size: 42px 42px;
      pointer-events: none;
    }

    .wrap {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 34px 0 60px;
      position: relative;
      z-index: 1;
    }

    .nav {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 42px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .mark {
      width: 34px;
      height: 34px;
      border: 1.5px solid var(--green);
      border-radius: 10px;
      display: grid;
      place-items: center;
      box-shadow: 0 0 28px rgba(74, 180, 100, 0.15);
    }

    .mark::after {
      content: "";
      width: 11px;
      height: 11px;
      background: var(--green);
      border-radius: 999px;
      box-shadow: 0 0 12px var(--green);
    }

    .brand-title {
      font-size: 17px;
      letter-spacing: 0.02em;
      font-weight: 650;
    }

    .brand-title span {
      color: var(--green);
    }

    .pill {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--green);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 7px 12px;
      background: rgba(74, 180, 100, 0.07);
    }

    .hero {
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 26px;
      align-items: stretch;
      margin-bottom: 26px;
    }

    .hero-card,
    .panel,
    .result-card {
      background: rgba(16, 23, 17, 0.88);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: 0 28px 80px rgba(0, 0, 0, 0.38);
      backdrop-filter: blur(10px);
    }

    .hero-card {
      padding: 34px;
    }

    .panel {
      padding: 24px;
    }

    .eyebrow {
      font-family: var(--mono);
      color: var(--green);
      font-size: 11px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      margin-bottom: 14px;
    }

    h1 {
      font-size: clamp(34px, 5vw, 62px);
      line-height: 0.96;
      letter-spacing: -0.055em;
      margin: 0 0 18px;
    }

    h1 em {
      color: var(--green);
      font-style: normal;
    }

    .lead {
      color: var(--muted);
      font-size: 16px;
      line-height: 1.72;
      max-width: 58ch;
      margin: 0 0 28px;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-top: 20px;
    }

    .stat {
      padding: 14px;
      border: 1px solid rgba(74, 180, 100, 0.14);
      border-radius: 14px;
      background: rgba(74, 180, 100, 0.045);
    }

    .stat strong {
      display: block;
      color: var(--text);
      font-size: 20px;
      margin-bottom: 4px;
    }

    .stat span {
      font-family: var(--mono);
      color: var(--muted2);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .panel-title {
      font-family: var(--mono);
      color: var(--muted2);
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 14px;
    }

    label {
      display: block;
      font-family: var(--mono);
      color: var(--muted2);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    textarea,
    input,
    select {
      width: 100%;
      background: #0b110c;
      color: var(--text);
      border: 1px solid rgba(74, 180, 100, 0.28);
      border-radius: 13px;
      padding: 13px 14px;
      font-size: 14px;
      outline: none;
      font-family: var(--sans);
      transition: border-color 0.18s, box-shadow 0.18s;
    }

    textarea {
      min-height: 116px;
      resize: vertical;
      line-height: 1.5;
    }

    textarea:focus,
    input:focus,
    select:focus {
      border-color: var(--green);
      box-shadow: 0 0 0 3px rgba(74, 180, 100, 0.10);
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 140px;
      gap: 12px;
      margin-top: 14px;
    }

    .btn-row {
      display: flex;
      gap: 10px;
      margin-top: 16px;
      flex-wrap: wrap;
    }

    button {
      appearance: none;
      border: 0;
      border-radius: 13px;
      padding: 13px 18px;
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.16s, opacity 0.16s, background 0.16s;
    }

    .primary {
      background: var(--green);
      color: #07120a;
    }

    .primary:hover {
      background: var(--green2);
      transform: translateY(-1px);
    }

    .ghost {
      color: var(--green);
      border: 1px solid var(--border);
      background: rgba(74, 180, 100, 0.07);
    }

    .ghost:hover {
      background: rgba(74, 180, 100, 0.13);
      transform: translateY(-1px);
    }

    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
      transform: none;
    }

    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }

    .chip {
      border: 1px solid rgba(91, 155, 212, 0.28);
      background: rgba(91, 155, 212, 0.08);
      color: var(--blue);
      border-radius: 999px;
      padding: 7px 10px;
      font-family: var(--mono);
      font-size: 11px;
      cursor: pointer;
    }

    .chip:hover {
      border-color: var(--blue);
    }

    .status {
      margin-top: 14px;
      min-height: 22px;
      color: var(--muted2);
      font-family: var(--mono);
      font-size: 11px;
    }

    .results {
      display: grid;
      gap: 16px;
      margin-top: 26px;
    }

    .result-card {
      padding: 22px;
    }

    .result-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      border-bottom: 1px solid rgba(74, 180, 100, 0.14);
      padding-bottom: 14px;
      margin-bottom: 16px;
    }

    .result-title {
      font-family: var(--mono);
      color: var(--muted2);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }

    .badge {
      font-family: var(--mono);
      font-size: 11px;
      border-radius: 999px;
      padding: 6px 11px;
      border: 1px solid rgba(74, 180, 100, 0.24);
      color: var(--green);
      background: rgba(74, 180, 100, 0.08);
      white-space: nowrap;
    }

    .badge.warn {
      color: var(--amber);
      border-color: rgba(212, 168, 67, 0.32);
      background: rgba(212, 168, 67, 0.08);
    }

    .badge.bad {
      color: var(--red);
      border-color: rgba(196, 92, 92, 0.32);
      background: rgba(196, 92, 92, 0.08);
    }

    .summary {
      color: var(--muted);
      line-height: 1.75;
      white-space: pre-wrap;
      font-size: 14px;
    }

    .finding {
      display: flex;
      gap: 10px;
      padding: 11px 0;
      border-top: 1px dashed rgba(74, 180, 100, 0.13);
    }

    .finding:first-child {
      border-top: 0;
    }

    .num {
      color: var(--green);
      font-family: var(--mono);
      font-size: 12px;
      min-width: 28px;
    }

    .finding-text {
      color: var(--text);
      line-height: 1.58;
      font-size: 14px;
    }

    .paper {
      padding: 13px 0;
      border-top: 1px dashed rgba(74, 180, 100, 0.13);
    }

    .paper:first-child {
      border-top: 0;
    }

    .paper a {
      color: var(--text);
      font-weight: 650;
      text-decoration: none;
    }

    .paper a:hover {
      color: var(--green);
    }

    .meta {
      color: var(--muted2);
      font-family: var(--mono);
      font-size: 11px;
      margin-top: 5px;
      line-height: 1.5;
    }

    .abstract {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.62;
      margin-top: 7px;
    }

    .grid2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }

    .mini {
      background: rgba(7, 11, 8, 0.55);
      border: 1px solid rgba(74, 180, 100, 0.13);
      border-radius: 16px;
      padding: 16px;
    }

    .mini h3 {
      margin: 0 0 8px;
      font-family: var(--mono);
      color: var(--muted2);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }

    .mini p {
      margin: 0;
      color: var(--muted);
      line-height: 1.65;
      font-size: 13px;
    }

    .error {
      color: var(--red);
      background: rgba(196, 92, 92, 0.08);
      border: 1px solid rgba(196, 92, 92, 0.22);
      border-radius: 16px;
      padding: 14px;
      font-size: 13px;
      line-height: 1.6;
    }

    .loading {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }

    .dot {
      width: 8px;
      height: 8px;
      background: var(--green);
      border-radius: 999px;
      animation: pulse 1s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 0.35; transform: scale(0.8); }
      50% { opacity: 1; transform: scale(1); }
    }

    .foot {
      margin-top: 30px;
      color: var(--muted2);
      font-family: var(--mono);
      font-size: 11px;
      line-height: 1.7;
      text-align: center;
    }

    @media (max-width: 860px) {
      .hero {
        grid-template-columns: 1fr;
      }

      .grid2 {
        grid-template-columns: 1fr;
      }

      .row {
        grid-template-columns: 1fr;
      }

      .stats {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <nav class="nav">
      <div class="brand">
        <div class="mark"></div>
        <div class="brand-title">Sci<span>Synth</span></div>
      </div>
      <div class="pill" id="health-pill">Checking server...</div>
    </nav>

    <section class="hero">
      <div class="hero-card">
        <div class="eyebrow">Biomedical Literature Intelligence</div>
        <h1>Turn PubMed evidence into a <em>scientific verdict.</em></h1>
        <p class="lead">
          Enter a biomedical claim or hypothesis. SciSynth searches PubMed, optionally enriches citation counts,
          and uses Groq GPT-OSS-120B to synthesize the evidence into a readable research verdict.
        </p>

        <div class="stats">
          <div class="stat">
            <strong>PubMed</strong>
            <span>Literature search</span>
          </div>
          <div class="stat">
            <strong>Groq</strong>
            <span>GPT-OSS-120B synthesis</span>
          </div>
          <div class="stat">
            <strong>JSON</strong>
            <span>Export-ready output</span>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-title">Run SciSynth</div>

        <label for="hypothesis">Hypothesis / scientific claim</label>
        <textarea id="hypothesis" placeholder="e.g. KRAS G12C drives non-small cell lung cancer progression"></textarea>

        <div class="row">
          <div>
            <label for="query">Search query override</label>
            <input id="query" placeholder="Leave blank to use the hypothesis" />
          </div>
          <div>
            <label for="maxResults">Max papers</label>
            <select id="maxResults">
              <option value="5">5</option>
              <option value="8" selected>8</option>
              <option value="10">10</option>
            </select>
          </div>
        </div>

        <div class="examples">
          <span class="chip" onclick="setExample('KRAS G12C drives non-small cell lung cancer progression')">KRAS G12C / NSCLC</span>
          <span class="chip" onclick="setExample('GLP-1 receptor agonists reduce cardiovascular mortality')">GLP-1 / CV mortality</span>
          <span class="chip" onclick="setExample('Coffee causes cancer')">Coffee / cancer</span>
        </div>

        <div class="btn-row">
          <button class="primary" id="runBtn" onclick="runSciSynth()">Search + Synthesize</button>
          <button class="ghost" onclick="searchOnly()">Search only</button>
          <button class="ghost" onclick="clearAll()">Clear</button>
        </div>

        <div class="status" id="status"></div>
      </div>
    </section>

    <section class="results" id="results"></section>

    <div class="foot">
      Research use only. Not medical advice. Not a substitute for scientific, clinical, regulatory, or legal review.<br/>
      API endpoints: /health - /search - /synthesize
    </div>
  </main>

  <script>
    let lastPapers = [];
    let lastSynthesis = null;

    const $ = (id) => document.getElementById(id);

    function escapeHTML(value) {
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function setStatus(text, loading = false) {
      $("status").innerHTML = loading
        ? '<span class="loading"><span class="dot"></span>' + escapeHTML(text) + '</span>'
        : escapeHTML(text || "");
    }

    function verdictClass(verdict) {
      const v = String(verdict || "").toLowerCase();
      if (v.includes("supported") && !v.includes("partially")) return "";
      if (v.includes("contradicted")) return "bad";
      return "warn";
    }

    async function checkHealth() {
      try {
        const response = await fetch("/health");
        const data = await response.json();
        $("health-pill").textContent = data.groq_key_configured
          ? "Server ready - Groq configured"
          : "Server ready - Groq key missing";
        $("health-pill").style.color = data.groq_key_configured ? "var(--green)" : "var(--amber)";
      } catch (e) {
        $("health-pill").textContent = "Server check failed";
        $("health-pill").style.color = "var(--red)";
      }
    }

    function setExample(text) {
      $("hypothesis").value = text;
      $("query").value = text;
    }

    function clearAll() {
      $("hypothesis").value = "";
      $("query").value = "";
      $("results").innerHTML = "";
      setStatus("");
      lastPapers = [];
      lastSynthesis = null;
    }

    async function fetchJSON(url, options = {}) {
      const response = await fetch(url, options);
      const text = await response.text();

      let data;
      try {
        data = JSON.parse(text);
      } catch (e) {
        throw new Error("Server returned non-JSON response: " + text.slice(0, 200));
      }

      if (!response.ok) {
        throw new Error(data.error || "HTTP error " + response.status);
      }

      return data;
    }

    async function getPapers() {
      const hypothesis = $("hypothesis").value.trim();
      const query = $("query").value.trim() || hypothesis;
      const maxResults = $("maxResults").value || "8";

      if (!query) {
        throw new Error("Please enter a hypothesis or search query.");
      }

      const data = await fetchJSON(
        "/search?query=" + encodeURIComponent(query) + "&max_results=" + encodeURIComponent(maxResults)
      );

      if (data.error && (!data.papers || !data.papers.length)) {
        throw new Error(data.error);
      }

      return data.papers || [];
    }

    function compactPapersForAI(papers) {
      return (papers || []).slice(0, 8).map((paper) => ({
        pmid: paper.pmid || "",
        title: paper.title || "No title",
        abstract: String(paper.abstract || "No abstract available").slice(0, 900),
        year: paper.year || "Unknown",
        journal: paper.journal || "Unknown Journal",
        authors: paper.authors || "Unknown authors",
        citations: paper.citations ?? null
      }));
    }

    async function searchOnly() {
      $("runBtn").disabled = true;
      setStatus("Searching PubMed...", true);

      try {
        const papers = await getPapers();
        lastPapers = papers;
        lastSynthesis = null;

        if (!papers.length) {
          $("results").innerHTML = '<div class="error">No papers found. Try a broader query.</div>';
          setStatus("No papers found.");
          return;
        }

        $("results").innerHTML = renderPapers(papers);
        setStatus("Found " + papers.length + " paper(s).");
      } catch (e) {
        $("results").innerHTML = '<div class="error">' + escapeHTML(e.message) + '</div>';
        setStatus("Search failed.");
      } finally {
        $("runBtn").disabled = false;
      }
    }

    async function runSciSynth() {
      const hypothesis = $("hypothesis").value.trim();

      if (!hypothesis) {
        $("results").innerHTML = '<div class="error">Please enter a hypothesis.</div>';
        return;
      }

      $("runBtn").disabled = true;
      setStatus("Searching PubMed...", true);

      try {
        const papers = await getPapers();
        lastPapers = papers;

        if (!papers.length) {
          $("results").innerHTML = '<div class="error">No papers found. Try a broader query.</div>';
          setStatus("No papers found.");
          return;
        }

        $("results").innerHTML = renderPapers(papers);
        setStatus("Synthesizing evidence with GPT-OSS-120B...", true);

        const synthesis = await fetchJSON("/synthesize", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            hypothesis,
            papers: compactPapersForAI(papers),
          }),
        });

        if (synthesis.error) {
          throw new Error(synthesis.error + (synthesis.details ? " - " + synthesis.details : ""));
        }

        lastSynthesis = synthesis;
        $("results").innerHTML = renderSynthesis(synthesis, hypothesis) + renderPapers(papers);
        setStatus("Done. Synthesized top " + Math.min(papers.length, 8) + " paper(s).");
      } catch (e) {
        $("results").innerHTML = '<div class="error">' + escapeHTML(e.message) + '</div>' + $("results").innerHTML;
        setStatus("Synthesis failed.");
      } finally {
        $("runBtn").disabled = false;
      }
    }

    function renderSynthesis(data, hypothesis) {
      const findings = Array.isArray(data.keyFindings) ? data.keyFindings : [];
      const badgeClass = verdictClass(data.verdict);

      return `
        <div class="result-card">
          <div class="result-head">
            <div>
              <div class="result-title">SciSynth Evidence Verdict</div>
              <div class="meta">Hypothesis: ${escapeHTML(hypothesis)}</div>
            </div>
            <span class="badge ${badgeClass}">${escapeHTML(data.verdict || "Unknown")} - ${escapeHTML(data.confidence || "Medium")}</span>
          </div>

          <div class="summary">${escapeHTML(data.summary || "No summary returned.")}</div>

          ${findings.length ? `
            <div style="margin-top: 18px;">
              ${findings.map((finding, index) => `
                <div class="finding">
                  <div class="num">${String(index + 1).padStart(2, "0")}</div>
                  <div class="finding-text">${escapeHTML(finding)}</div>
                </div>
              `).join("")}
            </div>
          ` : ""}

          <div class="grid2" style="margin-top: 18px;">
            <div class="mini">
              <h3>Consensus</h3>
              <p>${escapeHTML(data.consensus || "Not available.")}</p>
            </div>
            <div class="mini">
              <h3>Gaps / contradictions</h3>
              <p>${escapeHTML(data.gaps || "Not available.")}</p>
            </div>
          </div>

          <div class="btn-row" style="margin-top: 18px;">
            <button class="ghost" onclick="copyJSON()">Copy JSON</button>
            <button class="ghost" onclick="downloadJSON()">Download JSON</button>
            <button class="ghost" onclick="window.print()">Print / Save PDF</button>
          </div>
        </div>
      `;
    }

    function renderPapers(papers) {
      if (!papers || !papers.length) {
        return '<div class="result-card"><div class="summary">No papers found.</div></div>';
      }

      return `
        <div class="result-card">
          <div class="result-head">
            <div class="result-title">Retrieved PubMed Papers</div>
            <span class="badge">${papers.length} paper(s)</span>
          </div>

          ${papers.map((paper, index) => {
            const pmid = paper.pmid || "";
            const url = pmid ? "https://pubmed.ncbi.nlm.nih.gov/" + encodeURIComponent(pmid) + "/" : "#";
            const abstract = paper.abstract || "No abstract available";
            return `
              <div class="paper">
                <a href="${url}" target="_blank" rel="noopener">
                  [${index + 1}] ${escapeHTML(paper.title || "No title")}
                </a>
                <div class="meta">
                  ${escapeHTML(paper.authors || "Unknown authors")} -
                  ${escapeHTML(paper.year || "Unknown")} -
                  ${escapeHTML(paper.journal || "Unknown Journal")} -
                  PMID: ${escapeHTML(pmid || "N/A")} -
                  Citations: ${paper.citations === null || paper.citations === undefined ? "N/A" : escapeHTML(paper.citations)}
                </div>
                <div class="abstract">${escapeHTML(abstract.slice(0, 600))}${abstract.length > 600 ? "..." : ""}</div>
              </div>
            `;
          }).join("")}
        </div>
      `;
    }

    function copyJSON() {
      const payload = {
        generated_at: new Date().toISOString(),
        hypothesis: $("hypothesis").value.trim(),
        synthesis: lastSynthesis,
        papers: lastPapers,
      };

      navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      setStatus("Copied JSON to clipboard.");
    }

    function downloadJSON() {
      const payload = {
        generated_at: new Date().toISOString(),
        hypothesis: $("hypothesis").value.trim(),
        synthesis: lastSynthesis,
        papers: lastPapers,
      };

      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      });

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      a.href = url;
      a.download = "scisynth-" + ts + ".json";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setStatus("Downloaded JSON.");
    }

    checkHealth();
  </script>
</body>
</html>
    """


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tool": "SciSynth",
        "provider": "groq",
        "ai": GROQ_MODEL,
        "groq_key_configured": bool(GROQ_API_KEY),
    }


def get_all_text(element: Optional[ET.Element], default: str = "") -> str:
    if element is None:
        return default

    text = " ".join(" ".join(element.itertext()).split())
    return text if text else default


def get_first_text(article: ET.Element, path: str, default: str = "") -> str:
    return get_all_text(article.find(path), default)


def extract_abstract(article: ET.Element) -> str:
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


def clean_model_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_model_json(text: str) -> Dict[str, Any]:
    text = clean_model_text(text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Model returned JSON, but not an object.")
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("Could not parse synthesis response")

        parsed = json.loads(match.group())
        if not isinstance(parsed, dict):
            raise ValueError("Parsed JSON is not an object.")

        return parsed


@app.get("/search")
async def search(
    query: str,
    max_results: int = Query(default=8, ge=1, le=20),
):
    query = (query or "").strip()

    if not query:
        return {"papers": [], "error": "Query is required"}

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
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

            try:
                pmids = [p["pmid"] for p in papers if p.get("pmid")]

                if pmids:
                    sem_resp = await client.post(
                        f"{SEMANTIC_BASE}/paper/batch",
                        params={"fields": "citationCount,externalIds"},
                        json={"ids": [f"PMID:{pid}" for pid in pmids]},
                        timeout=10,
                    )

                    if sem_resp.status_code == 200:
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
                pass

            return {"papers": papers}

    except ET.ParseError:
        return {"papers": [], "error": "Failed to parse PubMed XML response"}
    except httpx.HTTPStatusError as e:
        return {
            "papers": [],
            "error": f"External API HTTP error: {e.response.status_code}",
            "details": e.response.text[:500],
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


def build_prompt(req: SynthesizeRequest) -> str:
    usable_papers = req.papers[:8]

    papers_text = "\n\n---\n\n".join(
        [
            (
                f"[{i + 1}] {paper.title} "
                f"({paper.authors}, {paper.year}, {paper.journal})\n"
                f"PMID: {paper.pmid or 'N/A'}\n"
                f"Citations: {paper.citations if paper.citations is not None else 'N/A'}\n"
                f"{(paper.abstract or 'No abstract available')[:900]}"
            )
            for i, paper in enumerate(usable_papers)
        ]
    )

    prompt = f"""
You are an expert biomedical literature reviewer. Synthesize these PubMed abstracts.

Hypothesis: "{req.hypothesis}"

Papers:
{papers_text}

Return ONLY a valid JSON object with exactly these keys:
{{
  "summary": "short 1 paragraph executive summary",
  "keyFindings": ["finding 1", "finding 2", "finding 3", "finding 4"],
  "consensus": "what the literature agrees on",
  "gaps": "key research gaps and contradictions",
  "verdict": "Supported OR Partially Supported OR Insufficient Evidence OR Contradicted",
  "confidence": "Low OR Medium OR High"
}}

Rules:
- Do not include markdown.
- Do not include backticks.
- Do not include any text before or after the JSON object.
- If evidence is weak, say Insufficient Evidence.
- Do not invent trial results or clinical claims not supported by the abstracts.
""".strip()

    return prompt


async def call_groq(prompt: str, use_json_mode: bool = True) -> Dict[str, Any]:
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON API for biomedical literature synthesis. "
                    "Return only a valid raw JSON object."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.3,
        "max_tokens": 1000,
    }

    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.post(
            GROQ_BASE,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        if resp.status_code >= 400:
            return {
                "error": f"Groq API HTTP error: {resp.status_code}",
                "details": resp.text[:500],
                "_status_code": resp.status_code,
            }

        data = resp.json()

        if "error" in data:
            return {
                "error": data["error"].get("message", "Groq API error"),
                "details": json.dumps(data["error"])[:500],
            }

        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not text:
            return {"error": "Empty synthesis response from model."}

        try:
            return parse_model_json(text)
        except Exception as e:
            return {
                "error": f"Model returned invalid JSON: {str(e)}",
                "raw_response": text[:500],
            }


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured on server."}

    if not req.papers:
        return {"error": "No papers provided for synthesis."}

    prompt = build_prompt(req)

    try:
        result = await call_groq(prompt, use_json_mode=True)

        if (
            isinstance(result, dict)
            and result.get("error")
            and result.get("_status_code") in {400, 422}
        ):
            fallback = await call_groq(prompt, use_json_mode=False)
            if not fallback.get("error"):
                return fallback

        return result

    except Exception as e:
        return {"error": f"Synthesis failed: {str(e)}"}
