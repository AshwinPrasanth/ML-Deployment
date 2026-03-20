# 🏥 Irish Health Insurance Recommender — Frontend

A Streamlit-based web application that recommends the best Irish health insurance plans for a user based on their health profile, using a two-layer AI pipeline (Rule Engine + LLM Reranker) with hybrid semantic search and PDF evidence highlighting.

---

## 🗂️ Frontend File Structure

```
Frontend/
├── streamlit_app_final.py        # Main Streamlit application
├── pdf_highlighter.py      # PDF evidence highlighting using PyMuPDF
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ How It Works

### Pipeline Overview

```
User Query (plain English)
        ↓
Profile Extractor     — parses age, conditions, medication, visits from text
        ↓
Hybrid RRF Retrieval  — FAISS (semantic) + BM25 (keyword) fused via Reciprocal Rank Fusion
        ↓
Rule Engine           — hard-rejects plans missing required cover, scores survivors
        ↓
LLM Reranker          — Llama-3.3-70B via Groq reasons holistically about each plan
        ↓
Top 3 Plans           — displayed side by side with scores, strengths, weaknesses
        ↓
PDF Highlighter       — auto-matches policy PDF, highlights evidence chunks in yellow
```

### Layer 1 — Rule Engine (Deterministic)
- Hard-rejects plans missing required cover (maternity / psychiatric / fertility / high-tech flags)
- Age soft-check — warns if outside 18–90 band
- Utilisation score — penalises high excess for frequent claimers
- Condition chunk-hit proxy — cardiac/cancer via keyword hit-rate

### Layer 2 — LLM Reranker (Intelligent)
- Reads surviving plan evidence and user profile
- Reasons about unknown conditions (neurological, diabetes, etc.)
- Produces fit score 0–10 and plain-English explanation per plan
- Handles nuance the rule engine cannot

---

## 🖥️ UI Features

| Feature | Description |
|---|---|
| **3 Plans Side by Side** | Top 3 recommended plans displayed in columns |
| **Fit Score Chips** | Colour-coded ⭐ Fit / ⚙️ Rule / 🔍 RRF scores per plan |
| **LLM Summary** | Plain-English explanation of why each plan suits the user |
| **Strengths / Weaknesses** | Green / red / amber pills per plan |
| **Coverage Gaps** | Conditions the plan appears weak on |
| **Numeric Data** | Excess, day limits, coverage %, copayments |
| **Rule Engine Notes** | Expandable — shows what the rule engine found |
| **Policy Chunks** | Expandable — shows the raw evidence text |
| **Inline PDF Viewer** | Highlighted PDF rendered below each plan card |
| **Full Screen Button** | Opens highlighted PDF in a new browser tab instantly |
| **Download Button** | Download highlighted PDF for each plan |
| **PDF Cache** | Highlighted PDFs cached in session — clear with sidebar button |

---

## 🚀 Running Locally

### Prerequisites
```bash
pip install -r requirements.txt
```

### Set your Groq API Key
```
Add to `Test2_Final.py`:

GROQ_API_KEY=gsk_your_key_here
```

### Run
```bash
streamlit run streamlit_app.py
```

App opens at `http://localhost:8501`


## 📦 Requirements

```
streamlit
sentence-transformers
faiss-cpu
rank-bm25
pymupdf
openai
numpy
python-dotenv
```

---

## 📁 Data Dependencies

The following files must be present (not committed to GitHub if large):

| File | Description |
|---|---|
| `Data-Engineering/data/rag_chunks.jsonl` | RAG chunks with plan text |
| `Data-Engineering/data/MASTER_STRUCTURED_SUPERSET_2026-1.jsonl` | Structured plan data with flags and numeric fields |
| `Data-Engineering/data/metadata.json` | Plan names and source URLs |
| `Data-Engineering/data/faiss_multi_provider_index.bin` | Pre-built FAISS index (commit this for fast startup) |
| `Data-Engineering/data/pdfs/` | Policy PDF files for inline highlighting |

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key for Llama-3.3-70B reranker — get free at [console.groq.com](https://console.groq.com) |

---

## 📄 PDF Highlighter

`pdf_highlighter.py` uses **PyMuPDF (fitz)** to highlight evidence chunks in policy PDFs.

- Splits each RAG chunk into short phrases (20–80 chars)
- Searches all pages when no page hint is available
- Highlights matched phrases in **yellow**
- Saves highlighted PDF as bytes for inline display and download

```python
from Frontend.pdf_highlighter import highlight_chunks

highlight_chunks(pdf_path, evidence_items, output_path)
```

---

## 🛠️ Key Configuration

In `streamlit_app.py`:

```python
# Path to your PDF folder
PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data-Engineering", "data", "pdfs")

# Engine import path — update if folder structure changes
from Policy.Main.Test2.Test2_Final import smart_search, extract_user_profile, CONDITION_MAP
```

---

## 💡 Supported Conditions

The profile extractor automatically detects these conditions from your query text:

`maternity` · `psychiatric` · `fertility` · `high_tech` · `cardiac` · `cancer` · `neurological` · `diabetes` · `orthopaedic` · `respiratory` · `physiotherapy` · `renal`

---

## 📝 Example Queries

```
"I'm 68 with Parkinson's. I see a neurologist monthly and take daily medication."

"I'm pregnant and looking for the best maternity cover with postnatal support."

"Heart condition, diabetic, hospitalised 3 times last year, 10 specialist visits."

"I'm 25 and healthy. Want the most affordable basic cover."
```
