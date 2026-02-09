# Data Strategy for Distinction-Level ML Deployment

## Real Policy PDF Curation (Recommended Approach)

The optimal balance between realism, legality, and academic value is:

> Semi-manual curation of official insurance policy PDFs from 2–3 Irish insurers.

No scraping.
No synthetic data.
No over-engineered crawling infrastructure.

Just:

- Download official “Table of Cover” PDFs
- Download Terms & Conditions documents
- Download benefit comparison sheets
- Document source URLs clearly in the report

This approach maximizes academic value while remaining manageable.

---

# ✅ Why This Is the Best Option

## Academic Strength

This enables:

- Real-world grounding
- Legitimate RAG use case
- Proper dataset engineering section
- Chunking strategy experimentation
- Retrieval comparison experiments
- Realistic hallucination evaluation
- Strong evaluation pipeline

You can confidently write:

> “We curated 27 official policy documents from Irish insurers to construct a domain-specific knowledge base.”

That immediately signals seriousness.

---

# 🚫 Why NOT Scrape Aggressively?

Scraping dynamic pricing pages:

- May violate terms of service
- Requires handling JavaScript rendering and session tokens
- Adds infrastructure complexity
- Provides little ML evaluation benefit

This module is about **ML system deployment**, not web scraping engineering.

Scraping = high effort, low grading payoff.

---

# 🚫 Why NOT Fully Simulate Policies?

If policies are synthetic:

- RAG evaluation becomes artificial
- Hallucination risks cannot be realistically tested
- The use case appears trivial
- The project feels like a demo rather than a deployment system

Healthcare insurance requires authentic documentation.

---

# 🧱 Phase 1 — Real Data Collection

## Data to Collect

From 2–3 insurers:

- 5–10 plan PDFs per insurer
- Terms & Conditions documents
- Tables of Cover
- Benefit comparison sheets

All manually downloaded.
All source links documented.

---

# 1️⃣ Dataset Engineering Section (4–5 Pages)

This is where you gain marks.

Document:

- PDF parsing method
- Text extraction pipeline
- Cleaning strategy
- Deduplication
- Section segmentation
- Metadata tagging

Metadata examples:

- Plan name
- Benefit category
- Coverage type
- Waiting period section
- Exclusion section

---

## Chunking Experiments

Compare:

- 256 token chunks
- 512 token chunks
- Overlap vs no-overlap
- Section-aware chunking

Report:

- Retrieval quality differences
- Context precision changes
- Latency differences

This directly satisfies:

> Dataset Engineering: Documentation of data curation, cleaning, and deduplication.

---

# 2️⃣ RAG Architecture

Implement and compare:

- BM25 retrieval
- Embedding-based retrieval
- Hybrid retrieval (BM25 + embedding fusion)
- With reranker vs without reranker

Evaluate:

- Retrieval accuracy
- Context precision
- Context recall
- Grounding strength

This matches:

> Specify retrieval algorithm (BM25 vs embedding-based).

---

# 3️⃣ Adaptation Strategy

Include:

- Few-shot prompting
- Structured explanation template
- Controlled reasoning format

Optional (if time allows):

- Lightweight LoRA fine-tuning for explanation style

This satisfies:

> Prompt engineering or PEFT/LoRA fine-tuning details.

---

# 4️⃣ Evaluation Pipeline (Critical Section)

Most groups underperform here.

Implement:

## A. RAG Metrics
- Context Precision
- Context Recall
- Hallucination Rate

## B. Explanation Quality

Use AI-as-a-Judge with custom rubric:

- Does explanation cite correct clauses?
- Are exclusions mentioned?
- Are waiting periods acknowledged?
- Is reasoning aligned with retrieved context?

## C. System Metrics
- TTFT (Time to First Token)
- TPOT (Time per Output Token)
- Throughput
- Retrieval latency

This satisfies:

> Systematic pipeline using AI-as-a-judge with custom rubrics.

---

# 5️⃣ Deployment & Optimisation

Include:

- Prompt caching
- Batch embedding
- Quantisation (if applicable)
- Logging & tracing
- Observability dashboard

Measure:

- Latency improvements
- Cost efficiency
- Scaling behavior

This matches:

> Quantisation, batching, prompt caching, TTFT, TPOT, observability.

---

# 📊 Final Recommendation Summary

## Do This

- Manually curate real PDFs
- Build structured metadata store
- Implement full RAG pipeline
- Compare retrieval strategies
- Implement evaluation framework
- Track system performance

## Avoid This

- Aggressive dynamic scraping
- Synthetic policy simulation
- Pure prompting without retrieval

---

# 🎯 Why This Wins Distinction

This approach demonstrates:

- Technical depth
- Real-world relevance
- Retrieval benchmarking
- Structured reasoning architecture
- Responsible AI safeguards
- Measurable deployment metrics

The workload remains manageable.
The evaluation becomes rigorous.
The report writes itself.
The grading becomes predictable.

This is how you turn a project into a deployment system.
