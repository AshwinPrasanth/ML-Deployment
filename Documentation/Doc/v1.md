# Hybrid Risk-Aware Health Insurance Recommendation System

This project implements a complete 6-phase multi-stage ranking pipeline for personalized health insurance recommendation.

The system combines:

- Lexical retrieval (BM25)
- Dense semantic retrieval (FAISS + MiniLM)
- Hybrid fusion scoring
- Structured medical risk modeling
- Multi-disease conditional reasoning
- Interpretable final ranking

All six phases are implemented (Version 1).

---

# System Architecture

Phase 1 → Plan Ingestion & Normalization  
Phase 2 → Dense Embedding Index (FAISS)  
Phase 3 → BM25 Baseline Retrieval  
Phase 4 → Hybrid Retrieval Fusion  
Phase 5 → Risk-Aware Medical Scoring  
Phase 6 → Full Fusion Ranking + Scenario Evaluation  

Each phase adds a distinct layer of intelligence and personalization.

---

## Phase 1 — Plan Ingestion & Normalization

- Load `chunk_metadata.json`
- Filter `doc_type == table_of_cover`
- Merge all chunks by `doc_id`
- Normalize to lowercase
- Build:
  - `plan_ids`
  - `corpus` (one full text per plan)

This eliminates chunk-level fragmentation bias and ensures plan-level ranking.

---

## Phase 2 — Dense Embedding Index (FAISS)

Model: `all-MiniLM-L6-v2`

- Encode full plan texts
- L2 normalize embeddings
- Store in `faiss.IndexFlatIP`

Enables semantic matching:
- "oncology treatment" ≈ "chemotherapy"
- "MRI scan" ≈ "neurological imaging"

---

## Phase 3 — BM25 Lexical Retrieval

- Regex-based tokenizer
- BM25Okapi scoring
- Exact token and numeric matching

Preserves precision for:
- Euro amounts
- Day limits
- Excess values
- Specific co-pay conditions

---

## Phase 4 — Hybrid Retrieval Fusion

Hybrid scoring:

hybrid_score = α * dense_norm + (1 - α) * bm25_norm

Where:
- α ∈ [0.4, 0.6]
- Scores are normalized before fusion

Benefits:
- Maintains semantic robustness
- Preserves numeric precision
- Improves retrieval stability

Hybrid relevance is computed globally over all plans.

---

## Phase 5 — Risk-Aware Medical Scoring

Structured personalization layer.

### Feature Extraction (Regex-based)

Extract from plan text:
- Inpatient consultant coverage
- Semi-private excess
- Cardiac co-pay
- Psychiatric day allowance
- High-tech hospital availability

### Disease Rule Engine

Each condition has explicit rules:

Example:
- Cancer → requires high-tech hospital access
- Cardiac disease → penalize high co-pay
- Psychiatric conditions → require sufficient psychiatric days

Rules are conditional:
- No cardiac penalty without heart disease
- No psychiatric boost without psychiatric condition
- High-tech bonus only when clinically relevant

### Risk Score Formula

risk_score =
    inpatient_component
  - excess_penalty
  - condition_specific_penalties
  + disease_match_weight
  + hightech_bonus_or_penalty
  + psychiatric_component

All components are numeric and interpretable.

---

## Phase 6 — Final Fusion Ranking

Final decision score:

final_score = (BETA * risk_score) + hybrid_score

Where:
- BETA controls personalization strength

This creates a multi-objective ranking:
- Hybrid score → contextual relevance
- Risk score → medical suitability

---

# Scenario Evaluation (Implemented)

System evaluates multiple patient profiles:

- Cardiac + Diabetes
- Cancer + Neurological
- Psychiatric + Diabetes
- Pregnancy
- Low-risk baseline

For each scenario:
- Compute hybrid relevance
- Compute structured risk score
- Fuse scores
- Rank all plans globally
- Output component breakdown

---

# Interpretability by Design

Each ranked plan outputs:

- Final score
- Hybrid score
- Risk score
- Component breakdown:
  - inpatient
  - excess_penalty
  - cardiac_penalty
  - disease_match
  - hightech
  - psychiatric

No black-box neural ranking.

All scores are decomposable and auditable.

---

# What This System Demonstrates

- Multi-stage IR architecture
- Semantic + lexical fusion
- Conditional disease-aware logic
- Multi-objective decision modeling
- Global ranking without retrieval bias
- Fully interpretable scoring pipeline

This is not:
- A simple RAG system
- A keyword matcher
- A cost-only heuristic

It is:
- A structured hybrid retrieval + medical reasoning engine
- A multi-disease decision-aware ranking framework

