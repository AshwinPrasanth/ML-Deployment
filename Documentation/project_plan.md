# Intelligent Insurance Recommendation System
## ML Deployment Architecture & Evaluation Framework

---

# 1️⃣ Problem Definition (Precise Use Case)

## Target User

Young adult (18–35) recently diagnosed with **Type 1 Diabetes** living in Ireland, seeking private health insurance with:

- Monthly budget under €X
- Need for regular GP visits
- Insulin therapy
- Possible specialist consultations
- Risk of hospitalization
- Concern about waiting periods and exclusions

---

## Core Constraints

The system must account for:

- Budget limits
- Pre-existing condition exclusions
- Waiting periods
- Co-pay limits
- Chronic illness coverage
- Hospital eligibility lists

---

## Problem Statement

> Design a document-grounded, explainable insurance recommendation system that suggests suitable private insurance plans for a young adult with Type 1 Diabetes in Ireland under budget constraints, while explicitly justifying coverage decisions using policy clauses.

Specificity improves justification quality and evaluation strength.

---

# 2️⃣ System Architecture

## 🏗 Application Layer

- Patient Profile Input Form
- Recommendation Dashboard
- Explanation Panel (policy-grounded reasoning)
- Human Review Button

---

## 🤖 Model Layer

### 1. Risk Assessment Module (Structured)
- Compute patient risk vector:
  - Chronic condition severity
  - Treatment frequency
  - Medication class
  - Hospitalization likelihood (heuristic-based)

### 2. Recommendation Engine
- Structured filtering
- Cost-benefit ranking
- Rule-based exclusions

### 3. Policy Retrieval (RAG)
- Retrieve relevant clauses from:
  - Tables of Cover PDFs
  - Terms & Conditions PDFs

### 4. Explanation Generator (LLM)
Inputs:
- Structured risk summary
- Retrieved policy clauses
- Plan metadata

Outputs:
- Grounded, personalized explanation

### 5. AI-as-Judge Evaluator
Evaluates:
- Clause citation correctness
- Alignment with retrieved context
- Coverage-exclusion consistency

---

## 🧱 Infrastructure Layer

- Vector Database (e.g., FAISS)
- Document Chunking Pipeline
- Logging & Tracing
- Latency Measurement (TTFT, TPOT)
- Caching Layer
- Model Gateway / API Wrapper

This transforms the system from an “LLM demo” into an ML deployment pipeline.

---

# 3️⃣ RAG is Mandatory (Not Optional)

Healthcare insurance is document-heavy → ideal RAG use case.

## Data Collection

- 10–20 real insurance policy PDFs
- Tables of Cover
- Terms & Conditions

## Processing

- Clause extraction
- Chunking (semantic or sliding window)
- Embedding generation
- Indexing in vector DB

---

## Retrieval Experiments

Compare:

- BM25 retrieval
- Embedding-based retrieval
- Hybrid retrieval (BM25 + embeddings)
- With reranker vs without

This enables:

- Context precision measurement
- Context recall measurement
- Retrieval benchmarking

This is high-value technical evaluation.

---

# 4️⃣ Risk Scoring Component (Structured Layer)

Instead of full LLM reasoning:

1. Collect structured patient data.
2. Compute risk vector:
   - Chronic severity score
   - Medication burden
   - Specialist frequency
   - Hospitalization probability (rule-based)

3. Filter incompatible plans.
4. Rank remaining plans by:
   - Coverage depth
   - Expected annual cost
   - Exclusion risk

LLM is used only for explanation and justification.

This shows architectural maturity.

---

# 5️⃣ Evaluation Pipeline

## A. RAG Metrics

- Context Precision
- Context Recall
- Hallucination Rate

---

## B. Explanation Quality

AI-as-a-Judge evaluates:

- Does the explanation cite correct clauses?
- Are exclusions mentioned?
- Are waiting periods acknowledged?
- Is reasoning aligned with retrieved context?

---

## C. System Metrics

Track:

- TTFT (Time to First Token)
- TPOT (Time per Output Token)
- Throughput
- Latency under load

Log metrics across iterations to demonstrate improvement.

---

# 6️⃣ Human-in-the-Loop

Concrete implementation:

- Insurance expert review mode
- Manual override of recommendations
- Feedback storage for evaluation
- Flagging incorrect justifications

Even simulated expert review demonstrates responsible AI design.

---

# 🚨 Safety & Risk Handling

Because this is healthcare-related:

Include:

- Decision-support disclaimer
- Bias analysis
- Guardrails against unsupported claims
- Handling incomplete patient profiles
- Explicit statement of uncertainty

Ignoring safety weakens credibility.

---

# 📈 Iteration Roadmap

## Week 4
- Basic RAG
- Clause-grounded explanation

## Week 6
- Hybrid retrieval
- Context precision/recall evaluation

## Week 8
- Add caching
- Implement latency tracking

## Week 10
- AI-as-judge evaluation pipeline
- Full system benchmarking

This structured progression demonstrates engineering evolution.

---

# ✅ Final Outcome

By the final iteration, the system will include:

- Structured risk filtering
- Document-grounded retrieval
- Hybrid search benchmarking
- Latency instrumentation
- AI-based evaluation
- Human oversight loop
- Safety documentation


