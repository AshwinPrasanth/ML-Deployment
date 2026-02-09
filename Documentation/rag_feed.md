# What You SHOULD Feed Into RAG?

## 1️⃣ Tables of Cover PDFs (Primary Source)

These are the most important documents.

They contain:
- Exact coverage clauses  
- Exclusions  
- Waiting periods  
- Coverage limits  
- Co-pay conditions  
- Hospital eligibility lists  

This is the **real policy logic**.

Your RAG system should retrieve chunks like:

> “Plan X covers outpatient consultant visits up to €Y per year with Z excess.”

That gives factual grounding and prevents hallucination.

---

## 2️⃣ Policy Terms & Conditions (Secondary Source)

These documents help with:

- Edge cases  
- Pre-existing condition exclusions  
- Chronic illness definitions  
- Maternity waiting periods  
- Specialist coverage rules  

This is critical for factual consistency evaluation and handling complex queries.

---

## 3️⃣ Structured Plan Metadata (NOT RAG — But Critical)

Webpage data (e.g., pricing, feature checkboxes) should **NOT** be embedded.

Instead, store it as structured data.

Example:

```json
{
  "plan_name": "First Cover",
  "monthly_price": 39.02,
  "hospital_cover_level": 1,
  "gp_cover": false,
  "digital_doctor": true
}
```

This structured layer is used for:

- Filtering
- Ranking
- Cost comparison
- Risk-based scoring

RAG should **not** be responsible for arithmetic or structured comparisons.

---

# Clean System Architecture

## Step 1: User Profile Input (Structured)

Collect structured inputs such as:

- Age  
- Chronic conditions  
- Medication frequency  
- Hospital visit frequency  
- Budget  
- Risk tolerance  

---

## Step 2: Structured Risk & Filtering Layer (Non-LLM)

Before invoking RAG:

- Score user risk
- Filter incompatible plans
- Rank by cost/benefit ratio

This demonstrates proper engineering separation between:
- Deterministic logic
- Retrieval
- Generation

---

## Step 3: RAG for Policy Justification

Use RAG to answer specific factual questions:

- “Does this plan cover insulin?”
- “Are psychiatric consultations covered?”
- “What waiting period applies?”
- “Are private hospitals included?”

Retrieve relevant policy chunks from:
- Tables of Cover
- Terms & Conditions

---

## Step 4: LLM Generates Explanation

Provide the LLM with:

- Structured plan summary
- Retrieved policy chunks
- User health profile

Then generate a grounded explanation:

> “Based on your Type 1 diabetes diagnosis requiring 4 GP visits per year and insulin therapy, Plan X is suitable because it covers outpatient consultations up to €800 annually and includes chronic medication reimbursement…”

Now the explanation is:
- Personalized
- Grounded
- Justifiable

---

# 🚨 What NOT To Do

Avoid:

- Embedding entire webpages blindly
- Asking GPT to “recommend a plan” without retrieval
- Skipping retrieval evaluation
- Mixing structured filtering and LLM reasoning

That leads to shallow system design and weaker evaluation marks.

---

# Evaluation Section Alignment

With this architecture, you can measure:

- Context Precision  
- Context Recall  
- Hallucination Rate  
- Instruction Following  
- Latency (TTFT, TPOT)  

Now your system aligns with practical evaluation requirements.

---

# For Distinction-Level Work

Run controlled comparisons:

- Pure prompting vs RAG  
- Embedding retrieval vs BM25  
- Hybrid retrieval  
- With reranker vs without  

Now you are evaluating a real ML system — not just prompting.

---

# ✅ Summary

- Use PDFs for RAG grounding.
- Use structured metadata for filtering and ranking.
- Separate logic layers cleanly.
- Evaluate retrieval quality rigorously.
- Compare retrieval strategies.

That’s production-grade architecture.
