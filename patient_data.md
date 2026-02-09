# Patient-Side Architecture: Structured Risk Profiling Layer

## Design Principle

Do NOT allow free-text inputs like:

> “I have diabetes and sometimes go to hospital.”

Free text:
- Is vague
- Is hard to evaluate
- Makes filtering nondeterministic
- Pushes too much responsibility to the LLM

Instead, build a structured intake system.

---

# 🏗 Step 1 — Patient Profile Schema

## A. Basic Demographics

- Age
- Employment status
- Monthly insurance budget
- Family coverage required? (Y/N)

---

## B. Medical Profile

- Chronic conditions (multi-select)
- Medication frequency (daily / weekly / monthly)
- GP visits per year (numeric)
- Specialist visits per year
- Hospital admissions (last 2 years)
- Planned procedures? (Y/N)

---

## C. Risk Sensitivity

- Preference: low premium vs high coverage
- Risk tolerance level (1–5 scale)

---

## Why This Matters

Structured inputs create:

- Evaluatable features
- Deterministic filtering logic
- Clean architecture separation
- Reproducible experiments

This directly strengthens your evaluation section.

---

# 🎯 Step 2 — Risk Scoring Module (Non-LLM Layer)

Do NOT let the LLM “infer” risk.

Instead, compute a structured risk score.

## Example Risk Formula

```
risk_score =
  chronic_condition_weight +
  medication_frequency_weight +
  hospital_visit_weight +
  age_weight
```

You can define risk tiers:

- Low Risk
- Medium Risk
- High Risk

This converts:

Structured patient → Risk vector → Plan filtering

This demonstrates system design maturity.

---

# 🔎 Step 3 — Matching Engine (Pre-RAG Filtering)

Before calling RAG:

Filter out plans that:

- Exclude pre-existing conditions
- Have insufficient outpatient coverage
- Exceed budget constraints

Then rank remaining plans by:

- Cost-to-coverage ratio
- Risk coverage adequacy score
- Waiting period penalties

This stage is deterministic.

LLM is NOT involved here.

---

# 🧾 Step 4 — RAG for Justification (Post-Selection)

Only after selecting top 2–3 candidate plans:

1. Retrieve relevant policy clauses
2. Generate grounded explanation

The LLM is responsible for:

- Explanation
- Clause citation
- Transparency
- Personalized summary

It is NOT responsible for:
- Budget filtering
- Risk scoring
- Eligibility logic

This separation is critical for grading.

---

# 🛡 Step 5 — Safety + Human-in-the-Loop

Because this is healthcare-related:

Include:

- Decision-support disclaimer banner
- Confidence score output
- “Request Expert Review” button
- Logging of override decisions

In the report:

> “The system maintains human oversight for high-risk recommendations.”

This signals responsible AI deployment.

---

# 📊 Evaluation Benefits

This layered design enables structured evaluation.

## 1️⃣ Recommendation Quality

- Budget compliance rate
- Risk threshold satisfaction
- Coverage adequacy score

---

## 2️⃣ Explanation Faithfulness

- Are retrieved clauses cited?
- Hallucination rate
- Alignment between retrieval and explanation

---

## 3️⃣ Edge Case Testing

Create synthetic patient profiles:

- High-risk chronic condition
- Low-risk healthy adult
- Budget-constrained user
- Maternity case
- Specialist-heavy case

Evaluate system consistency across scenarios.

---

# 🧬 Optional: Large-Scale Synthetic Testing

Generate 50–100 synthetic patient profiles.

Evaluate:

- Stability of recommendations
- Retrieval accuracy per condition category
- Distribution of risk tiers
- Explanation consistency

This transforms testing into systematic evaluation.

---

# ⚠ Core Design Principle

Patient data should NOT be fed directly into RAG.

Correct pipeline:

Patient → Structured Risk Layer → Plan Shortlist  
THEN  
RAG → Retrieve Clauses → Explain

This layered design should be clearly illustrated in your system architecture diagram.

---

# 🎓 Final Architecture Flow

User Input (Structured Form)  
↓  
Risk Scoring Module  
↓  
Plan Filtering & Ranking  
↓  
RAG (Policy Retrieval)  
↓  
LLM Explanation  
↓  
Evaluation + Logging  

This is a proper ML deployment system — not a chatbot demo.
