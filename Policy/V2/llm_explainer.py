import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TOP_K_PLANS = 3
MAX_NEW_TOKENS = 180

# =====================================================
# STEP 1 — PREPARE RANKED RESULTS
# =====================================================

dense_scores = compute_dense_scores(query)

ranked_results = []

for plan_id in plan_ids:
    risk_score, breakdown = compute_risk_score(plan_id, patient_profile)
    final_score = (BETA * risk_score) + dense_scores.get(plan_id, 0)

    ranked_results.append({
        "doc_id": plan_id,
        "final_score": final_score,
        "risk_score": risk_score,
        "dense_score": dense_scores.get(plan_id, 0),
        "breakdown": breakdown
    })

ranked_results.sort(key=lambda x: x["final_score"], reverse=True)

top_plans = ranked_results[:TOP_K_PLANS]

# =====================================================
# STEP 2 — BUILD STRUCTURED + DENSE EVIDENCE
# =====================================================

def build_dense_evidence(plan_id, profile):

    text = merge_plan_text(plan_id)
    features = extract_features(text)

    evidence = {
        "full_inpatient": features["full_inpatient"],
        "excess": features["excess"],
        "cardiac_copay": features["cardiac_copay"],
        "high_tech_available": features["high_tech_available"],
        "psychiatric_days": features["psychiatric_days"]
    }

    # Dense chunk evidence per condition
    retrieved_chunks = {}

    for condition in profile["chronic_conditions"]:
        condition_query = f"Coverage details for {condition.replace('_',' ')}"
        chunks = retrieve_plan_chunks(plan_id, condition_query, top_n=2)

        retrieved_chunks[condition] = [
            chunk_text[:200] for _, chunk_text in chunks
        ]

    evidence["retrieved_chunks"] = retrieved_chunks

    return evidence

# =====================================================
# STEP 3 — LOAD LLM (SAFE FP16 VERSION)
# =====================================================

print("\nLoading Explanation LLM...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

llm_model.eval()

print("LLM loaded.")

# =====================================================
# STEP 4 — EXPLANATION GENERATOR
# =====================================================

def generate_explanation(rank_data, next_plan=None):

    evidence = build_dense_evidence(rank_data["doc_id"], patient_profile)

    comparison_text = ""
    if next_plan:
        score_diff = round(
            rank_data["final_score"] - next_plan["final_score"], 2
        )

        comparison_text = f"""
Comparison With Next Ranked Plan:
Next Plan ID: {next_plan['doc_id']}
Score Difference: {score_diff}
Next Plan Risk Score: {round(next_plan['risk_score'],2)}
Next Plan Dense Score: {round(next_plan['dense_score'],3)}
"""

    prompt = f"""
You are a medical insurance ranking explanation engine.

IMPORTANT RULES:
- Use ONLY the structured evidence provided.
- Do NOT invent benefits.
- Do NOT assume coverage beyond evidence.
- Explain ranking strictly according to the scoring model.
- Mention inpatient impact, excess penalty, disease alignment, and dense relevance.
- Clarify that ranking is model-based.
- Maximum 170 words.

PATIENT PROFILE:
{json.dumps(patient_profile)}

PLAN ID: {rank_data['doc_id']}
FINAL SCORE: {round(rank_data['final_score'],2)}
RISK SCORE: {round(rank_data['risk_score'],2)}
DENSE SCORE: {round(rank_data['dense_score'],3)}

SCORE BREAKDOWN:
{json.dumps(rank_data['breakdown'], indent=2)}

EVIDENCE:
{json.dumps(evidence, indent=2)}

{comparison_text}

Explain why this plan ranked at this position.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(llm_model.device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt echo if present
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]

    return decoded.strip()

# =====================================================
# STEP 5 — RUN EXPLANATIONS
# =====================================================

print("\n====================================")
print("DENSE-AWARE COMPARATIVE EXPLANATIONS")
print("====================================")

for i, plan_data in enumerate(top_plans):

    next_plan = top_plans[i+1] if i+1 < len(top_plans) else None

    explanation = generate_explanation(plan_data, next_plan)

    print("\n====================================")
    print("PLAN:", plan_data["doc_id"])
    print("====================================")
    print(explanation)
