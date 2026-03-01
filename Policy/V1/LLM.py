import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# =====================================================
# CONFIG
# =====================================================

TOP_K_PLANS = 3
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# =====================================================
# RISK LEVEL LABEL
# =====================================================

def get_risk_label(risk_score):
    if risk_score >= 14:
        return "HIGH"
    elif risk_score >= 8:
        return "MEDIUM"
    else:
        return "LOW"

risk_level = get_risk_label(risk_score)

# =====================================================
# RANK PLANS
# =====================================================

ranked_plans = sorted(
    plans,
    key=lambda p: score_plan(p, patient_profile),
    reverse=True
)

top_plans = ranked_plans[:TOP_K_PLANS]

# =====================================================
# STRUCTURED EVIDENCE BUILDER
# =====================================================

def build_structured_evidence(plan, profile):

    evidence = {}

    text = plan.get("text", "")

    # Inpatient
    if plan.get("full_inpatient", False):
        evidence["inpatient"] = "Inpatient consultant fees and scans are covered."
    else:
        evidence["inpatient"] = "Inpatient consultant fees are not fully covered."

    # Excess
    excess = plan.get("excess", 0)
    evidence["excess"] = f"Plan excess is €{excess}."

    # Disease relevance
    disease_evidence = []

    for condition in profile["chronic_conditions"]:
        rules = DISEASE_RULES.get(condition)
        if not rules:
            continue

        matched = False
        for kw in rules["keywords"]:
            if kw in text:
                disease_evidence.append(
                    f"Mentions '{kw}' related to {condition}."
                )
                matched = True

        if not matched:
            disease_evidence.append(
                f"No explicit mention of {condition} coverage."
            )

    evidence["disease"] = disease_evidence

    # High-tech
    if plan.get("high_tech", False):
        evidence["hightech"] = "High-tech hospital coverage available."
    else:
        evidence["hightech"] = "High-tech hospital coverage not available or restricted."

    return evidence

# =====================================================
# LOAD MODEL (4-bit)
# =====================================================

print("Loading LLM...")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Highly recommended for better quality
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto" # Automatically maps layers to GPU/CPU
)

print(f"LLM loaded on {model.device}.")

# =====================================================
# EXPLANATION GENERATOR
# =====================================================

def compute_score_breakdown(plan, profile):
  for i, plan in enumerate(ranked_plans[:3], 1):
    return(f"{i}. {plan['doc_id']} Score: {score_plan(plan, patient_profile)} Excess: €{plan['semi_private_excess']} Cardiac Co-pay: €{plan['cardiac_copay']}   High-Tech %: {plan['high_tech_percent']}   High-Tech Restriction: {plan['high_tech_restriction']}  Psychiatric Days: {plan['psychiatric_days']}")

def generate_explanation(rank, plan, next_plan=None):

    total_score= score_plan(plan, patient_profile)
    breakdown=compute_score_breakdown(plan, patient_profile)
    evidence = build_structured_evidence(plan, patient_profile)

    comparison_line = ""
    if next_plan:
        comparison_line = (
            f"This plan ranked above {next_plan['doc_id']} "
            f"because it achieved a higher total score."
        )

    prompt = f"""
You are a medical insurance ranking explanation engine.

RULES:
- Use ONLY the structured evidence provided.
- Do NOT invent benefits.
- Connect patient conditions to coverage.
- Explain ranking logically.
- Be concise and factual.
- Max 180 words.

PATIENT PROFILE:
{json.dumps(patient_profile)}

RISK LEVEL: {risk_level}

PLAN RANK: #{rank}
PLAN ID: {plan["doc_id"]}
TOTAL SCORE: {total_score}

SCORE BREAKDOWN:
{json.dumps(breakdown)}

STRUCTURED EVIDENCE:
{json.dumps(evidence, indent=2)}

TASK:
Explain why this plan ranked #{rank}.
Link:
- inpatient component
- excess penalty
- disease match
- high-tech impact
Mention limitations clearly.
{comparison_line}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=220,
        temperature=0.0,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =====================================================
# RUN EXPLANATIONS
# =====================================================

print("\n==============================")
print("DECISION-AWARE EXPLANATIONS")
print("==============================")

for i, plan in enumerate(top_plans):

    next_plan = top_plans[i+1] if i+1 < len(top_plans) else None

    explanation = generate_explanation(
        rank=i+1,
        plan=plan,
        next_plan=next_plan
    )

    print("\n====================================")
    print("PLAN:", plan["doc_id"])
    print("====================================")
    print(explanation)
