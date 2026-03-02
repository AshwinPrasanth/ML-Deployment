# Run this

import json
import re
from collections import defaultdict

# ============================
# LOAD METADATA
# ============================

print("Loading metadata...")
with open("metadata.json", "r") as f:
    metadata = json.load(f)

metadata = [m for m in metadata if m["doc_type"] == "table_of_cover"]
print(f"Loaded {len(metadata)} table_of_cover documents")

# ============================
# MERGE TEXT PER PLAN
# ============================

documents = defaultdict(list)

for m in metadata:
    documents[m["doc_id"]].append(m["text"].lower())

merged_documents = {
    doc_id: " ".join(texts)
    for doc_id, texts in documents.items()
}

# ============================
# DISEASE RULES
# ============================

DISEASE_RULES = {
    "heart_disease": {
        "keywords": ["cardiac", "angioplasty", "stent"],
        "weight": 7,
        "requires_high_tech": True
    },
    "diabetes": {
        "keywords": ["diabetes", "insulin", "endocrinology"],
        "weight": 5
    },
    "cancer": {
        "keywords": ["oncology", "chemotherapy", "radiotherapy", "oncotype"],
        "weight": 9,
        "requires_high_tech": True
    },
    "psychiatric_disorder": {
        "keywords": ["psychiatric", "mental health"],
        "weight": 6
    },
    "neurological_disorder": {
        "keywords": ["neurology", "mri", "ct scan"],
        "weight": 7,
        "requires_high_tech": True
    },
    "orthopaedic_condition": {
        "keywords": ["orthopaedic", "joint replacement"],
        "weight": 6
    },
    "pregnancy": {
        "keywords": ["maternity", "obstetric"],
        "weight": 8
    }
}

# ============================
# FEATURE EXTRACTION
# ============================

def extract_plan_features(doc_id, text):

    features = {}
    features["doc_id"] = doc_id
    features["text"] = text

    # Inpatient coverage
    features["full_inpatient_cover"] = (
        "inpatient consultant fees" in text and "covered" in text
    )

    # Psychiatric days
    psych_matches = re.findall(r"(\d+)\s+days", text)
    features["psychiatric_days"] = max([int(x) for x in psych_matches], default=0)

    # Semi-private excess
    excess_match = re.search(r"€\s?(\d+)\s+excess", text)
    features["semi_private_excess"] = int(excess_match.group(1)) if excess_match else 0

    # Cardiac copay
    cardiac_match = re.search(
        r"€\s?(\d{2,5})\s*(?:co-?payment|copayment)[^\n]{0,40}cardiac",
        text
    )
    features["cardiac_copay"] = int(cardiac_match.group(1)) if cardiac_match else 0

    # ========================
    # HIGH-TECH PARSING
    # ========================

    hightech_match = re.search(
        r"high-tech hospital(.*?)(outpatient|day to day|members benefits|$)",
        text,
        re.DOTALL
    )

    hightech_text = hightech_match.group(1) if hightech_match else ""

    features["high_tech_percent"] = 0
    features["high_tech_restriction"] = 1.0
    features["high_tech_excess"] = 0
    features["high_tech_cardiac_copay"] = 0

    if hightech_text:

        percents = re.findall(r"(\d+)%\s*cover", hightech_text)

        if percents:
            features["high_tech_percent"] = sum(int(p) for p in percents) / len(percents)
        elif "covered" in hightech_text:
            features["high_tech_percent"] = 100

        if "beacon only" in hightech_text:
            features["high_tech_restriction"] = 0.4

        if "mater private" in hightech_text:
            features["high_tech_restriction"] = min(features["high_tech_restriction"], 0.6)

        ht_excess = re.search(r"€\s?(\d+)\s+excess", hightech_text)
        if ht_excess:
            features["high_tech_excess"] = int(ht_excess.group(1))

        ht_cardiac = re.search(r"€\s?(\d+)[^\n]{0,40}cardiac", hightech_text)
        if ht_cardiac:
            features["high_tech_cardiac_copay"] = int(ht_cardiac.group(1))

    return features


plans = [
    extract_plan_features(doc_id, text)
    for doc_id, text in merged_documents.items()
]

print("Extracted calibrated plan features.")

# ============================
# PATIENT PROFILE
# ============================

patient_profile = {
    "age": 52,
    "chronic_conditions": ["heart_disease", "diabetes"],
    "hospital_admissions_last_2_years": 2,
    "specialist_visits_per_year": 6
}

# ============================
# RISK CLASSIFICATION
# ============================

def compute_risk_score(profile):

    score = 0

    if profile["age"] > 60:
        score += 4
    elif profile["age"] > 45:
        score += 2

    score += len(profile["chronic_conditions"]) * 3
    score += profile["hospital_admissions_last_2_years"] * 2

    if profile["specialist_visits_per_year"] > 5:
        score += 2

    return score


def classify_risk(score):

    if score >= 12:
        return "high"
    elif score >= 6:
        return "medium"
    else:
        return "low"


risk_score = compute_risk_score(patient_profile)
risk_level = classify_risk(risk_score)

print("Risk Score:", risk_score)
print("Risk Level:", risk_level)

# ============================
# SCORING ENGINE
# ============================

def score_plan(plan, profile):

    score = 0

    # Risk scaling
    if risk_level == "high":
        hospital_weight = 1.5
        excess_weight = 1.0
        hightech_weight = 1.5
    elif risk_level == "medium":
        hospital_weight = 1.0
        excess_weight = 1.0
        hightech_weight = 1.0
    else:
        hospital_weight = 0.6
        excess_weight = 1.5
        hightech_weight = 0.6

    # Inpatient
    score += (5 if plan["full_inpatient_cover"] else -5) * hospital_weight

    # Psychiatric differentiation
    psych_boost = min(plan["psychiatric_days"], 150) / 25
    score += psych_boost

    # Excess penalty
    score -= (plan["semi_private_excess"] / 80) * excess_weight

    # Cardiac copay penalty
    score -= plan["cardiac_copay"] / 100

    # Disease logic
    for condition in profile["chronic_conditions"]:

        rules = DISEASE_RULES.get(condition)
        if not rules:
            continue

        for keyword in rules["keywords"]:
            if keyword in plan["text"]:
                score += rules["weight"]

        if rules.get("requires_high_tech"):

            ht_score = (
                (plan["high_tech_percent"] / 100) ** 1.5
            ) * 6

            ht_score *= plan["high_tech_restriction"]
            ht_score *= hightech_weight

            ht_score -= plan["high_tech_excess"] / 150
            ht_score -= plan["high_tech_cardiac_copay"] / 200

            score += ht_score

    # Over-insurance penalty for low risk
    if risk_level == "low":

        if plan["high_tech_percent"] > 80:
            score -= 2

        if plan["psychiatric_days"] > 120:
            score -= 2

    return round(score, 2)


ranked_plans = sorted(
    plans,
    key=lambda p: score_plan(p, patient_profile),
    reverse=True
)

# ============================
# OUTPUT
# ============================

print("\nTop Recommended Plans:\n")

for i, plan in enumerate(ranked_plans[:3], 1):
    print(f"{i}. {plan['doc_id']}")
    print(f"   Score: {score_plan(plan, patient_profile)}")
    print(f"   Excess: €{plan['semi_private_excess']}")
    print(f"   Cardiac Co-pay: €{plan['cardiac_copay']}")
    print(f"   High-Tech %: {plan['high_tech_percent']}")
    print(f"   High-Tech Restriction: {plan['high_tech_restriction']}")
    print(f"   Psychiatric Days: {plan['psychiatric_days']}")
    print("-" * 60)
