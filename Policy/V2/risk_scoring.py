import json
import re
import numpy as np
import faiss
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# ======================================================
# CONFIG
# ======================================================

INDEX_PATH = "faiss_index_v2.bin"
METADATA_PATH = "metadata_v2.json"

BETA = 0.6           # Risk weight
FINAL_TOP_K = 3

# ======================================================
# LOAD MODEL + INDEX
# ======================================================

print("Loading MXBAI model...")
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

plan_ids = list(set(m["doc_id"] for m in metadata))
print(f"Loaded {len(plan_ids)} unique plans")

# ======================================================
# DENSE RELEVANCE (Chunk → Plan)
# ======================================================

def compute_dense_scores(query):

    prefixed_query = "Represent this sentence for searching relevant passages: " + query

    q_emb = model.encode([prefixed_query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")

    dense_scores, dense_indices = index.search(q_emb, 50)

    dense_plan_scores = defaultdict(list)

    for score, idx in zip(dense_scores[0], dense_indices[0]):
        plan_id = metadata[idx]["doc_id"]
        dense_plan_scores[plan_id].append(score)

    dense_plan_scores = {
        plan: np.mean(scores)
        for plan, scores in dense_plan_scores.items()
    }

    # Normalize 0–1
    if dense_plan_scores:
        vals = np.array(list(dense_plan_scores.values()))
        min_v, max_v = vals.min(), vals.max()

        dense_plan_scores = {
            k: (v - min_v) / (max_v - min_v + 1e-8)
            for k, v in dense_plan_scores.items()
        }

    return dense_plan_scores

# ======================================================
# DENSE RETRIEVAL HELPER (Plan-Specific)
# ======================================================

def retrieve_plan_chunks(plan_id, query, top_n=5):

    prefixed_query = "Represent this sentence for searching relevant passages: " + query

    q_emb = model.encode([prefixed_query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")

    scores, indices = index.search(q_emb, 50)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if metadata[idx]["doc_id"] == plan_id:
            results.append((score, metadata[idx]["chunk_text"]))
            if len(results) >= top_n:
                break

    return results

# ======================================================
# DISEASE RULES
# ======================================================

DISEASE_RULES = {
    "heart_disease": {
        "keywords": ["cardiac", "angioplasty", "stent"],
        "weight": 8,
        "requires_high_tech": True
    },
    "diabetes": {
        "keywords": ["diabetes", "insulin"],
        "weight": 5
    },
    "cancer": {
        "keywords": ["oncology", "chemotherapy", "radiotherapy"],
        "weight": 10,
        "requires_high_tech": True
    },
    "psychiatric_disorder": {
        "keywords": ["psychiatric", "mental health"],
        "weight": 7,
        "requires_psych_days": True
    },
    "neurological_disorder": {
        "keywords": ["neurology", "mri", "ct scan"],
        "weight": 8,
        "requires_high_tech": True
    },
    "pregnancy": {
        "keywords": ["maternity", "obstetric"],
        "weight": 8
    }
}

# ======================================================
# STRUCTURED FEATURE EXTRACTION
# ======================================================

def merge_plan_text(plan_id):
    return " ".join(
        m["chunk_text"].lower()
        for m in metadata
        if m["doc_id"] == plan_id
    )

def extract_features(text):

    features = {}

    features["full_inpatient"] = (
        "inpatient consultant fees" in text and "fully covered" in text
    )

    excess_match = re.search(r"€\s?(\d+)\s+excess", text)
    features["excess"] = int(excess_match.group(1)) if excess_match else 0

    copay_match = re.search(r"€\s?(\d+)[^\n]{0,50}cardiac", text)
    features["cardiac_copay"] = int(copay_match.group(1)) if copay_match else 0

    psych_days = re.findall(r"(\d+)\s+days", text)
    features["psychiatric_days"] = max([int(x) for x in psych_days], default=0)

    features["high_tech_available"] = (
        "high-tech hospital" in text and "not covered" not in text
    )

    return features

# ======================================================
# DENSE-GUIDED DISEASE SCORING
# ======================================================

def dense_disease_score(plan_id, condition):

    rules = DISEASE_RULES.get(condition)
    if not rules:
        return 0

    query = f"Coverage details for {condition.replace('_',' ')} including hospital and treatment"

    chunks = retrieve_plan_chunks(plan_id, query)

    score = 0

    for _, chunk in chunks:
        chunk = chunk.lower()

        for kw in rules["keywords"]:
            if kw in chunk:
                score += rules["weight"]

        if rules.get("requires_high_tech"):
            if "high-tech" in chunk and "not covered" not in chunk:
                score += 6
            elif "not covered" in chunk:
                score -= 8

        if rules.get("requires_psych_days"):
            days = re.findall(r"(\d+)\s+days", chunk)
            if days:
                score += min(int(max(days)), 150) / 20

    return score

# ======================================================
# FINAL RISK SCORING
# ======================================================

def compute_risk_score(plan_id, profile):

    text = merge_plan_text(plan_id)
    f = extract_features(text)

    score = 0
    breakdown = {}

    inpatient = 8 if f["full_inpatient"] else -8
    score += inpatient
    breakdown["inpatient"] = inpatient

    excess_penalty = f["excess"] / 60
    score -= excess_penalty
    breakdown["excess_penalty"] = -round(excess_penalty, 2)

    cardiac_penalty = 0
    if "heart_disease" in profile["chronic_conditions"]:
        cardiac_penalty = f["cardiac_copay"] / 40
        score -= cardiac_penalty
    breakdown["cardiac_penalty"] = -round(cardiac_penalty, 2)

    disease_total = 0
    for condition in profile["chronic_conditions"]:
        disease_total += dense_disease_score(plan_id, condition)

    score += disease_total
    breakdown["dense_disease_score"] = round(disease_total, 2)

    return score, breakdown

# ======================================================
# MULTI-SCENARIO EVALUATION
# ======================================================

SCENARIOS = {
    "Cardiac + Diabetes": {
        "profile": {
            "age": 55,
            "chronic_conditions": ["heart_disease", "diabetes"]
        },
        "query": """
        Cardiac procedures, insulin support,
        frequent inpatient admission,
        consultant coverage, high-tech hospital.
        """
    },

    "Cancer + Neurological": {
        "profile": {
            "age": 60,
            "chronic_conditions": ["cancer", "neurological_disorder"]
        },
        "query": """
        Oncology treatment, chemotherapy,
        MRI scans, neurological admission,
        high-tech hospital access.
        """
    },

    "Psychiatric + Diabetes": {
        "profile": {
            "age": 40,
            "chronic_conditions": ["psychiatric_disorder", "diabetes"]
        },
        "query": """
        Psychiatric admission days,
        mental health coverage,
        diabetes management.
        """
    },

    "Pregnancy Case": {
        "profile": {
            "age": 30,
            "chronic_conditions": ["pregnancy"]
        },
        "query": """
        Maternity cover,
        obstetric services,
        hospital delivery,
        minimal excess.
        """
    }
}

print("\n====================================")
print("AUTOMATED MULTI-SCENARIO EVALUATION (DENSE ONLY)")
print("====================================")

for scenario_name, scenario_data in SCENARIOS.items():

    print(f"\n==============================")
    print(f"Scenario: {scenario_name}")
    print("==============================")

    patient_profile = scenario_data["profile"]
    query = scenario_data["query"]

    dense_scores = compute_dense_scores(query)

    results = []

    for plan_id in plan_ids:

        risk_score, breakdown = compute_risk_score(plan_id, patient_profile)

        final_score = (BETA * risk_score) + dense_scores.get(plan_id, 0)

        results.append((plan_id, final_score, risk_score,
                        dense_scores.get(plan_id, 0), breakdown))

    results.sort(key=lambda x: x[1], reverse=True)

    for rank, (doc_id, final_score, risk_score,
               dense_score, breakdown) in enumerate(results[:FINAL_TOP_K], 1):

        print(f"\n{rank}. {doc_id}")
        print(f"   Final Score: {round(final_score,2)}")
        print(f"   Risk Score: {round(risk_score,2)}")
        print(f"   Dense Score: {round(dense_score,3)}")

        print("   Breakdown:")
        for k, v in breakdown.items():
            print(f"     {k}: {v}")

    margin = results[0][1] - results[1][1]

    print("\nConfidence Margin:", round(margin, 3))
