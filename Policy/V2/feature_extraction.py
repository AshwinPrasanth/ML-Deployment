import json
import numpy as np
import faiss
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# ----------------------------
# CONFIG
# ----------------------------

INDEX_PATH = "faiss_index_v2.bin"
METADATA_PATH = "metadata_v2.json"
TOP_K = 3

# ----------------------------
# LOAD MODEL + INDEX
# ----------------------------

print("Loading embedding model...")
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# ----------------------------
# SEARCH FUNCTION (Chunk → Plan Aggregation)
# ----------------------------

def search(query, k=20):  # retrieve more chunks first
    prefixed_query = "Represent this sentence for searching relevant passages: " + query

    query_embedding = model.encode(
        [prefixed_query],
        normalize_embeddings=True
    )

    query_embedding = np.array(query_embedding).astype("float32")

    scores, indices = index.search(query_embedding, k)

    # Aggregate scores per plan
    plan_scores = defaultdict(list)

    for score, idx in zip(scores[0], indices[0]):
        plan_id = metadata[idx]["doc_id"]
        plan_scores[plan_id].append(score)

    # Average score per plan
    aggregated = []
    for plan_id, score_list in plan_scores.items():
        avg_score = sum(score_list) / len(score_list)
        aggregated.append((plan_id, avg_score))

    # Sort descending
    aggregated.sort(key=lambda x: x[1], reverse=True)

    # Return top unique plans
    return [plan_id for plan_id, _ in aggregated[:TOP_K]]

# ----------------------------
# AUTO DOC_ID RESOLUTION
# ----------------------------

def get_doc_id(plan_keyword):
    matches = [
        m["doc_id"]
        for m in metadata
        if plan_keyword.lower() in m["plan_name"].lower()
    ]
    if not matches:
        raise ValueError(f"No doc_id found for keyword: {plan_keyword}")
    return matches[0]

# ----------------------------
# BUILD EVALUATION SET
# ----------------------------

evaluation_set = [
    {
        "query": "Does Horizon 4 cover inpatient consultant fees?",
        "expected_doc_id": get_doc_id("Horizon 4")
    },
    {
        "query": "Which plan has a €300 excess for semi-private room admission?",
        "expected_doc_id": get_doc_id("300")
    },
    {
        "query": "How many days of psychiatric treatment are covered under Plan A?",
        "expected_doc_id": get_doc_id("Plan A")
    },
    {
        "query": "Are inpatient scans fully covered under Health Plan 26.1?",
        "expected_doc_id": get_doc_id("26.1")
    }
]

# ----------------------------
# METRIC COMPUTATION
# ----------------------------

correct_top1 = 0
correct_topk = 0
reciprocal_ranks = []

print("\nRunning evaluation...\n")

for item in evaluation_set:
    query = item["query"]
    expected = item["expected_doc_id"]

    returned_ids = search(query)

    print("Query:", query)
    print("Expected:", expected)
    print("Returned:", returned_ids)
    print("-" * 60)

    if returned_ids[0] == expected:
        correct_top1 += 1

    if expected in returned_ids:
        correct_topk += 1
        rank = returned_ids.index(expected) + 1
        reciprocal_ranks.append(1.0 / rank)
    else:
        reciprocal_ranks.append(0.0)

# ----------------------------
# FINAL METRICS
# ----------------------------

total = len(evaluation_set)

top1_accuracy = correct_top1 / total
topk_recall = correct_topk / total
mrr = sum(reciprocal_ranks) / total

print("\n===== FINAL METRICS =====")
print(f"Top-1 Accuracy: {top1_accuracy:.3f}")
print(f"Top-{TOP_K} Recall: {topk_recall:.3f}")
print(f"MRR: {mrr:.3f}")
