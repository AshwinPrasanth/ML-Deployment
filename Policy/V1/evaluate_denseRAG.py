import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------
# CONFIG
# ----------------------------

INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.json"
TOP_K = 3

# ----------------------------
# LOAD MODEL + INDEX
# ----------------------------

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# ----------------------------
# SEARCH FUNCTION
# ----------------------------

def search(query, k=TOP_K):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Normalize query for cosine similarity
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, k)

    return [metadata[idx] for idx in indices[0]]


# ----------------------------
# AUTO DOC_ID RESOLUTION
# ----------------------------

def get_doc_id(plan_keyword):
    matches = [m["doc_id"] for m in metadata if plan_keyword.lower() in m["plan_name"].lower()]
    if not matches:
        raise ValueError(f"No doc_id found for keyword: {plan_keyword}")
    return matches[0]  # assume unique plan names

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

    results = search(query)
    returned_ids = [r["doc_id"] for r in results]

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
