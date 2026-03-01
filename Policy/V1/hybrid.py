import json
import numpy as np
import faiss
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ----------------------------
# CONFIG
# ----------------------------

INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.json"
TOP_K = 3
ALPHA = 0.6
BETA = 0.4

# ----------------------------
# LOAD DATA
# ----------------------------

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# Filter to table_of_cover only
filtered_indices = [
    i for i, m in enumerate(metadata)
    if m["doc_type"] == "table_of_cover"
]

filtered_metadata = [metadata[i] for i in filtered_indices]

print(f"Using {len(filtered_metadata)} table_of_cover documents")

# ----------------------------
# BM25 SETUP
# ----------------------------

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9€]+", " ", text)
    return text.split()

corpus = [m["text"] for m in filtered_metadata]
tokenized_corpus = [tokenize(doc) for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

# ----------------------------
# HYBRID SEARCH
# ----------------------------

def normalize(scores):
    min_s = np.min(scores)
    max_s = np.max(scores)
    if max_s - min_s == 0:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)

def hybrid_search(query, k=TOP_K):

    # Dense scores for all docs
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    dense_scores_all, _ = index.search(query_embedding, len(metadata))
    dense_scores_all = -dense_scores_all[0]

    # Filter dense scores
    dense_scores = np.array([dense_scores_all[i] for i in filtered_indices])

    # BM25 scores
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize
    dense_norm = normalize(dense_scores)
    bm25_norm = normalize(bm25_scores)

    # Fusion
    final_scores = ALPHA * dense_norm + BETA * bm25_norm

    top_indices = np.argsort(final_scores)[::-1][:k]
    return [filtered_metadata[i] for i in top_indices]

# ----------------------------
# EVALUATION
# ----------------------------

def get_doc_id(plan_keyword):
    matches = [
        m["doc_id"]
        for m in filtered_metadata
        if plan_keyword.lower() in m["plan_name"].lower()
    ]
    return matches[0]

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

correct_top1 = 0
correct_topk = 0
reciprocal_ranks = []

print("\nRunning CLEAN Hybrid evaluation...\n")

for item in evaluation_set:
    query = item["query"]
    expected = item["expected_doc_id"]

    results = hybrid_search(query)
    returned_ids = [r["doc_id"] for r in results]

    print("Query:", query)
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

total = len(evaluation_set)

print("\n===== HYBRID (FILTERED) =====")
print("Top-1:", correct_top1 / total)
print("Top-3:", correct_topk / total)
print("MRR:", sum(reciprocal_ranks) / total)
