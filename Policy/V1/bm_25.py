import json
import numpy as np
import re
from rank_bm25 import BM25Okapi

# ----------------------------
# CONFIG
# ----------------------------

METADATA_PATH = "metadata.json"
TOP_K = 3

# ----------------------------
# LOAD + FILTER METADATA
# ----------------------------

print("Loading metadata...")
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# Keep only Table of Cover documents
filtered_metadata = [
    m for m in metadata
    if m["doc_type"] == "table_of_cover"
]

print(f"Using {len(filtered_metadata)} table_of_cover documents")

# ----------------------------
# TOKENIZATION
# ----------------------------

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9€]+", " ", text)
    return text.split()

corpus = [m["text"] for m in filtered_metadata]
tokenized_corpus = [tokenize(doc) for doc in corpus]

print("Building BM25 index...")
bm25 = BM25Okapi(tokenized_corpus)

# ----------------------------
# SEARCH
# ----------------------------

def search_bm25(query, k=TOP_K):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    return [filtered_metadata[i] for i in top_indices]

# ----------------------------
# DOC ID RESOLUTION
# ----------------------------

def get_doc_id(plan_keyword):
    matches = [
        m["doc_id"]
        for m in filtered_metadata
        if plan_keyword.lower() in m["plan_name"].lower()
    ]
    return matches[0]

# ----------------------------
# EVALUATION SET
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
# METRICS
# ----------------------------

correct_top1 = 0
correct_topk = 0
reciprocal_ranks = []

print("\nRunning BM25 (clean) evaluation...\n")

for item in evaluation_set:
    query = item["query"]
    expected = item["expected_doc_id"]

    results = search_bm25(query)
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

print("\n===== BM25 (FILTERED) =====")
print("Top-1:", correct_top1 / total)
print("Top-3:", correct_topk / total)
print("MRR:", sum(reciprocal_ranks) / total)
