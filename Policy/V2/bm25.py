import json
import numpy as np
import re
from collections import defaultdict
from rank_bm25 import BM25Okapi

# ----------------------------
# CONFIG
# ----------------------------

METADATA_PATH = "metadata_v2.json"
TOP_K = 3

# ----------------------------
# LOAD CHUNKED METADATA
# ----------------------------

print("Loading chunked metadata...")
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

print(f"Using {len(metadata)} chunks")

# ----------------------------
# TOKENIZATION
# ----------------------------

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9€]+", " ", text)
    return text.split()

corpus = [m["chunk_text"] for m in metadata]
tokenized_corpus = [tokenize(doc) for doc in corpus]

print("Building BM25 chunk-level index...")
bm25 = BM25Okapi(tokenized_corpus)

# ----------------------------
# SEARCH (Chunk → Plan Aggregation)
# ----------------------------

def search_bm25(query, k=20):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Aggregate scores per plan
    plan_scores = defaultdict(list)

    for idx, score in enumerate(scores):
        plan_id = metadata[idx]["doc_id"]
        plan_scores[plan_id].append(score)

    aggregated = []
    for plan_id, score_list in plan_scores.items():
        avg_score = sum(score_list) / len(score_list)
        aggregated.append((plan_id, avg_score))

    aggregated.sort(key=lambda x: x[1], reverse=True)

    return [plan_id for plan_id, _ in aggregated[:TOP_K]]

# ----------------------------
# DOC ID RESOLUTION
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

print("\nRunning BM25 v2 evaluation...\n")

for item in evaluation_set:
    query = item["query"]
    expected = item["expected_doc_id"]

    returned_ids = search_bm25(query)

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

print("\n===== BM25 v2 =====")
print("Top-1:", correct_top1 / total)
print("Top-3:", correct_topk / total)
print("MRR:", sum(reciprocal_ranks) / total)
