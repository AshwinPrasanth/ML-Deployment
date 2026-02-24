import json
import numpy as np
import faiss
import re
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ----------------------------
# CONFIG
# ----------------------------

INDEX_PATH = "faiss_index_v2.bin"
METADATA_PATH = "metadata_v2.json"
TOP_K = 3
ALPHA = 0.6
BETA = 0.4

# ----------------------------
# LOAD DATA
# ----------------------------

print("Loading MXBAI model...")
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
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

bm25 = BM25Okapi(tokenized_corpus)

# ----------------------------
# NORMALIZATION
# ----------------------------

def normalize(scores_dict):
    scores = np.array(list(scores_dict.values()))
    min_s = scores.min()
    max_s = scores.max()

    normalized = {}
    for k, v in scores_dict.items():
        if max_s - min_s == 0:
            normalized[k] = 0.0
        else:
            normalized[k] = (v - min_s) / (max_s - min_s)
    return normalized

# ----------------------------
# HYBRID SEARCH
# ----------------------------

def hybrid_search(query, k=TOP_K):

    # -------- DENSE --------
    prefixed_query = "Represent this sentence for searching relevant passages: " + query

    query_embedding = model.encode(
        [prefixed_query],
        normalize_embeddings=True
    )

    query_embedding = np.array(query_embedding).astype("float32")

    dense_scores, dense_indices = index.search(query_embedding, 50)

    dense_plan_scores = defaultdict(list)

    for score, idx in zip(dense_scores[0], dense_indices[0]):
        plan_id = metadata[idx]["doc_id"]
        dense_plan_scores[plan_id].append(score)

    dense_plan_scores = {
        plan: sum(scores)/len(scores)
        for plan, scores in dense_plan_scores.items()
    }

    # -------- BM25 --------
    tokenized_query = tokenize(query)
    bm25_scores_all = bm25.get_scores(tokenized_query)

    bm25_plan_scores = defaultdict(list)

    for idx, score in enumerate(bm25_scores_all):
        plan_id = metadata[idx]["doc_id"]
        bm25_plan_scores[plan_id].append(score)

    bm25_plan_scores = {
        plan: sum(scores)/len(scores)
        for plan, scores in bm25_plan_scores.items()
    }

    # -------- NORMALIZE --------
    dense_norm = normalize(dense_plan_scores)
    bm25_norm = normalize(bm25_plan_scores)

    # -------- FUSION --------
    final_scores = {}

    all_plans = set(dense_norm.keys()) | set(bm25_norm.keys())

    for plan in all_plans:
        d = dense_norm.get(plan, 0)
        b = bm25_norm.get(plan, 0)
        final_scores[plan] = ALPHA * d + BETA * b

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    return [plan_id for plan_id, _ in ranked[:k]]

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

print("\nRunning Hybrid v2 evaluation...\n")

for item in evaluation_set:
    query = item["query"]
    expected = item["expected_doc_id"]

    returned_ids = hybrid_search(query)

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

total = len(evaluation_set)

print("\n===== HYBRID v2 RESULTS =====")
print("Top-1:", correct_top1 / total)
print("Top-3:", correct_topk / total)
print("MRR:", sum(reciprocal_ranks) / total)
