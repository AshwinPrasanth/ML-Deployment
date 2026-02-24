import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
DATA_PATH = "/content/rag_chunks_engineered_v2.jsonl"
INDEX_PATH = "faiss_index_v2.bin"
METADATA_PATH = "metadata_v2.json"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 200

# =========================
# LOAD MODEL (MXBAI LARGE)
# =========================
e_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# =========================
# CHUNKING FUNCTION
# =========================
def chunk_text(text, size=900, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

documents = []
metadata = []

# =========================
# LOAD + CHUNK DATA
# =========================
with open(DATA_PATH, "r") as f:
    for line in f:
        obj = json.loads(line)

        if obj["doc_type"] != "table_of_cover":
            continue

        full_text = obj["text"]
        plan_name = obj.get("plan_name", "Unknown Plan")
        insurer = obj.get("insurer", "Unknown Insurer")
        doc_id = obj.get("doc_id")

        chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):

            enriched_chunk = (
                f"[Insurer: {insurer}]\n"
                f"[Plan: {plan_name}]\n"
                f"[Chunk: {i}]\n\n"
                f"{chunk}"
            )

            documents.append(enriched_chunk)

            metadata.append({
                "doc_id": doc_id,
                "insurer": insurer,
                "plan_name": plan_name,
                "chunk_id": i,
                "chunk_text": chunk,
                "full_text": full_text
            })

print(f"Total chunks indexed: {len(documents)}")

# =========================
# CREATE EMBEDDINGS
# =========================
embeddings = e_model.encode(
    documents,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")

# =========================
# BUILD FAISS INDEX (Cosine)
# =========================
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f)

print("🔥 FAISS v2 index built successfully.")

# ==========================================================
# SEARCH FUNCTION (Parent-Plan Aggregation)
# ==========================================================
def search(query, k=5):
    # MXBAI works better with query prefix
    prefixed_query = "Represent this sentence for searching relevant passages: " + query

    query_embedding = e_model.encode(
        [prefixed_query],
        normalize_embeddings=True
    )

    query_embedding = np.array(query_embedding).astype("float32")

    scores, indices = index.search(query_embedding, k)

    results = []
    seen_plans = set()

    for idx in indices[0]:
        plan_name = metadata[idx]["plan_name"]

        if plan_name not in seen_plans:
            seen_plans.add(plan_name)

            results.append({
                "plan_name": plan_name,
                "insurer": metadata[idx]["insurer"],
                "matched_chunk": metadata[idx]["chunk_text"],
                "full_text": metadata[idx]["full_text"]
            })

    return results


# =========================
# TEST QUERY
# =========================
query = """
Coverage for cardiac treatment, frequent hospital visits,
consultant fees, medication support,
and minimal excess for inpatient admission.
"""

results = search(query)

for r in results:
    print("Plan:", r["plan_name"])
    print("Insurer:", r["insurer"])
    print("Matched Section Preview:")
    print(r["matched_chunk"][:500])
    print("=" * 80)
