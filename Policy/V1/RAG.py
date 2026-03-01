import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "/content/rag_chunks_engineered_v2.jsonl"
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
metadata = []

with open(DATA_PATH, "r") as f:
    for line in f:
        obj = json.loads(line)

        # Filter only table_of_cover if desired
        if obj["doc_type"] == "table_of_cover":
            documents.append(obj["text"])
            metadata.append(obj)

embeddings = model.encode(documents, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f)

print("FAISS index built successfully.")

def search(query, k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results
