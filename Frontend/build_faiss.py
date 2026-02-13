import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = os.path.join(
    "ConstructionJSON", "data", "rag_chunks_engineered_v2.jsonl"
)
INDEX_PATH = os.path.join(
    "ConstructionJSON", "faiss_index.bin"
)

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["text"])

embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)
print(f"FAISS index saved to {INDEX_PATH}")
