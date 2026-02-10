import json
import re
from collections import defaultdict

INPUT_PATH = "/Users/ambikaprasanth/Desktop/Gallery/rag_chunks.jsonl"
OUTPUT_PATH = "/Users/ambikaprasanth/Desktop/Gallery/rag_chunks_engineered_v2.jsonl"


# ----------------------------
# CLEAN TEXT
# ----------------------------
def clean_text(text):
    if not text:
        return ""

    # Remove regulatory header
    text = re.sub(
        r"Irish Life Health dac is regulated by the Central Bank of Ireland\.",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove footnote numbers like "for2", "Procedures1"
    text = re.sub(r"([A-Za-z])\d+", r"\1", text)

    # Fix broken lines inside sentences
    text = re.sub(r"\n+", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ----------------------------
# PROCESS FULL DOCUMENT
# ----------------------------
def process_file():
    documents = defaultdict(list)

    # Group by doc_id
    with open(INPUT_PATH, "r") as infile:
        for line in infile:
            obj = json.loads(line)
            documents[obj["doc_id"]].append(obj)

    with open(OUTPUT_PATH, "w") as outfile:

        for doc_id, chunks in documents.items():

            # Sort pages
            chunks = sorted(chunks, key=lambda x: x.get("page_start", 0))

            # Merge full text
            full_text = " ".join(chunk.get("text", "") for chunk in chunks)
            full_text = clean_text(full_text)

            metadata = chunks[0]

            merged_doc = {
                "doc_id": doc_id,
                "insurer": metadata.get("insurer"),
                "plan_name": metadata.get("plan_name"),
                "doc_type": metadata.get("doc_type"),
                "version_date": metadata.get("version_date"),
                "source_url": metadata.get("source_url"),
                "text": full_text
            }

            outfile.write(json.dumps(merged_doc) + "\n")

    print("Document-level processing complete.")


if __name__ == "__main__":
    process_file()
