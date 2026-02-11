import os
import re
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import requests
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


OUT_DIR = "data"
PDF_DIR = os.path.join(OUT_DIR, "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)

DOC_REGISTRY_PATH = os.path.join(OUT_DIR, "doc_registry.json")
CHUNKS_JSONL_PATH = os.path.join(OUT_DIR, "rag_chunks.jsonl")

def safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())
    return s[:180] if len(s) > 180 else s

def sha1_short(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]

def download_pdf(url: str, out_path: str) -> None:
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://levelhealth.ie/",
        "Connection": "keep-alive",
    }

    with requests.Session() as s:
        r = s.get(url, headers=headers, stream=True, timeout=60)
        r.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def normalize_text(t: str) -> str:
    # Fix hyphen line breaks: "hospi-\ntal" -> "hospital"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    # Join broken lines conservatively
    t = re.sub(r"\n{3,}", "\n\n", t)
    # Remove excessive whitespace
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()

@dataclass
class DocMeta:
    doc_id: str
    insurer: str
    plan_name: str
    doc_type: str          # e.g., "table_of_cover" or "terms_conditions"
    version_date: str      # e.g., "2025-01-01" (string for simplicity)
    source_url: str
    file_path: str

# ----------------------------
# Main pipeline
# ----------------------------
def run_pipeline(doc_sources: List[Dict[str, str]]) -> None:
    """
    doc_sources item format:
    {
      "insurer": "VHI",
      "plan_name": "Plan A",
      "doc_type": "table_of_cover",
      "version_date": "2025-01-01",
      "url": "https://....pdf"
    }
    """
    registry: List[DocMeta] = []
    all_chunks: List[Dict[str, Any]] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,      # ~300-600 tokens depending on text
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for item in tqdm(doc_sources, desc="Processing PDFs"):
        insurer = item["insurer"]
        plan_name = item["plan_name"]
        doc_type = item["doc_type"]
        version_date = item.get("version_date", "")
        url = item["url"]

        # Build IDs / paths
        base = f"{insurer}_{plan_name}_{doc_type}_{version_date}".strip("_")
        doc_id = safe_filename(base).lower()
        pdf_name = safe_filename(doc_id) + ".pdf"
        pdf_path = os.path.join(PDF_DIR, pdf_name)

        # Download if missing
        if not os.path.exists(pdf_path):
            download_pdf(url, pdf_path)

        registry.append(
            DocMeta(
                doc_id=doc_id,
                insurer=insurer,
                plan_name=plan_name,
                doc_type=doc_type,
                version_date=version_date,
                source_url=url,
                file_path=pdf_path,
            )
        )

        # Load and chunk with LangChain loader
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()  # one Document per page, with metadata {"page": i}

        for p in pages:
            page_num = int(p.metadata.get("page", -1)) + 1  # make it 1-indexed
            text = normalize_text(p.page_content)
            if not text:
                continue

            # Chunk per page to preserve citation locality
            chunks = splitter.split_text(text)

            for idx, chunk_text in enumerate(chunks):
                chunk_hash = sha1_short(f"{doc_id}|p{page_num}|{idx}|{chunk_text}")
                chunk_id = f"{doc_id}_p{page_num:03d}_{idx:03d}_{chunk_hash}"

                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "insurer": insurer,
                        "plan_name": plan_name,
                        "doc_type": doc_type,
                        "version_date": version_date,
                        "source_url": url,
                        "page_start": page_num,
                        "page_end": page_num,
                        "text": chunk_text,
                    }
                )

    # Write outputs
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(DOC_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in registry], f, ensure_ascii=False, indent=2)

    with open(CHUNKS_JSONL_PATH, "w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved registry: {DOC_REGISTRY_PATH}")
    print(f"Saved chunks:   {CHUNKS_JSONL_PATH}")
    print(f"PDF folder:     {PDF_DIR}")
    print(f"Total chunks:   {len(all_chunks)}")

if __name__ == "__main__":

    DOC_SOURCES = [
        {
        "insurer": "Irish Life Health",
        "plan_name": "First Cover",
        "doc_type": "table_of_cover",
        "version_date": "2026-01-01",
        "url": "https://www.irishlifehealth.ie/mediafiles/pdfs/tables-of-cover/First-Cover-TOC.pdf"
    },
    {
        "insurer": "Irish Life Health",
        "plan_name": "BeneFit",
        "doc_type": "table_of_cover",
        "version_date": "2026-01-01",
        "url": "https://www.irishlifehealth.ie/mediafiles/pdfs/tables-of-cover/BeneFit-TOC.pdf"
    },
    {
        "insurer": "Irish Life Health",
        "plan_name": "Horizon 4",
        "doc_type": "table_of_cover",
        "version_date": "2026-01-01",
        "url": "https://www.irishlifehealth.ie/mediafiles/pdfs/tables-of-cover/Horizon_4_TOC.pdf"
    },
    {
        "insurer": "Irish Life Health",
        "plan_name": "Health Plan 26.1",
        "doc_type": "table_of_cover",
        "version_date": "2026-01-01",
        "url": "https://www.irishlifehealth.ie/mediafiles/pdfs/tables-of-cover/Health_Plan_26_1_TOC.pdf"
    },
    {
        "insurer": "Irish Life Health",
        "plan_name": "Select Starter",
        "doc_type": "table_of_cover",
        "version_date": "2026-01-01",
        "url": "https://www.irishlifehealth.ie/mediafiles/pdfs/tables-of-cover/Select-Starter-TOC.pdf"
    },
    {
        "insurer": "Irish Life Health",
        "plan_name": "Health Plans Membership Handbook",
        "doc_type": "terms_conditions",
        "version_date": "2026-01-01",
        "url": "https://www.irishlifehealth.ie/IrishLifeHealth/media/docs/ILH-Health-Plans-Handbook-Jan-2026.pdf"
    },
    {
        "insurer": "Irish Life Health",
        "plan_name": "Tailored Health Plans Membership Handbook",
        "doc_type": "terms_conditions",
        "version_date": "2026-01-01",
        "url": "https://www.irishlifehealth.ie/IrishLifeHealth/media/docs/ILH-Tailored-Health-Plans-Handbook-Jan-2026.pdf"
    },
    {
        "insurer": "Vhi",
        "plan_name": "Company Plan Plus Level 1",
        "doc_type": "table_of_cover",
        "version_date": "2024-10-01",
        "url": "https://www.vhi.ie/pdf/myvhi/TOBCPPL1%20V65%20Oct24.pdf"
    },
    {
        "insurer": "Vhi",
        "plan_name": "Health Access",
        "doc_type": "table_of_cover",
        "version_date": "2023-12-31",
        "url": "https://www.vhi.ie/pdf/myvhi/TOBHEALTHA%20V4402%20Dec23.pdf"
    },
    {
        "insurer": "Vhi",
        "plan_name": "Hospital Plans Rules (Terms & Conditions)",
        "doc_type": "terms_conditions",
        "version_date": "2025-10-01",
        "url": "https://www.vhi.ie/pdf/products/Hospital%20Plans%20Rules_01Oct25_VHIMR2.pdf"
    }]

    DOC_SOURCES += [
    {
        "insurer": "Laya Healthcare",
        "plan_name": "Inspire (Table of Benefits)",
        "doc_type": "table_of_cover",
        "version_date": "2025-08-01",
        "url": "https://www.hia.ie/sites/default/files/Inspire.pdf"
    },
    {
        "insurer": "Laya Healthcare",
        "plan_name": "Prime Plan Table of Benefits",
        "doc_type": "table_of_cover",
        "version_date": "2019-08-01",
        "url": "https://www.hia.ie/sites/default/files/Table%20of%20Benefits%20Prime%2001.08.2019.pdf"
    },
    {
        "insurer": "Laya Healthcare",
        "plan_name": "First & Family Plan – Table of Benefits",
        "doc_type": "table_of_cover",
        "version_date": "2021-02-01",
        "url": "https://www.hia.ie/sites/default/files/First%20%26%20Family%20Plan.pdf"
    },
    {
        "insurer": "Laya Healthcare",
        "plan_name": "PMI Plan B – Table of Benefits",
        "doc_type": "table_of_cover",
        "version_date": "2021-10-01",
        "url": "https://www.hia.ie/sites/default/files/PMI%2001%2010.pdf"
    }]

    DOC_SOURCES += [
    {
        "insurer": "Level Health",
        "plan_name": "Plan A (Table of Cover)",
        "doc_type": "table_of_cover",
        "version_date": "2025-06-27",
        "url": "https://levelhealth.ie/api/download?path=o%2Fassets%252Fbf5aeb09845647e5a8182770c0ca0fdc%252F9b1981371b404059b483765a6329a4f9%3Falt%3Dmedia%26token%3Dddfe44f4-eeed-46db-9b9d-3b783ba1bfee"
    },
    {
        "insurer": "Level Health",
        "plan_name": "Plan B (with €150 Excess) (Table of Cover)",
        "doc_type": "table_of_cover",
        "version_date": "2025-06-27",
        "url": "https://levelhealth.ie/api/download?path=o%2Fassets%252Fbf5aeb09845647e5a8182770c0ca0fdc%252Fd9fe12370802462a8913837d173d2821%3Falt%3Dmedia%26token%3D0958c8b2-3f54-4d83-9a1b-625d7b8979a7"
    },
    {
        "insurer": "Level Health",
        "plan_name": "Plan B (with €300 Excess) (Table of Cover)",
        "doc_type": "table_of_cover",
        "version_date": "2025-06-27",
        "url": "https://levelhealth.ie/api/download?path=o%2Fassets%252Fbf5aeb09845647e5a8182770c0ca0fdc%252F466dbce0188f47308dc32f12d9db6710%3Falt%3Dmedia%26token%3D3c4e0d86-77ef-4275-b244-88bdb7d850fa"
    },
    {
        "insurer": "Level Health",
        "plan_name": "Plan C (Table of Cover)",
        "doc_type": "table_of_cover",
        "version_date": "2025-06-27",
        "url": "https://levelhealth.ie/api/download?path=o%2Fassets%252Fbf5aeb09845647e5a8182770c0ca0fdc%252Ffc78d45331d14d3bb3203be51029e608%3Falt%3Dmedia%26token%3D0fcbd2b3-db9f-4bea-80a1-4b60069b7937"
    },
    {
        "insurer": "Level Health",
        "plan_name": "Plan D (Table of Cover)",
        "doc_type": "table_of_cover",
        "version_date": "2025-06-27",
        "url": "https://levelhealth.ie/api/download?path=o%2Fassets%252Fbf5aeb09845647e5a8182770c0ca0fdc%252Fa738c8a3851b4113889f5e0cb00f28fc%3Falt%3Dmedia%26token%3Db7207980-b88c-4259-984b-e3eaab5bcf5a"
    },
    ]
    run_pipeline(DOC_SOURCES)