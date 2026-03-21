# Copyright 2026 Ashwin Prasanth, Konstantinos Sklavenitis, Kiran, Charalampos Theodoridis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import re
import fitz  # PyMuPDF


def _clean(text: str) -> str:
    """Collapse all whitespace to single spaces."""
    return re.sub(r'\s+', ' ', text).strip()


def _phrases(chunk: str, min_len=20, max_len=80):
    """
    Split a chunk into short phrases that are likely to appear
    verbatim in the PDF text layer.
    """
    # Split on newlines and sentence boundaries
    parts = re.split(r'\n+|(?<=[.?!])\s+', chunk)
    result = []
    for part in parts:
        part = _clean(part)
        if not part:
            continue
        if len(part) <= max_len:
            if len(part) >= min_len:
                result.append(part)
        else:
            # Too long — split further on comma/semicolon/colon
            for sub in re.split(r'[,;:]\s+', part):
                sub = _clean(sub)
                if min_len <= len(sub) <= max_len:
                    result.append(sub)

    # Deduplicate preserving order
    seen, unique = set(), []
    for p in result:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def highlight_chunks(pdf_path: str, evidence: list, output_path: str) -> str:
    """
    Highlight evidence chunks in a PDF and save to output_path.

    evidence: list of dicts with:
        "text"       – chunk text (required)
        "page_start" – 1-based page hint (optional, None = search ALL pages)
        "page_end"   – 1-based page hint (optional)
    """
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    total_hits = 0

    for item in evidence:
        chunk = _clean(item.get("text") or "")
        if not chunk:
            continue

        # --- page range ---
        ps = item.get("page_start")
        pe = item.get("page_end")
        if ps is not None:
            try:
                start = max(0, int(ps) - 1)
                end   = min(max(0, int(pe if pe is not None else ps) - 1), n_pages - 1)
            except Exception:
                start, end = 0, n_pages - 1
        else:
            start, end = 0, n_pages - 1   # ← KEY FIX: search all pages when no hint

        phrases = _phrases(chunk)

        # Always add a short prefix as extra candidate
        prefix = _clean(chunk[:60])
        if len(prefix) >= 15 and prefix not in phrases:
            phrases.append(prefix)

        for pn in range(start, end + 1):
            page = doc[pn]
            for phrase in phrases:
                hits = page.search_for(phrase)
                for rect in hits:
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=[1, 0.85, 0])  # yellow
                    annot.update()
                    total_hits += 1

    print(f"[pdf_highlighter] {total_hits} highlights across {n_pages} pages.")
    doc.save(output_path)
    doc.close()
    return output_path
