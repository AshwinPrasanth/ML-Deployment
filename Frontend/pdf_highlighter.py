import fitz

def highlight_chunks(pdf_path, evidence, output_path):
    doc = fitz.open(pdf_path)

    for item in evidence:
        chunk = (item.get("text") or "").strip()
        page_start = item.get("page_start")
        page_end = item.get("page_end")

        if not chunk or page_start is None:
            continue

        try:
            start_idx = max(0, int(page_start) - 1)
            end_idx = max(0, int(page_end if page_end is not None else page_start) - 1)
        except Exception:
            continue

        end_idx = min(end_idx, len(doc) - 1)

        for page_num in range(start_idx, end_idx + 1):
            page = doc[page_num]
            rects = page.search_for(chunk[:120])
            for rect in rects:
                annot = page.add_highlight_annot(rect)
                annot.update()

    doc.save(output_path)
    doc.close()
    return output_path