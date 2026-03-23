import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import base64
import tempfile
import streamlit as st

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Irish Health Insurance Advisor",
    page_icon="🏥",
    layout="wide",
)

# =============================================================================
# ENGINE — cached so it loads once per session
# =============================================================================
@st.cache_resource(show_spinner="Loading RAG engine (first run only)…")
def load_engine():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from Policy.Main.Test2.Test2_Final import (
        smart_search, extract_user_profile, CONDITION_MAP, URL_METADATA_PATH
    )
    return smart_search, extract_user_profile, CONDITION_MAP, URL_METADATA_PATH

# PDF folder
PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data-Engineering", "data", "pdfs")
# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"]   { font-family: 'DM Sans', sans-serif; }
h1, h2, h3                   { font-family: 'Lora', serif !important; }

div[data-testid="stSidebar"]            { background: #0d1f33; }
div[data-testid="stSidebar"] *          { color: #c9dff2 !important; }
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3         { color: #ffffff !important;
                                          font-family: 'Lora', serif !important; }
div[data-testid="stSidebar"] label      { color: #6fa3cc !important;
                                          font-size: 0.76rem;
                                          text-transform: uppercase;
                                          letter-spacing: 0.07em; }
div[data-testid="stSidebar"] .stButton > button {
    background: #1762a8; color: #fff; border: none;
    border-radius: 8px; font-weight: 600;
    width: 100%; padding: 0.65rem 0;
    font-size: 0.95rem; margin-top: 0.5rem;
}
div[data-testid="stSidebar"] .stButton > button:hover { background: #124f88; }

.profile-banner {
    background: #e8f0fe; border: 1px solid #b8d0f8;
    border-radius: 10px; padding: 0.65rem 1rem;
    font-size: 0.84rem; color: #0d2d5e;
    margin-bottom: 1.2rem; line-height: 1.7;
}
.plan-card {
    border: 1.5px solid #dce9f6;
    border-radius: 14px;
    padding: 1.1rem 1.1rem 1rem;
    background: #f7fbff;
}
.rank-badge {
    display: inline-block;
    background: #0d1f33; color: #fff;
    font-size: 0.66rem; font-weight: 700;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: 0.08em; margin-bottom: 0.45rem;
}
.plan-title {
    font-family: 'Lora', serif;
    font-size: 1.05rem; color: #0d1f33;
    margin-bottom: 0.1rem; line-height: 1.3;
}
.provider-sub { color: #5a7a96; font-size: 0.82rem; margin-bottom: 0.7rem; }
.score-row   { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.75rem; }
.chip        { border-radius: 7px; padding: 3px 10px; font-size: 0.75rem; font-weight: 600; }
.chip-fit    { background: #0d1f33; color: #fff; }
.chip-rule   { background: #dbeafe; color: #1e3a5f; }
.chip-rrf    { background: #dcfce7; color: #14532d; }
.fit-summary {
    background: #eef5ff;
    border-left: 3px solid #1762a8;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 0.9rem;
    font-size: 0.82rem; line-height: 1.7;
    color: #182f4a; margin: 0.45rem 0 0.8rem;
}
.sw-label {
    font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-bottom: 0.25rem; margin-top: 0.5rem;
}
.pill {
    display: inline-block; border-radius: 20px;
    padding: 2px 9px; font-size: 0.73rem; margin: 2px 2px 2px 0;
}
.pill-green  { background:#e8f5e9; color:#1b5e20; border:1px solid #a5d6a7; }
.pill-red    { background:#fce4ec; color:#880e4f; border:1px solid #f48fb1; }
.pill-amber  { background:#fff8e1; color:#6d4c00; border:1px solid #ffe082; }
.rule-note   { font-size: 0.78rem; color: #3a5a78; padding: 1px 0; }
.src-link    { font-size: 0.78rem; color: #1762a8; }

/* PDF tab viewer */
.pdf-tab-bar {
    display: flex; gap: 6px; margin-top: 1rem; margin-bottom: 0;
}
.pdf-tab {
    background: #e8f0fe; border: 1.5px solid #c5d8fa;
    border-bottom: none; border-radius: 8px 8px 0 0;
    padding: 6px 16px; font-size: 0.78rem; font-weight: 600;
    color: #0d2d5e; cursor: pointer;
}
.pdf-viewer-box {
    border: 1.5px solid #c5d8fa; border-radius: 0 8px 8px 8px;
    overflow: hidden; background: #fff;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🏥 Health Insurance\nAdvisor")
    st.markdown("---")

    st.markdown("### Your Query")
    query = st.text_area(
        "Describe your situation in plain English",
        value="I'm 68 with Parkinson's. I see a neurologist monthly and take daily medication.",
        height=140,
    )

    st.markdown("### Override / Add Details")
    st.caption("Use these if your query does not mention them explicitly.")

    age_override        = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
    condition_override  = st.multiselect("Extra conditions", options=list(CONDITION_MAP.keys()), default=[])
    med_freq            = st.selectbox("Medication frequency", ["(not set)", "daily", "weekly", "monthly"], index=0)
    specialist_visits   = st.number_input("Specialist visits / year",          min_value=0, value=0, step=1)
    hospital_admissions = st.number_input("Hospital admissions (last 2 yrs)",  min_value=0, value=0, step=1)

    st.markdown("---")
    run = st.button("🔍 Find Best Plans", type="primary", use_container_width=True)
    if st.button("🗑️ Clear PDF Cache", use_container_width=True, help="Force re-highlight all PDFs"):
        for key in list(st.session_state.keys()):
            if key.startswith("pdf_cache_"):
                del st.session_state[key]
        st.success("Cache cleared — PDFs will be re-highlighted on next search.")


# =============================================================================
# HELPERS
# =============================================================================
def build_enriched_query(base, age, conditions, med, visits, admissions):
    parts = [base.strip()]
    if age > 0:
        parts.append(f"I am {age} years old.")
    for cond in conditions:
        kw = CONDITION_MAP[cond][0][0]
        if kw.lower() not in base.lower():
            parts.append(kw + ".")
    if med != "(not set)":
        parts.append(f"I take {med} medication.")
    if visits > 0:
        parts.append(f"I have {visits} specialist visits per year.")
    if admissions > 0:
        parts.append(f"I had {admissions} hospital admissions in the last 2 years.")
    return " ".join(parts)


def fit_chip_colour(score):
    if score >= 8: return "#1b5e20"
    if score >= 6: return "#0d1f33"
    if score >= 4: return "#e65100"
    return "#b71c1c"


def _normalize(text):
    import re
    text = text.lower()
    text = re.sub(r'\b(table of cover|table of benefits|toc|tob)\b', '', text)
    text = re.sub(r'[^a-z0-9]', '', text)
    return text.strip()


def resolve_pdf_path(result):
    url = result.get("source_url", "")
    if url and not url.startswith("http") and url.lower().endswith(".pdf") and os.path.isfile(url):
        return url

    internal_key = result.get("internal_key", "")
    plan_name    = result.get("plan_name", "")

    if internal_key:
        candidate = os.path.join(PDF_DIR, internal_key + ".pdf")
        if os.path.isfile(candidate):
            return candidate
    if plan_name:
        candidate = os.path.join(PDF_DIR, plan_name + ".pdf")
        if os.path.isfile(candidate):
            return candidate

    if not os.path.isdir(PDF_DIR):
        return None

    pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        return None

    key_norm  = _normalize(internal_key)
    name_norm = _normalize(plan_name)

    def trigrams(s):
        return set(s[i:i+3] for i in range(len(s) - 2)) if len(s) >= 3 else {s}

    best_path, best_score = None, 0
    for fname in pdfs:
        stem_norm = _normalize(fname[:-4])
        if not stem_norm:
            continue
        for query_norm in (key_norm, name_norm):
            if not query_norm:
                continue
            shorter, longer = sorted([query_norm, stem_norm], key=len)
            if not longer:
                continue
            shared = trigrams(shorter) & trigrams(longer)
            score  = len(shared) / max(len(trigrams(shorter)), len(trigrams(longer)), 1)
            if shorter in longer or longer in shorter:
                score += 0.5
            if score > best_score:
                best_score = score
                best_path  = os.path.join(PDF_DIR, fname)

    return best_path if best_score >= 0.3 else None


def build_evidence_items(evidence_text: str) -> list:
    items = []
    for chunk in evidence_text.split("\n\n---\n\n"):
        chunk = chunk.strip()
        if chunk:
            items.append({"text": chunk, "page_start": None, "page_end": None})
    return items


def generate_highlighted_pdf(pdf_path: str, evidence_text: str):
    try:
        from pdf_highlighter import highlight_chunks
    except ImportError:
        return None, "pdf_highlighter.py not found or PyMuPDF not installed."
    try:
        evidence_items = build_evidence_items(evidence_text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_out:
            tmp_out_path = tmp_out.name
        highlight_chunks(pdf_path, evidence_items, tmp_out_path)
        with open(tmp_out_path, "rb") as f:
            pdf_bytes = f.read()
        os.unlink(tmp_out_path)
        return pdf_bytes, None
    except Exception as e:
        return None, str(e)


def show_pdf_viewer(pdf_bytes: bytes, rank: int, plan_name: str, height: int = 620):
    """
    - Inline view  : <object> via st.markdown — data: URI works here, scroll works
    - Full Screen  : components.html button that builds a blob URL and calls window.open
                     (components iframe has real JS execution, no sandbox block)
    """
    import streamlit.components.v1 as components
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    data_uri = f"data:application/pdf;base64,{b64}"

    # ── 1. Inline scrollable viewer ──────────────────────────────────────────
    st.markdown(
        f"""
        <div style="border:1.5px solid #c5d8fa; border-radius:12px 12px 0 0;
                    overflow:hidden; margin-top:0.8rem;">
          <div style="background:#e8f0fe; padding:6px 14px;
                      font-size:0.76rem; font-weight:600; color:#0d2d5e;
                      border-bottom:1px solid #c5d8fa;
                      font-family:'DM Sans',sans-serif;">
            📄 Plan #{rank}: {plan_name}
          </div>
          <object
            data="{data_uri}"
            type="application/pdf"
            width="100%"
            height="{height}px"
            style="display:block; border:none;"
          >
            <p style="padding:1rem;font-size:0.85rem;color:#555;">
              PDF cannot display inline.
            </p>
          </object>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 2. Full Screen button — runs in components iframe where JS works ──────
    components.html(
        f"""
        <style>
          body {{ margin:0; padding:4px 0 0 0; background:transparent; }}
          button {{
            width: 100%;
            background: #1762a8; color: #fff;
            border: none; border-radius: 0 0 10px 10px;
            padding: 7px 0; font-size: 13px; font-weight: 700;
            cursor: pointer; letter-spacing: 0.03em;
            font-family: 'DM Sans', sans-serif;
          }}
          button:hover {{ background: #124f88; }}
        </style>
        <button id="fsBtn">⛶ Open Full Screen</button>
        <script>
          var _blobUrl = null;
          document.getElementById("fsBtn").addEventListener("click", function() {{
            if (_blobUrl) {{ window.open(_blobUrl, "_blank"); return; }}
            var bin = atob("{b64}");
            var buf = new Uint8Array(bin.length);
            for (var i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
            var blob = new Blob([buf], {{type: "application/pdf"}});
            _blobUrl = URL.createObjectURL(blob);
            window.open(_blobUrl, "_blank");
          }});
        </script>
        """,
        height=48,
        scrolling=False,
    )


def render_plan_column(col, rank, r):
    plan_name   = r.get("plan_name",     "Unknown Plan")
    provider    = r.get("provider",      "")
    source_url  = r.get("source_url",    "")
    fit_score   = r.get("fit_score",     0)
    rule_score  = r.get("rule_score",    0)
    rrf_score   = r.get("rrf_score",     0)
    fit_summary = r.get("fit_summary",   "")
    strengths   = r.get("strengths",     [])
    weaknesses  = r.get("weaknesses",    [])
    gaps        = r.get("coverage_gaps", [])
    rule_notes  = r.get("rule_notes",    [])
    numeric_sum = r.get("numeric_summary","")
    evidence    = r.get("evidence",      "")

    with col:
        # ── Plan card ────────────────────────────────────────────────────────
        st.markdown('<div class="plan-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="rank-badge">RANK #{rank}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="plan-title">{plan_name}</div>', unsafe_allow_html=True)
        if provider:
            st.markdown(f'<div class="provider-sub">{provider}</div>', unsafe_allow_html=True)

        fc = fit_chip_colour(fit_score)
        st.markdown(
            f'<div class="score-row">'
            f'<span class="chip chip-fit" style="background:{fc};">⭐ {fit_score}/10</span>'
            f'<span class="chip chip-rule">⚙️ {rule_score}</span>'
            f'<span class="chip chip-rrf">🔍 {rrf_score}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if fit_summary:
            st.markdown(f'<div class="fit-summary">{fit_summary}</div>', unsafe_allow_html=True)

        if strengths:
            st.markdown('<div class="sw-label" style="color:#1b5e20">✅ Strengths</div>', unsafe_allow_html=True)
            st.markdown("".join(f'<span class="pill pill-green">{s}</span>' for s in strengths), unsafe_allow_html=True)

        if weaknesses:
            st.markdown('<div class="sw-label" style="color:#880e4f">⚠️ Weaknesses</div>', unsafe_allow_html=True)
            st.markdown("".join(f'<span class="pill pill-red">{w}</span>' for w in weaknesses), unsafe_allow_html=True)

        if gaps:
            st.markdown('<div class="sw-label" style="color:#6d4c00">🔍 Gaps</div>', unsafe_allow_html=True)
            st.markdown("".join(f'<span class="pill pill-amber">{g}</span>' for g in gaps), unsafe_allow_html=True)

        if numeric_sum and numeric_sum != "No structured numeric data available.":
            with st.expander("📊 Numeric data"):
                st.code(numeric_sum, language=None)

        if rule_notes:
            with st.expander(f"⚙️ Rule notes ({len(rule_notes)})"):
                for note in rule_notes:
                    st.markdown(f'<div class="rule-note">• {note}</div>', unsafe_allow_html=True)

        if evidence:
            with st.expander("📑 Policy chunks"):
                st.markdown(evidence)

        if source_url and source_url != "No URL Available":
            st.markdown(f'<a class="src-link" href="{source_url}" target="_blank">🔗 View policy online →</a>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Auto PDF highlight + inline viewer ───────────────────────────────
        pdf_path = resolve_pdf_path(r)

        if pdf_path:
            cache_key = f"pdf_cache_{rank}_{os.path.basename(pdf_path)}"

            if cache_key not in st.session_state:
                with st.spinner(f"Highlighting PDF for Plan #{rank}…"):
                    pdf_bytes, err = generate_highlighted_pdf(pdf_path, evidence)
                if err:
                    st.warning(f"Could not highlight PDF: {err}")
                    pdf_bytes = None
                else:
                    st.session_state[cache_key] = pdf_bytes
            else:
                pdf_bytes = st.session_state[cache_key]

            if pdf_bytes:
                show_pdf_viewer(pdf_bytes, rank, plan_name, height=620)
                st.download_button(
                    label="⬇️ Download Highlighted PDF",
                    data=pdf_bytes,
                    file_name=f"plan{rank}_highlighted.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"dl_pdf_{rank}",
                )
        else:
            existing = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")] if os.path.isdir(PDF_DIR) else []
            pdf_list = ", ".join(f"`{f}`" for f in existing) if existing else f"_No PDFs found in `{PDF_DIR}`_"
            key_norm = _normalize(r.get("internal_key", ""))
            st.warning(
                f"**No PDF matched for Plan #{rank} — {plan_name}**\n\n"
                f"Doc ID: `{r.get('internal_key', '')}` → normalized: `{key_norm}`\n\n"
                f"PDFs in folder: {pdf_list}"
            )


# =============================================================================
# MAIN HEADER
# =============================================================================
st.markdown("# 🏥 Irish Health Insurance Advisor")
st.markdown(
    "<p style='color:#5a7a96;font-size:0.93rem;margin-top:-0.4rem;'>"
    "Rule engine &nbsp;·&nbsp; Hybrid RRF retrieval &nbsp;·&nbsp; "
    "Llama-3.3-70B reranker via Groq"
    "</p>",
    unsafe_allow_html=True,
)

h1, h2, h3 = st.columns(3)
h1.info("**⚙️ Rule Engine**\nHard-rejects plans missing required cover. Scores on excess, utilisation & chunk hit-rate.")
h2.info("**🤖 LLM Reranker**\nReads surviving plans holistically. Reasons about complex & unknown conditions.")
h3.info("**🧠 Profile Extractor**\nAuto-parses age, conditions, visits & medication directly from your free-text query.")

st.divider()


# =============================================================================
# SEARCH + RESULTS
# =============================================================================
if run:
    if not query.strip():
        st.error("Please enter a query in the sidebar.")
        st.stop()

    enriched = build_enriched_query(
        query, age_override, condition_override,
        med_freq, specialist_visits, hospital_admissions,
    )

    # ── Profile banner ───────────────────────────────────────────────────────
    p = extract_user_profile(enriched)
    st.markdown(
        f'<div class="profile-banner">'
        f'<strong>🧠 Detected profile</strong>&nbsp;&nbsp;'
        f'Age: <strong>{p["age"] or "not detected"}</strong>'
        f'&nbsp;·&nbsp;Conditions: <strong>{", ".join(p["conditions"]) or "none"}</strong>'
        f'&nbsp;·&nbsp;Medication: <strong>{p["medication_freq"] or "not set"}</strong>'
        f'&nbsp;·&nbsp;Specialist visits/yr: <strong>{p["specialist_visits"] if p["specialist_visits"] is not None else "—"}</strong>'
        f'&nbsp;·&nbsp;Hospital admissions: <strong>{p["hospital_admissions"] if p["hospital_admissions"] is not None else "—"}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Run pipeline ─────────────────────────────────────────────────────────
    with st.spinner("Retrieving → Rule engine → LLM reranker…"):
        try:
            results = smart_search(enriched, top_n=3)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    if not results:
        st.warning("No plans passed the rule engine. Try rephrasing or removing condition filters.")
        st.stop()

    # Clear stale PDF cache on every new search
    for key in list(st.session_state.keys()):
        if key.startswith("pdf_cache_"):
            del st.session_state[key]

    st.subheader("Top 3 Recommended Plans")

    cols = st.columns(3)
    for rank, (col, r) in enumerate(zip(cols, results), 1):
        render_plan_column(col, rank, r)

else:
    st.info("👈 Enter your situation in the sidebar and click **Find Best Plans**.")