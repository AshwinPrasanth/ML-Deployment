import os
import streamlit as st

from engine_rag import RAGEngine, EngineConfig, DISEASE_RULES

st.set_page_config(page_title="Policy Plan Recommender", layout="wide")

# ---- Cache engine so it loads once ----
@st.cache_resource
def load_engine():
    return RAGEngine(EngineConfig())

engine = load_engine()

st.title("🏥 Health Insurance Policy Plan Recommender (Streamlit)")
st.caption("Uses your repo files: rag_chunks_engineered_v2.jsonl + doc_registry.json + auto-built faiss_index.bin")

# ---- Sidebar ----
with st.sidebar:
    st.header("Patient Profile")

    age = st.slider("Age", 18, 90, 32)

    conditions = st.multiselect(
        "Chronic conditions",
        options=list(DISEASE_RULES.keys()),
        default=[],
    )

    meds = st.selectbox("Medication frequency", ["none", "monthly", "weekly", "daily"], index=0)
    specialist_visits = st.slider("Specialist visits per year", 0, 20, 2)
    admissions = st.slider("Hospital admissions (last 2 years)", 0, 10, 0)

    st.divider()
    st.header("User Intent / Query")
    query = st.text_area(
        "Describe your needs",
        value="I want a plan with good inpatient coverage, low excess, and good support for my conditions.",
        height=110
    )

    run = st.button("Find best plans", type="primary", use_container_width=True)

profile = {
    "age": age,
    "chronic_conditions": conditions,
    "medication_frequency": meds,
    "specialist_visits_per_year": specialist_visits,
    "hospital_admissions_last_2_years": admissions,
}

# ---- Layout ----
colA, colB = st.columns([1.1, 1])

with colA:
    st.subheader("Profile summary")
    st.json(profile)

with colB:
    st.subheader("What this app does")
    st.write(
        "- Retrieves **policy chunks** with a **hybrid** approach (FAISS semantic + BM25 keyword)\n"
        "- Aggregates results to **plan (doc_id)**\n"
        "- Applies **transparent rules** for a final score\n"
        "- Shows **evidence chunks** (text + page/pdf if present)"
    )

st.divider()

if run:
    if not query.strip():
        st.error("Please type a query.")
    else:
        profile_query = (
            f"{query}\n"
            f"Age: {age}. "
            f"Conditions: {', '.join(conditions) if conditions else 'none'}. "
            f"Medication frequency: {meds}. "
            f"Specialist visits per year: {specialist_visits}. "
            f"Hospital admissions last 2 years: {admissions}. "
            f"Preferences: low excess, strong inpatient cover."
        )

        results = engine.recommend(profile=profile, query=profile_query)


        if not results:
            st.warning("No results returned. Check that your JSONL chunks have doc_id/text fields.")
        else:
            st.subheader("Top recommendations")

            for rank, r in enumerate(results, start=1):
                with st.container(border=True):
                    left, right = st.columns([1.25, 1])

                    with left:
                        title = f"#{rank} — {r['display_name']}"
                        if r.get("company"):
                            title += f" ({r['company']})"
                        st.markdown(f"### {title}")

                        st.write(
                            f"**Total score:** {r['total_score']:.2f}  \n"
                            f"Rule score: {r['rule_score']:.2f} · Retrieval score: {r['retrieval_score']:.3f}"
                        )

                        chips = []
                        chips.append(f"💶 Excess €{r.get('excess', 0)}")
                        chips.append(f"🧠 Psych days {r.get('psychiatric_days', 0)}")
                        if r.get("full_inpatient"): chips.append("✅ Inpatient cover")
                        if r.get("high_tech"): chips.append("🏥 High-tech access")
                        if r.get("outpatient"): chips.append("🩺 Outpatient support")
                        if r.get("maternity"): chips.append("🤰 Maternity support")

                        st.write(" • ".join(chips))

                        st.markdown("**Why this plan?**")
                        for reason in r.get("reasons", [])[:10]:
                            st.write(f"- {reason}")

                    with right:
                        st.markdown("**Evidence (top matching chunks)**")
                        for ev in r.get("evidence", []):
                            meta = []
                            if ev.get("source_pdf"): meta.append(f"PDF: `{ev['source_pdf']}`")
                            if ev.get("page") is not None: meta.append(f"Page: {ev['page']}")
                            if ev.get("section"): meta.append(f"Section: {ev['section']}")
                            meta_str = " · ".join(meta) if meta else "Chunk evidence"

                            with st.expander(f"{meta_str} (score {ev.get('chunk_score', 0):.3f})"):
                                st.write(ev.get("text", ""))

else:
    st.info("Fill your profile + query in the sidebar and click **Find best plans**.")
