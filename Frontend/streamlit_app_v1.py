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

import pandas as pd
import streamlit as st

from engine_rag_v1 import RAGEngine, EngineConfig, DISEASE_RULES

st.set_page_config(page_title="Policy Plan Recommender", layout="wide")

# ---- Cache engine so it loads once ----
@st.cache_resource
def load_engine():
    return RAGEngine(EngineConfig())

engine = load_engine()

st.title("🏥 Health Insurance Policy Plan Recommender (Streamlit)")
st.caption("Uses your repo files: metadata.json + doc_registry.json + faiss_index.bin (V1 setup)")


# ----------------------------
# Helpers (UI-only)
# ----------------------------
def risk_level_badge(profile: dict) -> tuple[str, str, int]:
    """
    Return (label, color, numeric_score) for the user risk indicator.
    Transparent heuristic: age + conditions + meds + admissions + specialist visits.
    """
    score = 0
    age = int(profile.get("age", 0) or 0)
    if age >= 60:
        score += 3
    elif age >= 45:
        score += 2

    score += len(profile.get("chronic_conditions", [])) * 3

    meds = profile.get("medication_frequency", "none")
    score += {"none": 0, "monthly": 1, "weekly": 2, "daily": 3}.get(meds, 0)

    score += int(profile.get("hospital_admissions_last_2_years", 0) or 0) * 2

    if int(profile.get("specialist_visits_per_year", 0) or 0) > 5:
        score += 3

    if score >= 12:
        return "HIGH", "red", score
    if score >= 7:
        return "MEDIUM", "orange", score
    return "LOW", "green", score


def pick_best_value_doc_id(plans: list[dict]) -> str | None:
    """
    Pick 'best value' plan via transparent heuristic:
      maximize (rule_score / (excess + 50))
    """
    if not plans:
        return None

    def value_metric(p: dict) -> float:
        excess = float(p.get("excess", 0) or 0)
        rule = float(p.get("rule_score", 0) or 0)
        return rule / (excess + 50.0)

    best = max(plans, key=value_metric)
    return best.get("doc_id")


def generate_explanation(plan: dict) -> str:
    parts = []

    if plan.get("high_tech"):
        parts.append("offers access to high-tech hospitals")
    if plan.get("full_inpatient"):
        parts.append("includes strong inpatient hospital cover")
    if plan.get("outpatient"):
        parts.append("mentions outpatient benefits")
    if plan.get("maternity"):
        parts.append("includes maternity benefits")
    if int(plan.get("excess", 0) or 0) <= 75:
        parts.append("has a low excess cost")

    if not parts:
        return "This plan matches your profile based on the risk-weighted scoring and retrieved policy evidence."
    if len(parts) == 1:
        return "This plan " + parts[0] + "."

    return "This plan " + ", ".join(parts[:-1]) + " and " + parts[-1] + "."


# ----------------------------
# Sidebar
# ----------------------------
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
        height=110,
    )

    run = st.button("Find best plans", type="primary", use_container_width=True)

    st.divider()
    st.header("Display Options")
    compare_mode = st.toggle("Compare plans side-by-side", value=True)
    show_table = st.toggle("Show comparison table", value=True)
    show_evidence = st.toggle("Show evidence panels", value=True)


profile = {
    "age": age,
    "chronic_conditions": conditions,
    "medication_frequency": meds,
    "specialist_visits_per_year": specialist_visits,
    "hospital_admissions_last_2_years": admissions,
}

# ----------------------------
# Layout
# ----------------------------
colA, colB = st.columns([1.1, 1])

with colA:
    st.subheader("Profile summary")
    st.json(profile)

with colB:
    st.subheader("What this app does")
    st.write(
        "- Retrieves **policy chunks** with a **dense semantic** approach (FAISS embeddings)\n"
        "- Aggregates results to **plan (doc_id)**\n"
        "- Applies **risk-aware, clinically weighted rules** (V3)\n"
        "- Uses dense relevance as a **tie-breaker**, not the main decider\n"
        "- Shows **evidence chunks** for transparency"
    )

st.divider()

# ----------------------------
# Run
# ----------------------------
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

        results = engine.recommend(profile=profile, query=query)

        if not results:
            st.warning("No results returned. Check that metadata_v2.json has doc_id + text/chunk_text fields.")
        else:
            st.subheader("Top recommendations")

            # Risk indicator
            risk_label, risk_color, risk_score = risk_level_badge(profile)
            st.markdown(
                f"**Patient Risk Level:** :{risk_color}[{risk_label}] "
                f"(risk score = {risk_score}; transparent heuristic)"
            )

            top_results = results[:3]
            best_value_doc_id = pick_best_value_doc_id(top_results)

            # Cards
            if compare_mode:
                containers = st.columns(3)
                card_slots = list(zip(containers, enumerate(top_results, start=1)))
            else:
                card_slots = [(st.container(), item) for item in enumerate(top_results, start=1)]

            for container, (rank, r) in card_slots:
                with container:
                    is_best_value = (r.get("doc_id") == best_value_doc_id)
                    if is_best_value:
                        st.success("🏆 Best Value (rule score vs excess)")

                    # Badges (clean)
                    badges = []
                    if r.get("high_tech"):
                        badges.append("🏥 High-tech")
                    if r.get("full_inpatient"):
                        badges.append("✅ Full inpatient")
                    if r.get("outpatient"):
                        badges.append("🩺 Outpatient")
                    if r.get("maternity"):
                        badges.append("🤰 Maternity")

                    title = f"#{rank} — {r.get('display_name', r.get('doc_id', 'Plan'))}"
                    company = r.get("company", "")
                    if company:
                        title += f" ({company})"

                    st.markdown(f"### {title}")
                    if badges:
                        st.markdown(" · ".join(badges))

                    st.write(
                        f"**Total score:** {float(r.get('total_score', 0)):.2f}  \n"
                        f"Rule score: {float(r.get('rule_score', 0)):.2f} · "
                        f"Retrieval score: {float(r.get('retrieval_score_norm', 0)):.3f} (norm)"
                    )

                    st.write(f"**Confidence:** {float(r.get('confidence', 0.0)):.2f}")

                    st.write(
                        f"💶 Excess **€{int(r.get('excess', 0) or 0)}** · "
                        f"🧠 Psych days **{int(r.get('psychiatric_days', 0) or 0)}**"
                    )

                    st.info(generate_explanation(r))

                    with st.expander("Why this plan?"):
                        # Show concise clinical bullets first
                        for bullet in (r.get("clinical_summary", []) or [])[:12]:
                            st.write("• " + bullet)
                        st.divider()
                        # Then show the scoring traces
                        for reason in (r.get("reasons", []) or [])[:12]:
                            st.write("- " + reason)

                    if show_evidence:
                        with st.expander("Evidence (top matching chunks)"):
                            for ev in (r.get("evidence", []) or []):
                                st.markdown(f"**Chunk** (score {float(ev.get('chunk_score', 0)):.3f})")
                                st.write(ev.get("text", ""))
                                st.divider()

            # Comparison table
            if show_table and top_results:
                st.divider()
                st.subheader("Plan comparison")

                rows = []
                for r in top_results:
                    rows.append(
                        {
                            "Plan": r.get("display_name", ""),
                            "Company": r.get("company", ""),
                            "Total": round(float(r.get("total_score", 0) or 0), 2),
                            "Rule": round(float(r.get("rule_score", 0) or 0), 2),
                            "Retrieval (norm)": round(float(r.get("retrieval_score_norm", 0) or 0), 3),
                            "Confidence": round(float(r.get("confidence", 0) or 0), 2),
                            "Excess (€)": int(r.get("excess", 0) or 0),
                            "Psych days": int(r.get("psychiatric_days", 0) or 0),
                            "Inpatient": "✅" if r.get("full_inpatient") else "—",
                            "High-tech": "✅" if r.get("high_tech") else "—",
                            "Outpatient": "✅" if r.get("outpatient") else "—",
                            "Maternity": "✅" if r.get("maternity") else "—",
                            "Best Value": "🏆" if r.get("doc_id") == best_value_doc_id else "",
                        }
                    )

                df = pd.DataFrame(rows)

                def highlight_best_value(row):
                    return [
                        "background-color: rgba(0, 200, 0, 0.15)"
                        if row.get("Best Value", "") == "🏆"
                        else ""
                        for _ in row
                    ]

                try:
                    st.dataframe(df.style.apply(highlight_best_value, axis=1), use_container_width=True)
                except Exception:
                    st.dataframe(df, use_container_width=True)

                st.caption("🏆 Best Value = maximize (rule_score / (excess + 50)) — transparent heuristic.")
