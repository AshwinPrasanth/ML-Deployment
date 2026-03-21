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

"""
Smart RAG Engine — Rule Engine + LLM Reranker (combined)
=========================================================
Two-layer pipeline:

  Layer 1: RULE ENGINE  (fast, deterministic, no API calls)
  ──────────────────────────────────────────────────────────
  - Hard flag filters  → kills plans missing required cover  (maternity/psychiatric/fertility/high_tech)
  - Age soft-check     → warns if outside 18-90
  - Utilisation score  → penalises high excess for frequent claimers
  - Condition filters  → cardiac/cancer via chunk hit-rate proxy
  This runs on ALL retrieved candidates. Ineligible plans are removed before LLM ever sees them.

  Layer 2: LLM RERANKER  (intelligent, holistic, one API call)
  ─────────────────────────────────────────────────────────────
  - Reads surviving plan evidence + user profile
  - Reasons about unknown conditions (neurological, diabetes, etc.)
  - Produces fit score 0-10 + plain-English explanation per plan
  - Handles nuance the rule engine cannot (e.g. "older person needs low excess" across all conditions)

Why both?
  Rule engine  = correctness guarantee (no wrong plan ever surfaces)
  LLM reranker = intelligence        (right plan rises to the top with explanation)
"""

import json, re, os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import OpenAI

# =============================================================================
# SECTION 0 — FREE LLM CONFIG
# =============================================================================
# Get free key at https://console.groq.com  (no credit card)
# export GROQ_API_KEY="gsk_..."

LLM_CLIENT = OpenAI(
    api_key  = os.environ.get("GROQ_API_KEY", "<use groq api>"),
    base_url = "https://api.groq.com/openai/v1",
)
LLM_MODEL = "llama-3.3-70b-versatile"   # llama-3.1-70b-versatile was decommissioned Dec 2024

# Ollama (fully local, no internet):
# LLM_CLIENT = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
# LLM_MODEL  = "llama3"

RAG_CHUNKS_PATH   = "rag_chunks.jsonl"
SUPERSET_PATH     = "MASTER_STRUCTURED_SUPERSET_2026-1.jsonl"
URL_METADATA_PATH = "metadata (1).json"
INDEX_PATH        = "faiss_multi_provider_index.bin"


# =============================================================================
# SECTION 1 — USER PROFILE EXTRACTOR
# =============================================================================

CONDITION_MAP = {
    # name: (query keywords, superset_flag or None)
    "maternity":    (["maternity","pregnancy","pregnant","prenatal","postnatal","antenatal"], "maternity_flag"),
    "psychiatric":  (["psychiatric","mental health","counselling","anxiety","depression","psycho"],  "psychiatric_flag"),
    "fertility":    (["fertility","ivf","infertility","egg freezing"],                        "fertility_flag"),
    "high_tech":    (["biologic","biologics","infusion","high-tech drug","high tech drug"],   "high_tech_flag"),
    "cardiac":      (["cardiac","heart disease","cardiology","coronary","arrhythmia"],        None),
    "cancer":       (["cancer","oncology","oncotype","chemotherapy","tumour","tumor"],        None),
    "neurological": (["neurolog","epilepsy","multiple sclerosis","parkinson","stroke","dementia"], None),
    "diabetes":     (["diabetes","diabetic","insulin","glucose"],                             None),
    "orthopaedic":  (["orthopaedic","joint replacement","hip replacement","knee replacement"],None),
    "respiratory":  (["asthma","copd","respiratory","pulmonary","lung disease"],              None),
    "physiotherapy":(["physiotherapy","physio","rehabilitation"],                             None),
    "renal":        (["kidney","renal","dialysis"],                                           None),
}

def extract_user_profile(query: str) -> dict:
    q = query.lower()
    profile = {
        "raw_query":           query,
        "age":                 None,
        "conditions":          [],     # all detected condition names
        "flagged_conditions":  [],     # only those with a superset flag
        "medication_freq":     None,
        "specialist_visits":   None,
        "hospital_admissions": None,
    }

    for pat in [
        r'\b(\d{1,2})\s*(?:year[s]?\s*old|yo|yrs?)\b',  # 68 year old, 68yo
        r'\bage[d]?\s*(\d{1,2})\b',                       # aged 68, age 68
        r"(?:i'?m|i am)\s+(\d{1,2})\b",                  # I'm 68, I am 68
        r'\bim\s+(\d{1,2})\b',                            # im 68 (no apostrophe)
    ]:
        m = re.search(pat, q)
        if m: profile["age"] = int(m.group(1)); break

    for name, (keywords, flag) in CONDITION_MAP.items():
        if any(kw in q for kw in keywords):
            profile["conditions"].append(name)
            if flag:
                profile["flagged_conditions"].append((name, flag))

    if any(w in q for w in ["daily","every day"]) and \
       any(w in q for w in ["medication","medicine","drug","tablet","prescription"]):
        profile["medication_freq"] = "daily"
    elif "weekly"  in q and "medication" in q: profile["medication_freq"] = "weekly"
    elif "monthly" in q and "medication" in q: profile["medication_freq"] = "monthly"

    m = re.search(r'(\d+)\s*(?:specialist|consultant|outpatient)\s*(?:visit[s]?|time[s]?|appointment[s]?)?', q)
    if m: profile["specialist_visits"] = int(m.group(1))

    m = re.search(r'(\d+)\s*(?:hospital\s*admission[s]?|inpatient\s*stay[s]?|time[s]?\s*(?:in\s*)?hospital)', q)
    if m: profile["hospital_admissions"] = int(m.group(1))

    return profile


# =============================================================================
# SECTION 2 — LOAD DATA
# =============================================================================

print("Loading embedding model...")
emb_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

url_lookup, human_plan_names = {}, {}
try:
    with open(URL_METADATA_PATH) as f:
        for item in json.load(f):
            k = item.get("doc_id","").lower()
            if k:
                url_lookup[k]       = item.get("source_url","No URL Available")
                human_plan_names[k] = item.get("plan_name","Unknown Plan")
    print(f"Loaded {len(url_lookup)} plan URLs.")
except FileNotFoundError:
    print("Warning: metadata not found.")

def get_plan_details(doc_id):
    k = doc_id.lower()
    return human_plan_names.get(k, doc_id.title()), url_lookup.get(k,"No URL Available")

documents_for_faiss, documents_for_bm25, chunk_metadata = [], [], []
plan_chunk_texts = {}   # plan_name -> [lowercased chunk texts]

print("Loading RAG chunks...")
try:
    with open(RAG_CHUNKS_PATH) as f:
        for line in f:
            obj       = json.loads(line)
            provider  = obj.get("insurer","Unknown Provider")
            plan_name = obj.get("plan_name","")
            text      = obj.get("text","")
            documents_for_faiss.append(f"Provider: {provider}. Plan: {plan_name}. Document text: {text}")
            documents_for_bm25.append(text)
            chunk_metadata.append({
                "provider": provider, "doc_id": obj.get("doc_id",""),
                "plan_name": plan_name, "chunk_text": text,
            })
            plan_chunk_texts.setdefault(plan_name, []).append(text.lower())
    print(f"Loaded {len(documents_for_faiss)} chunks across {len(plan_chunk_texts)} plans.")
except FileNotFoundError:
    print("Error: rag_chunks.jsonl not found.")

structured_data = {}
try:
    with open(SUPERSET_PATH) as f:
        for line in f:
            obj = json.loads(line)
            structured_data.setdefault(obj.get("plan_name","").lower(), []).append(obj)
    print(f"Loaded structured data for {len(structured_data)} plans.")
except FileNotFoundError:
    print("Warning: superset not found.")


# =============================================================================
# SECTION 3 — BUILD FAISS + BM25
# =============================================================================

def tokenize(text):
    return re.findall(r'\w+', text.lower())

print("Building FAISS index...")
if documents_for_faiss:
    embeddings = emb_model.encode(
        documents_for_faiss, batch_size=32,
        show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

print("Building BM25 index...")
bm25 = BM25Okapi([tokenize(d) for d in documents_for_bm25])
print("Indices ready.\n")


# =============================================================================
# SECTION 4 — RULE ENGINE  (Layer 1 — deterministic, no LLM)
# =============================================================================

def _find_entries(plan_name: str):
    k = plan_name.lower()
    return next((v for key, v in structured_data.items() if key in k or k in key), None)

def _chunk_hit_rate(plan_name: str, keywords: list) -> float:
    """Fraction of a plan's chunks mentioning any of the keywords."""
    chunks = plan_chunk_texts.get(plan_name, [])
    if not chunks: return 0.0
    hits = sum(1 for c in chunks if any(kw in c for kw in keywords))
    return hits / len(chunks)

def rule_engine(plan_name: str, profile: dict) -> tuple[float, bool, list]:
    """
    Returns (rule_score: float, passes: bool, rule_notes: list[str])

    passes=False  → plan is hard-rejected, excluded before LLM sees it
    rule_score    → 0-1, fed as context into final blending
    rule_notes    → human-readable explanation of what the rules found
    """
    entries  = _find_entries(plan_name)
    notes    = []

    # ── A. HARD FLAG FILTERS ──────────────────────────────────────────────────
    # Only fire when the user explicitly needs that condition AND the flag is False
    if entries:
        for condition, flag in profile.get("flagged_conditions", []):
            has_cover = any(e.get(flag) for e in entries)
            if not has_cover:
                return 0.0, False, [f"Hard rejected: no {condition} cover ({flag}=False)."]
            notes.append(f"{condition}: covered ({flag}=True)")

    # ── B. AGE SOFT CHECK ─────────────────────────────────────────────────────
    age = profile.get("age")
    if age is not None and not (18 <= age <= 90):
        notes.append(f"Age {age} outside standard 18-90 band — verify eligibility.")

    # ── C. UTILISATION SCORE ──────────────────────────────────────────────────
    # Map user's usage intensity to plan numeric fields
    med_freq   = profile.get("medication_freq")
    spec_visits= profile.get("specialist_visits")
    hosp_admit = profile.get("hospital_admissions")

    intensity, signals = 0.0, 0
    if med_freq:
        intensity += {"daily": 1.0, "weekly": 0.6, "monthly": 0.3}[med_freq]; signals += 1
    if spec_visits  is not None: intensity += min(spec_visits  / 20.0, 1.0); signals += 1
    if hosp_admit   is not None: intensity += min(hosp_admit   / 10.0, 1.0); signals += 1
    if signals: intensity /= signals

    util_score = 0.5   # neutral default
    if entries and signals > 0:
        excess_vals = [float(e["excess_amount"])      for e in entries if e.get("excess_amount")      is not None]
        day_vals    = [float(e["day_limit"])           for e in entries if e.get("day_limit")          is not None]
        cap_vals    = [float(e["limit_amount"])        for e in entries if e.get("limit_amount")       is not None]
        cov_vals    = [float(e["coverage_percentage"]) for e in entries if e.get("coverage_percentage") is not None]

        avg_excess = np.mean(excess_vals) if excess_vals else 500
        avg_day    = np.mean(day_vals)    if day_vals    else 30
        avg_cap    = np.mean(cap_vals)    if cap_vals    else 30000
        avg_cov    = np.mean(cov_vals)    if cov_vals    else 50

        # High utilisation = penalise excess, reward day-limit and cap
        excess_score = 1.0 / (1.0 + avg_excess / 250)
        day_score    = min(avg_day / 100.0, 1.0)
        cap_score    = min(avg_cap / 100000.0, 1.0)
        cov_score    = avg_cov / 100.0

        util_score = (
            excess_score * (0.4 * intensity + 0.1) +
            day_score    * (0.3 * intensity + 0.05) +
            cap_score    * 0.25 +
            cov_score    * 0.20
        )
        notes.append(
            f"Utilisation intensity={intensity:.2f} | "
            f"avg_excess=€{avg_excess:.0f} | avg_days={avg_day:.0f} | "
            f"util_score={util_score:.3f}"
        )

    # ── D. CONDITION CHUNK-HIT PROXY (no flag conditions) ────────────────────
    chunk_boost = 0.0
    for condition in profile.get("conditions", []):
        flag = CONDITION_MAP[condition][1]
        if flag:
            continue   # already handled by hard filter above
        keywords = CONDITION_MAP[condition][0]
        hit_rate = _chunk_hit_rate(plan_name, keywords)
        chunk_boost += hit_rate * 0.1   # max +0.10 per unflagged condition
        if hit_rate > 0:
            notes.append(f"{condition}: {hit_rate:.0%} chunk hit-rate (proxy, no structured flag)")
        else:
            notes.append(f"{condition}: NOT found in plan text — data gap, LLM will assess")

    # ── E. COMBINE ────────────────────────────────────────────────────────────
    rule_score = min(util_score + chunk_boost, 1.0)
    return round(rule_score, 4), True, notes


def get_numeric_summary(plan_name: str) -> str:
    """Human-readable numeric snapshot passed to LLM as context."""
    entries = _find_entries(plan_name)
    if not entries: return "No structured numeric data available."

    def safe_avg(vals): return f"€{np.mean(vals):.0f}" if vals else "N/A"
    def safe_min(vals): return f"€{min(vals):.0f}"     if vals else "N/A"

    excess = [float(e["excess_amount"])      for e in entries if e.get("excess_amount")      is not None]
    caps   = [float(e["limit_amount"])       for e in entries if e.get("limit_amount")       is not None]
    days   = [float(e["day_limit"])          for e in entries if e.get("day_limit")          is not None]
    cov    = [float(e["coverage_percentage"])for e in entries if e.get("coverage_percentage") is not None]
    copay  = [float(e["copayment_amount"])   for e in entries if e.get("copayment_amount")    is not None]
    flags  = {
        "maternity":    any(e.get("maternity_flag")    for e in entries),
        "psychiatric":  any(e.get("psychiatric_flag")  for e in entries),
        "fertility":    any(e.get("fertility_flag")    for e in entries),
        "high_tech":    any(e.get("high_tech_flag")    for e in entries),
        "international":any(e.get("international_flag")for e in entries),
    }
    return "\n".join([
        f"Excess:     min {safe_min(excess)} / avg {safe_avg(excess)}",
        f"Cap:        avg {safe_avg(caps)}",
        f"Day limits: avg {np.mean(days):.0f} days" if days else "Day limits: N/A",
        f"Coverage:   avg {np.mean(cov):.0f}%"     if cov  else "Coverage: N/A",
        f"Copayment:  avg {safe_avg(copay)}",
        "Flags: " + ", ".join(f"{k}={'YES' if v else 'NO'}" for k, v in flags.items()),
    ])


# =============================================================================
# SECTION 5 — RRF RETRIEVAL
# =============================================================================

def retrieve_candidates(query: str, k: int = 60, top_n: int = 10) -> list:
    q_lower = query.lower()
    mentioned_plan = None
    for plan in set(m["plan_name"] for m in chunk_metadata):
        clean = plan.replace("(Table of Cover)","").replace("(Table of Benefits)","").strip().lower()
        if clean and clean in q_lower:
            mentioned_plan = plan; break

    search_query = query
    if mentioned_plan:
        target = re.escape(mentioned_plan.replace("(Table of Cover)","").replace("(Table of Benefits)","").strip())
        search_query = re.sub(target, "", search_query, flags=re.IGNORECASE).strip()

    prefixed = "Represent this sentence for searching relevant passages: " + search_query
    q_emb    = emb_model.encode([prefixed], normalize_embeddings=True)
    _, f_idx = index.search(np.array(q_emb, dtype="float32"), k)

    b_scores = bm25.get_scores(tokenize(search_query))
    b_idx    = np.argsort(b_scores)[::-1][:k]

    RRF_K, rrf = 60, {}
    for rank, idx in enumerate(f_idx[0]):
        rrf[int(idx)] = rrf.get(int(idx), 0) + 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(b_idx):
        if b_scores[idx] > 0:
            rrf[int(idx)] = rrf.get(int(idx), 0) + 1.0 / (RRF_K + rank + 1)

    plan_chunks, provider_map, doc_id_map = {}, {}, {}
    for idx, score in rrf.items():
        meta = chunk_metadata[idx]
        pn   = meta["plan_name"]
        if pn not in plan_chunks:
            plan_chunks[pn]  = []
            provider_map[pn] = meta["provider"]
            doc_id_map[pn]   = meta["doc_id"]
        plan_chunks[pn].append({"text": meta["chunk_text"], "score": score})

    candidates = []
    for plan_name, chunks in plan_chunks.items():
        if mentioned_plan and plan_name != mentioned_plan: continue
        chunks.sort(key=lambda x: x["score"], reverse=True)
        doc_id = doc_id_map[plan_name]
        human_name, source_url = get_plan_details(doc_id)
        if human_name == doc_id.title(): human_name = plan_name

        candidates.append({
            "plan_name":       human_name,
            "internal_key":    plan_name,
            "provider":        provider_map[plan_name],
            "source_url":      source_url,
            "rrf_score":       round(chunks[0]["score"], 4),
            "evidence":        "\n\n---\n\n".join(c["text"] for c in chunks[:3]),
            "numeric_summary": get_numeric_summary(plan_name),
        })

    candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
    return candidates[:top_n]


# =============================================================================
# SECTION 6 — LLM RERANKER  (Layer 2 — intelligent, holistic)
# =============================================================================

RERANKER_SYSTEM = """You are an expert Irish health insurance advisor.
You will receive a user's health profile and a set of pre-filtered candidate
insurance plans (plans that already passed eligibility checks) with real policy
text and numeric data.

IMPORTANT: In your JSON response, use the EXACT plan name as given in the
=== PLAN N: <name> === header. Do not shorten or paraphrase plan names.

Your job is intelligent fit scoring — reason holistically about this specific person:
- Older person (60+): low excess critical, high day-limits, strong inpatient cover
- Daily medication / chronic condition: each claim costs the excess — low excess is essential
- High specialist visits: outpatient depth, copayment per visit, annual outpatient cap
- Multiple hospital admissions: inpatient day-limits, private hospital access
- Conditions not in structured data (neurological, diabetes etc): infer from policy text
  whether chronic specialist care, monitoring, and medication cover exists
- Maternity: explicit maternity benefits, consultant fees, postnatal support
- Psychiatric: inpatient day limits, counselling sessions, substance abuse policy

Return ONLY valid JSON — no markdown fences, no explanation outside JSON.
A JSON array, sorted best-fit first, each element:
{
  "plan_name": "<exact name from the === PLAN N: header ===",
  "fit_score": <integer 0-10>,
  "fit_summary": "<2-3 sentence plain English — why this plan suits or doesn't suit this person>",
  "strengths": ["<specific benefit that matches their need>", "..."],
  "weaknesses": ["<specific gap or concern for this person>", "..."],
  "coverage_gaps": ["<conditions mentioned that plan seems silent or weak on>"]
}"""


def llm_rerank(candidates: list, profile: dict) -> list:
    lines = [f"User query: {profile['raw_query']}"]
    if profile["age"]:              lines.append(f"Age: {profile['age']}")
    if profile["conditions"]:       lines.append(f"Health conditions: {', '.join(profile['conditions'])}")
    if profile["medication_freq"]:  lines.append(f"Medication frequency: {profile['medication_freq']}")
    if profile["specialist_visits"] is not None:
        lines.append(f"Specialist visits/year: {profile['specialist_visits']}")
    if profile["hospital_admissions"] is not None:
        lines.append(f"Hospital admissions (last 2 yrs): {profile['hospital_admissions']}")

    # Cap at 7 to keep prompt within token limits — rule engine already filtered worst plans
    candidates_for_llm = candidates[:7]
    plan_blocks = []
    for i, c in enumerate(candidates_for_llm, 1):
        plan_blocks.append(
            f"=== PLAN {i}: {c['plan_name']} ({c['provider']}) ===\n"
            f"Rule engine score: {c['rule_score']} | Notes: {'; '.join(c['rule_notes'][:2])}\n"
            f"Numeric data:\n{c['numeric_summary']}\n\n"
            f"Policy text:\n{c['evidence'][:800]}"
        )

    user_message = (
        f"USER PROFILE:\n" + "\n".join(lines) +
        f"\n\nCANDIDATE PLANS (pre-filtered by rule engine):\n\n" +
        "\n\n".join(plan_blocks)
    )

    print(f"  Sending {len(candidates_for_llm)} plans to {LLM_MODEL} (max_tokens=4000)...")
    response = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL, max_tokens=4000, temperature=0.1,
        messages=[
            {"role": "system", "content": RERANKER_SYSTEM},
            {"role": "user",   "content": user_message},
        ],
    )
    raw = response.choices[0].message.content.strip()

    # --- JSON repair: handle truncated responses ---
    def repair_json(text):
        # Strip markdown fences
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Truncated mid-array: find last complete object and close the array
        last_complete = text.rfind('},')
        if last_complete == -1:
            last_complete = text.rfind('}')
        if last_complete != -1:
            text = text[:last_complete + 1] + ']'
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        # Last resort: extract whatever complete objects exist
        objects = re.findall(r'\{[^{}]*"plan_name"[^{}]*"fit_score"[^{}]*\}', text, re.DOTALL)
        if objects:
            return json.loads('[' + ','.join(objects) + ']')
        raise ValueError(f"Could not parse LLM response: {text[:200]}")

    llm_results = repair_json(raw)

    # Fuzzy match: LLM often shortens plan names — match by substring both ways
    def fuzzy_match(candidate_name, results):
        cn = candidate_name.lower()
        for r in results:                                    # exact
            if r["plan_name"].lower() == cn: return r
        for r in results:                                    # substring
            ln = r["plan_name"].lower()
            if ln in cn or cn in ln: return r
        for r in results:                                    # word overlap >50%
            words = [w for w in r["plan_name"].lower().split() if len(w) > 3]
            if words and sum(1 for w in words if w in cn) / len(words) >= 0.5:
                return r
        return {}

    enriched = []
    for c in candidates_for_llm:   # only match plans that were actually sent to LLM
        llm = fuzzy_match(c["plan_name"], llm_results)
        enriched.append({
            **c,
            "fit_score":     llm.get("fit_score", 0),
            "fit_summary":   llm.get("fit_summary", "Not evaluated."),
            "strengths":     llm.get("strengths", []),
            "weaknesses":    llm.get("weaknesses", []),
            "coverage_gaps": llm.get("coverage_gaps", []),
        })

    enriched.sort(key=lambda x: x["fit_score"], reverse=True)
    return enriched


# =============================================================================
# SECTION 7 — FULL PIPELINE
# =============================================================================

def smart_search(query: str, top_n: int = 5) -> list:
    """
    1. Extract user profile
    2. RRF retrieval → up to 10 candidates
    3. Rule engine → hard-reject ineligible plans, score survivors
    4. LLM reranker → intelligent fit scoring on survivors only
    5. Return top_n with full explanation
    """
    print(f"\n{'='*65}")
    print(f"Query: {query}")

    profile = extract_user_profile(query)
    print(f"Profile: age={profile['age']} | conditions={profile['conditions']} | "
          f"freq={profile['medication_freq']} | visits={profile['specialist_visits']} | "
          f"admissions={profile['hospital_admissions']}")

    # Step 1: retrieve
    candidates = retrieve_candidates(query, top_n=10)
    print(f"Retrieved {len(candidates)} candidates via RRF.")

    # Step 2: rule engine filters + scores each candidate
    survivors = []
    rejected  = []
    for c in candidates:
        rule_score, passes, notes = rule_engine(c["internal_key"], profile)
        if passes:
            c["rule_score"] = rule_score
            c["rule_notes"] = notes
            survivors.append(c)
        else:
            rejected.append((c["plan_name"], notes[0] if notes else "rejected"))

    if rejected:
        print(f"Rule engine rejected {len(rejected)} plans:")
        for name, reason in rejected:
            print(f"  x {name}: {reason}")

    print(f"Rule engine passed {len(survivors)} plans to LLM.")

    if not survivors:
        print("No plans passed the rule engine for this query.")
        return []

    # Step 3: LLM reranks survivors
    results = llm_rerank(survivors, profile)
    return results[:top_n]


# =============================================================================
# SECTION 8 — DEMO
# =============================================================================

if __name__ == "__main__":
    queries = [
        # unknown condition — LLM infers, rule engine can't help, but won't wrongly reject
        "I'm 68 with Parkinson's. I see a neurologist monthly and take daily medication.",

        # maternity — rule engine will hard-reject plans with maternity_flag=False
        "I'm pregnant. Looking for the best maternity cover with postnatal support.",

        # high utilisation — rule engine scores low excess plans higher
        "Heart condition, diabetic, hospitalised 3 times last year, 10 specialist visits.",

        # young healthy — minimal needs
        "I'm 25 and healthy. Want the most affordable basic cover.",
    ]

    for query in queries[:2]:
        results = smart_search(query)

        print(f"\nTop {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            print(f"  Rank {i}: {r['plan_name']} ({r['provider']})")
            print(f"  Fit: {r['fit_score']}/10 | Rule score: {r['rule_score']}")
            print(f"  {r['fit_summary']}")
            if r["strengths"]:
                print(f"  + {' | '.join(r['strengths'][:2])}")
            if r["weaknesses"]:
                print(f"  - {' | '.join(r['weaknesses'][:2])}")
            if r["coverage_gaps"]:
                print(f"  ! Gaps: {' | '.join(r['coverage_gaps'])}")
            print(f"  {r['source_url']}\n")
