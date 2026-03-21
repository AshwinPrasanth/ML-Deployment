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

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# DISEASE RULES
# =========================================================
DISEASE_RULES: Dict[str, Dict[str, Any]] = {
    "heart_disease": {
        "keywords": ["cardiac", "angioplasty", "stent", "heart"],
        "requires_high_tech": True,
        "priority": "critical",
    },
    "cancer": {
        "keywords": ["oncology", "chemotherapy", "radiotherapy", "tumour", "cancer"],
        "requires_high_tech": True,
        "priority": "critical",
    },
    "neurological_disorder": {
        "keywords": ["neurology", "mri", "ct scan", "brain", "neuro"],
        "requires_high_tech": True,
        "priority": "critical",
    },
    "pregnancy": {
        "keywords": ["maternity", "antenatal", "postnatal", "delivery", "home birth", "obstetric"],
        "priority": "specialised",
    },
    "psychiatric_disorder": {
        "keywords": ["psychiatric", "mental health", "psychotherapy", "counselling"],
        "requires_psych_days": True,
        "priority": "moderate",
    },
    "diabetes": {
        "keywords": ["diabetes", "insulin", "endocrin"],
        "priority": "chronic",
    },
}

# =========================================================
# DISEASE PRIORITY WEIGHTS
# =========================================================
DISEASE_PRIORITY = {
    "critical": {"high_tech": 18, "inpatient": 10, "outpatient": 6, "excess_mult": 2.0},
    "specialised": {"maternity": 15, "inpatient": 8, "outpatient": 4, "excess_mult": 1.5},
    "moderate": {"psychiatric": 12, "inpatient": 6, "outpatient": 4, "excess_mult": 1.2},
    "chronic": {"outpatient": 8, "inpatient": 6, "excess_mult": 1.0},
}

# =========================================================
# Helpers
# =========================================================
def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _normalize_01_dict(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype="float32")
    mn = float(vals.min())
    mx = float(vals.max())
    if mx - mn == 0:
        return {k: 0.0 for k in d.keys()}
    return {k: float((v - mn) / (mx - mn)) for k, v in d.items()}


def _clean_name_from_doc_id(doc_id: str) -> str:
    if not doc_id:
        return ""
    s = doc_id.replace("_", " ")
    s = re.sub(r"\d{4}-\d{2}-\d{2}", "", s)
    s = s.replace("table of cover", "").replace("table of benefits", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def _infer_company_from_name(display_name: str) -> str:
    dn = (display_name or "").lower()
    if "irish life" in dn or "health plan" in dn or "benefit" in dn or "first cover" in dn or "select" in dn or "horizon" in dn:
        return "Irish Life Health"
    if "vhi" in dn or "health access" in dn or "company plan" in dn:
        return "VHI"
    if "laya" in dn or "inspire" in dn or "prime" in dn or "first & family" in dn:
        return "Laya Healthcare"
    if "level" in dn or "plan a" in dn or "plan b" in dn or "plan c" in dn or "plan d" in dn:
        return "Level Health"
    return ""


# =========================================================
# Query preference parsing (V2)
# =========================================================
def parse_query_prefs(q: str) -> Dict[str, Any]:
    """
    Parses explicit intent signals. These influence ranking and optionally enforce constraints.
    - want_inpatient vs avoid_inpatient (e.g., "no inpatient")
    - want_outpatient / outpatient_only
    - want_maternity
    - want_high_tech (enforce)
    - low_excess
    """
    q = (q or "").lower().strip()

    avoid_inpatient = any(k in q for k in [
        "no inpatient",
        "without inpatient",
        "dont need inpatient",
        "don't need inpatient",
        "avoid inpatient",
        "outpatient only",
        "outpatient-only",
        "out patient only",
    ])

    outpatient_only = any(k in q for k in [
        "outpatient only",
        "outpatient-only",
        "out patient only",
        "no inpatient",
        "without inpatient",
    ])

    return {
        "low_excess": any(k in q for k in ["low excess", "cheap", "minimum excess", "no excess", "lowest excess"]),
        "want_inpatient": any(k in q for k in ["inpatient", "hospital cover", "private hospital", "ward", "room"]) and not avoid_inpatient,
        "avoid_inpatient": avoid_inpatient,
        "want_outpatient": ("outpatient" in q or "out-patient" in q or "out patient" in q),
        "outpatient_only": outpatient_only,
        "want_maternity": ("maternity" in q or "pregnan" in q or "antenatal" in q or "postnatal" in q),
        "want_high_tech": ("high-tech" in q or "high tech" in q or "hi-tech" in q),
    }


# =========================================================
# Feature extraction from plan text (robust)
# =========================================================
def extract_features(text: str) -> Dict[str, Any]:
    """
    Keys (Streamlit schema):
      excess, psychiatric_days, full_inpatient, high_tech_available, outpatient, maternity
    """
    t = (text or "").lower()

    inpatient = any(k in t for k in [
        "inpatient consultant fees",
        "inpatient",
        "private hospital inpatient",
        "hospital accommodation",
        "semi-private room",
        "private room",
        "day case",
        "in-patient",
    ]) and "not covered" not in t

    high_tech = any(k in t for k in [
        "high-tech hospital",
        "high tech hospital",
        "hi-tech hospital",
    ]) and "not covered" not in t

    outpatient = any(k in t for k in [
        "out-patient",
        "outpatient",
        "out patient",
        "out-patient benefits",
        "outpatient benefits",
    ]) and "not covered" not in t

    maternity = any(k in t for k in [
        "maternity",
        "antenatal",
        "postnatal",
        "delivery",
        "home birth",
        "obstetric",
    ]) and "not covered" not in t

    # psych days (max)
    days = re.findall(r"(\d+)\s+days", t)
    psych_days = max([_safe_int(x, 0) for x in days], default=0)

    # excess
    excess_match = re.search(r"€\s?(\d{1,5})\s*excess", t)
    if not excess_match:
        excess_match = re.search(r"excess\s*€\s?(\d{1,5})", t)
    excess = _safe_int(excess_match.group(1), 0) if excess_match else 0

    # cardiac copay (optional)
    copay_match = re.search(r"€\s?(\d{1,5})[^\n]{0,80}cardiac", t)
    cardiac_copay = _safe_int(copay_match.group(1), 0) if copay_match else 0

    return {
        "full_inpatient": bool(inpatient),
        "high_tech_available": bool(high_tech),
        "outpatient": bool(outpatient),
        "maternity": bool(maternity),
        "psychiatric_days": int(psych_days),
        "excess": int(excess),
        "cardiac_copay": int(cardiac_copay),
    }


# =========================================================
# Config
# =========================================================
@dataclass
class EngineConfig:
    metadata_path: str = os.path.join(BASE_DIR, "ConstructionJSON", "metadata.json")
    faiss_path: str = os.path.join(BASE_DIR, "ConstructionJSON", "faiss_index.bin")
    registry_path: str = os.path.join(BASE_DIR, "ConstructionJSON", "data", "doc_registry.json")

    embed_model: str = "mixedbread-ai/mxbai-embed-large-v1"

    dense_top_k: int = 300
    evidence_per_plan: int = 5
    top_n_plans: int = 3


# =========================================================
# Engine
# =========================================================
class RAGEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg

        # --- Load metadata ---
        if not os.path.exists(cfg.metadata_path):
            raise FileNotFoundError(f"metadata not found: {cfg.metadata_path}")

        with open(cfg.metadata_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        if not isinstance(loaded, list):
            raise RuntimeError("metadata_v2.json must be a JSON list aligned to FAISS rows.")

        self.metadata: List[Dict[str, Any]] = [m for m in loaded if isinstance(m, dict)]
        self.texts: List[str] = [
            (m.get("chunk_text") or m.get("text") or m.get("chunk") or m.get("content") or "")
            for m in self.metadata
        ]
        self.plan_ids: List[str] = sorted(list({m.get("doc_id") for m in self.metadata if m.get("doc_id")}))

        # --- Load registry (plan_name/company) ---
        self.registry: Dict[str, Any] = {}
        if os.path.exists(cfg.registry_path):
            try:
                with open(cfg.registry_path, "r", encoding="utf-8") as f:
                    reg_loaded = json.load(f)
                if isinstance(reg_loaded, dict):
                    self.registry = reg_loaded
                elif isinstance(reg_loaded, list):
                    tmp = {}
                    for item in reg_loaded:
                        if isinstance(item, dict) and "doc_id" in item:
                            tmp[item["doc_id"]] = item
                    self.registry = tmp
            except Exception:
                self.registry = {}

        # --- Embedding model ---
        self.model = SentenceTransformer(cfg.embed_model)

        # --- FAISS ---
        self.index = self._load_or_build_faiss()

    def _pref(self, s: str) -> str:
        return "Represent this sentence for searching relevant passages: " + (s or "")

    def _load_or_build_faiss(self) -> faiss.Index:
        probe = self.model.encode([self._pref("probe")], normalize_embeddings=True)
        probe = np.array(probe).astype("float32")
        expected_dim = int(probe.shape[1])

        if os.path.exists(self.cfg.faiss_path):
            index = faiss.read_index(self.cfg.faiss_path)
            if int(index.ntotal) == len(self.texts) and int(index.d) == expected_dim:
                return index
            try:
                os.remove(self.cfg.faiss_path)
            except OSError:
                pass

        emb = self.model.encode(self.texts, normalize_embeddings=True)
        emb = np.array(emb).astype("float32")
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, self.cfg.faiss_path)
        return index

    # ----------------------------
    # Dense plan relevance
    # ----------------------------
    def compute_dense_scores(self, query: str) -> Dict[str, float]:
        q_emb = self.model.encode([self._pref(query)], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype("float32")

        k = min(self.cfg.dense_top_k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, k)

        plan_scores: Dict[str, List[float]] = defaultdict(list)
        for s, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            pid = self.metadata[int(idx)].get("doc_id")
            if pid:
                plan_scores[pid].append(float(s))

        mean_scores = {pid: float(np.mean(vals)) for pid, vals in plan_scores.items()}
        for pid in self.plan_ids:
            mean_scores.setdefault(pid, 0.0)

        return _normalize_01_dict(mean_scores)

    # ----------------------------
    # Evidence retrieval
    # ----------------------------
    def retrieve_plan_chunks(self, plan_id: str, query: str, top_n: int = 5) -> List[Tuple[float, str]]:
        q_emb = self.model.encode([self._pref(query)], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype("float32")

        k = min(self.cfg.dense_top_k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, k)

        results: List[Tuple[float, str]] = []
        for s, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            m = self.metadata[int(idx)]
            if m.get("doc_id") != plan_id:
                continue
            txt = m.get("chunk_text") or m.get("text") or m.get("chunk") or m.get("content") or ""
            if txt:
                results.append((float(s), txt))
            if len(results) >= top_n:
                break
        return results

    # ----------------------------
    # Merge plan text for extraction
    # ----------------------------
    def merge_plan_text(self, plan_id: str) -> str:
        parts = []
        for m in self.metadata:
            if m.get("doc_id") == plan_id:
                parts.append(
                    (m.get("chunk_text") or m.get("text") or m.get("chunk") or m.get("content") or "").lower()
                )
        return " ".join(parts)

    def _clinical_summary(self, features: Dict[str, Any]) -> List[str]:
        summary = []
        if features.get("high_tech_available"):
            summary.append("High-tech hospital access available")
        if features.get("full_inpatient"):
            summary.append("Strong inpatient hospital cover")
        if features.get("outpatient"):
            summary.append("Outpatient benefits mentioned")
        if features.get("maternity"):
            summary.append("Maternity benefits mentioned")

        ex = int(features.get("excess", 0) or 0)
        if ex == 0:
            summary.append("No excess detected (or not stated in chunks)")
        elif ex <= 75:
            summary.append("Low excess")
        elif ex <= 200:
            summary.append("Medium excess")
        else:
            summary.append("High excess")

        pdays = int(features.get("psychiatric_days", 0) or 0)
        if pdays >= 100:
            summary.append("Psychiatric cover days detected (>=100)")
        return summary

    # ----------------------------
    # Clinical score (V2) — Query controls + safe ordering
    # ----------------------------
    def compute_clinical_score(
        self, plan_id: str, profile: Dict[str, Any], query: str
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:

        merged = self.merge_plan_text(plan_id)
        features = extract_features(merged)

        # Infer missing flags from top evidence FIRST (prevents false negatives)
        evidence_flags = self.retrieve_plan_chunks(
            plan_id,
            "table of benefits hospital cover inpatient outpatient maternity high-tech excess psychiatric days",
            top_n=12,
        )
        evidence_text = " ".join(t.lower() for _, t in evidence_flags)
        inferred = extract_features(evidence_text)

        for k in ["full_inpatient", "high_tech_available", "outpatient", "maternity"]:
            if not bool(features.get(k, False)) and bool(inferred.get(k, False)):
                features[k] = True

        conditions = profile.get("chronic_conditions") or []

        breakdown: Dict[str, float] = {}
        score = 0.0

        # Base excess multiplier from diseases
        excess_mult = 1.0
        for cond in conditions:
            rule = DISEASE_RULES.get(cond)
            if rule:
                priority = DISEASE_PRIORITY.get(rule.get("priority", ""), {})
                excess_mult = max(excess_mult, float(priority.get("excess_mult", 1.0)))

        # ----------------------------
        # Query preferences (V2)
        # ----------------------------
        prefs = parse_query_prefs(query)

        # Query hard requirement: if user requests high-tech, enforce it
        if prefs["want_high_tech"] and not features.get("high_tech_available"):
            return -1000.0, {"failed_query_high_tech_requirement": -1000.0}, features

        # Query soft preferences: bonuses/penalties
        query_bonus = 0.0
        query_penalty = 0.0

        if prefs["want_inpatient"] and features.get("full_inpatient"):
            query_bonus += 4.0

        if prefs["want_outpatient"] and features.get("outpatient"):
            query_bonus += 3.0

        if prefs["want_maternity"] and features.get("maternity"):
            query_bonus += 4.0

        # NEW: "no inpatient" / outpatient-only intent
        # Penalize plans with inpatient emphasis when user explicitly avoids it.
        if prefs["avoid_inpatient"] and features.get("full_inpatient"):
            query_penalty += 5.0

        # If outpatient-only and plan lacks outpatient, penalize
        if prefs["outpatient_only"] and not features.get("outpatient"):
            query_penalty += 6.0

        # If user says low excess, make excess matter more
        if prefs["low_excess"]:
            excess_mult *= 1.5

        if query_bonus > 0:
            score += query_bonus
            breakdown["query_bonus"] = round(query_bonus, 2)

        if query_penalty > 0:
            score -= query_penalty
            breakdown["query_penalty"] = -round(query_penalty, 2)

        # ----------------------------
        # OPTIONAL: profile utilisation (sliders) affects inpatient/excess
        # (This makes your non-condition sliders actually change ranking)
        # ----------------------------
        age = int(profile.get("age", 0) or 0)
        med_freq = (profile.get("medication_frequency") or "none").lower()
        spec_visits = int(profile.get("specialist_visits_per_year", 0) or 0)
        admissions = int(profile.get("hospital_admissions_last_2_years", 0) or 0)

        utilisation = 0.0
        utilisation += 0.5 * admissions
        utilisation += 0.2 * spec_visits
        utilisation += {"none": 0.0, "monthly": 0.5, "weekly": 1.0, "daily": 1.5}.get(med_freq, 0.0)
        if age >= 60:
            utilisation += 1.0
        elif age >= 45:
            utilisation += 0.5

        if utilisation >= 2.0:
            if features.get("full_inpatient"):
                score += 3.0
                breakdown["utilisation_inpatient_bonus"] = 3.0
            excess_mult *= 1.25
            breakdown["utilisation_excess_mult"] = round(excess_mult, 2)

        # ----------------------------
        # HARD requirements per condition (clinical safety)
        # ----------------------------
        for cond in conditions:
            rule = DISEASE_RULES.get(cond)
            if not rule:
                continue

            if rule.get("requires_high_tech") and not features.get("high_tech_available"):
                return -1000.0, {"failed_high_tech_requirement": -1000.0}, features

            if rule.get("requires_psych_days") and int(features.get("psychiatric_days", 0) or 0) < 100:
                return -1000.0, {"failed_psych_days_requirement": -1000.0}, features

            if cond == "pregnancy" and not features.get("maternity"):
                return -1000.0, {"failed_maternity_requirement": -1000.0}, features

        # ----------------------------
        # Apply weights per condition
        # ----------------------------
        for cond in conditions:
            rule = DISEASE_RULES.get(cond)
            if not rule:
                continue

            priority = DISEASE_PRIORITY.get(rule.get("priority", ""), {})

            if features.get("high_tech_available"):
                score += float(priority.get("high_tech", 0.0))
            if features.get("full_inpatient"):
                score += float(priority.get("inpatient", 0.0))
            if features.get("outpatient"):
                score += float(priority.get("outpatient", 0.0))

            # pregnancy special handling
            if cond == "pregnancy":
                if features.get("maternity"):
                    score += float(priority.get("maternity", 0.0))
                    if features.get("full_inpatient"):
                        score += 4.0
                else:
                    score -= 20.0

            # psychiatric signal
            if cond == "psychiatric_disorder":
                pdays = float(features.get("psychiatric_days", 0) or 0)
                if pdays > 0:
                    score += min(pdays / 25.0, float(priority.get("psychiatric", 0.0)))

            # keyword evidence bonus (small)
            chunks = self.retrieve_plan_chunks(plan_id, f"Coverage for {cond.replace('_',' ')}", top_n=5)
            kw_hits = 0
            for _, txt in chunks:
                tl = (txt or "").lower()
                for kw in rule.get("keywords", []):
                    if kw in tl:
                        kw_hits += 1
            score += min(kw_hits * 0.8, 6.0)

        # ----------------------------
        # Cost penalties (once)
        # ----------------------------
        excess = float(features.get("excess", 0) or 0.0)
        excess_penalty = (excess / 25.0) * excess_mult
        score -= excess_penalty
        breakdown["excess_penalty"] = -round(excess_penalty, 2)

        # Optional cardiac copay penalty
        if "heart_disease" in conditions:
            copay = float(features.get("cardiac_copay", 0) or 0.0)
            copay_penalty = copay / 40.0
            score -= copay_penalty
            breakdown["cardiac_copay_penalty"] = -round(copay_penalty, 2)

        breakdown["clinical_score"] = round(score, 2)
        return float(score), breakdown, features

    # ----------------------------
    # Recommend (Streamlit schema)
    # ----------------------------
    def recommend(self, profile: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        dense_scores = self.compute_dense_scores(query)
        results: List[Dict[str, Any]] = []

        for plan_id in self.plan_ids:
            clinical_score, breakdown, features = self.compute_clinical_score(plan_id, profile, query)
            if clinical_score <= -100:
                continue

            dense_norm = float(dense_scores.get(plan_id, 0.0))

            # Combine: clinical dominates, dense supports (still query-sensitive via prefs + dense)
            final_score = (clinical_score * 0.75) + (dense_norm * 10.0 * 0.25)

            # Cost-adjusted value index (final)
            excess = float(features.get("excess", 0) or 0.0)
            value_index = final_score / (1.0 + excess / 100.0)

            # Display name + company
            reg = self.registry.get(plan_id, {}) if isinstance(self.registry, dict) else {}
            display_name = reg.get("plan_name") or reg.get("title") or _clean_name_from_doc_id(plan_id) or plan_id
            company = reg.get("company") or reg.get("provider") or _infer_company_from_name(display_name) or ""

            # Confidence (0..1)
            confidence = float(min(1.0, max(0.0, (clinical_score / 40.0) + (dense_norm * 0.35))))

            # Evidence
            ev_chunks = self.retrieve_plan_chunks(plan_id, query, top_n=self.cfg.evidence_per_plan)
            evidence = []
            for s, t in ev_chunks:
                evidence.append({"chunk_score": float(s), "text": t, "page": None, "source_pdf": "", "section": ""})

            clinical_summary = self._clinical_summary(features)

            results.append(
                {
                    "doc_id": plan_id,
                    "display_name": display_name,
                    "company": company,

                    "total_score": float(value_index),
                    "rule_score": float(clinical_score),
                    "retrieval_score_norm": float(dense_norm),
                    "retrieval_score": float(dense_norm),

                    "confidence": confidence,
                    "clinical_summary": clinical_summary,

                    "excess": int(features.get("excess", 0) or 0),
                    "psychiatric_days": int(features.get("psychiatric_days", 0) or 0),

                    "full_inpatient": bool(features.get("full_inpatient", False)),
                    "high_tech": bool(features.get("high_tech_available", False)),
                    "outpatient": bool(features.get("outpatient", False)),
                    "maternity": bool(features.get("maternity", False)),

                    "reasons": [
                        f"V3.2 clinical score (weighted): {clinical_score:.2f}",
                        f"Dense relevance (normalized): {dense_norm:.3f}",
                        f"Cost-adjusted value index applied using excess €{int(features.get('excess', 0) or 0)}",
                        f"Breakdown: {breakdown}",
                    ],
                    "evidence": evidence,
                }
            )

        results.sort(key=lambda x: x["total_score"], reverse=True)
        return results[: self.cfg.top_n_plans]
