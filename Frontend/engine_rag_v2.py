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
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from pdf_highlighter import highlight_chunks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HIGHLIGHT_DIR = os.path.join(BASE_DIR, "highlighted_pdfs")
os.makedirs(HIGHLIGHT_DIR, exist_ok=True)

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


def _resolve_pdf_path(raw_path: str, registry_path: str) -> str | None:
    if not raw_path:
        return None

    if os.path.isabs(raw_path) and os.path.exists(raw_path):
        return raw_path

    candidates = [
        raw_path,
        os.path.join(BASE_DIR, raw_path),
        os.path.join(os.path.dirname(registry_path), raw_path),
        os.path.join(os.path.dirname(os.path.dirname(registry_path)), raw_path),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(registry_path))), raw_path),
    ]

    for c in candidates:
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c

    return None


# =========================================================
# Query preference parsing (V2 UPGRADED)
# =========================================================
def parse_query_prefs(q: str) -> Dict[str, Any]:
    """
    Parses explicit intent signals. These influence ranking and optionally enforce constraints.
    - want_inpatient vs avoid_inpatient (e.g., "no inpatient")
    - want_outpatient / outpatient_only
    - want_maternity
    - want_high_tech
    - avoid_high_tech
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

    want_high_tech = any(k in q for k in ["high-tech", "high tech", "hi-tech", "hitech", "hi tech"])
    avoid_high_tech = any(k in q for k in [
        "no high-tech",
        "no high tech",
        "without high-tech",
        "without high tech",
        "avoid high-tech",
        "avoid high tech",
        "don't want high-tech",
        "dont want high-tech",
        "don't want high tech",
        "dont want high tech",
        "not high-tech",
        "not high tech",
    ])

    return {
        "low_excess": any(k in q for k in ["low excess", "cheap", "minimum excess", "no excess", "lowest excess", "budget", "affordable"]),
        "want_inpatient": any(k in q for k in ["inpatient", "hospital cover", "private hospital", "ward", "room"]) and not avoid_inpatient,
        "avoid_inpatient": avoid_inpatient,
        "want_outpatient": ("outpatient" in q or "out-patient" in q or "out patient" in q),
        "outpatient_only": outpatient_only,
        "want_maternity": ("maternity" in q or "pregnan" in q or "antenatal" in q or "postnatal" in q),
        "want_high_tech": want_high_tech,
        "avoid_high_tech": avoid_high_tech,
    }


# =========================================================
# Feature extraction from plan text (robust)
# =========================================================
def extract_features(text: str) -> Dict[str, Any]:
    """
    Extracts key plan features from retrieved evidence text.

    Output keys (used by Streamlit):
      - excess (int)
      - psychiatric_days (int)
      - full_inpatient (bool)
      - high_tech_available (bool)
      - outpatient (bool)
      - maternity (bool)
      - cardiac_copay (int)
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

    # ---------------------------------------------------------
    # Psychiatric days (FIXED): only capture days near psych terms
    # Prevents false matches like "14 days waiting period"
    # ---------------------------------------------------------
    psych_matches = re.findall(
        r"(psychiatric|mental health)[^\.]{0,80}?(\d+)\s+days",
        t
    )
    psychiatric_days = max([_safe_int(m[1], 0) for m in psych_matches], default=0)

    # ---------------------------------------------------------
    # Excess (FIXED): avoid capturing €10, 10%, etc.
    # Allows €0 or typical 2-4 digit excess amounts.
    # ---------------------------------------------------------
    excess_match = re.search(r"€\s?(0|\d{2,4})\s*excess", t)
    if not excess_match:
        excess_match = re.search(r"excess\s*(?:of\s*)?€\s?(0|\d{2,4})", t)
    excess = _safe_int(excess_match.group(1), 0) if excess_match else 0

    # cardiac copay (optional)
    copay_match = re.search(r"€\s?(\d{1,5})[^\n]{0,80}cardiac", t)
    cardiac_copay = _safe_int(copay_match.group(1), 0) if copay_match else 0

    return {
        "full_inpatient": bool(inpatient),
        "high_tech_available": bool(high_tech),
        "outpatient": bool(outpatient),
        "maternity": bool(maternity),
        "psychiatric_days": int(psychiatric_days),
        "excess": int(excess),
        "cardiac_copay": int(cardiac_copay),
    }


# =========================================================
# Config
# =========================================================
@dataclass
class EngineConfig:
    metadata_path: str = os.path.join(BASE_DIR, "ConstructionJSON", "metadata_v2.json")
    faiss_path: str = os.path.join(BASE_DIR, "ConstructionJSON", "faiss_index_v2.bin")
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

        plan_scores: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)

        for s, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            doc_id = self.metadata[idx].get("doc_id")
            if not doc_id:
                continue
            plan_scores[doc_id] += float(s)
            counts[doc_id] += 1

        # average per-plan score
        for pid in list(plan_scores.keys()):
            c = max(counts.get(pid, 1), 1)
            plan_scores[pid] = float(plan_scores[pid] / c)

        return plan_scores

    # ----------------------------
    # Evidence chunks per plan
    # ----------------------------
    def gather_evidence(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        q_emb = self.model.encode([self._pref(query)], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype("float32")

        k = min(self.cfg.dense_top_k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, k)

        per_plan: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for s, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            m = self.metadata[idx]
            doc_id = m.get("doc_id")
            if not doc_id:
                continue

            txt = (m.get("chunk_text") or m.get("text") or "")
            if not txt:
                continue

            per_plan[doc_id].append(
                {
                    "chunk_score": float(s),
                    "text": txt,
                    "page_start": m.get("page_start"),
                    "page_end": m.get("page_end"),
                }
            )

        evidence: Dict[str, List[Dict[str, Any]]] = {}
        for doc_id, items in per_plan.items():
            items.sort(key=lambda x: x["chunk_score"], reverse=True)
            evidence[doc_id] = items[: self.cfg.evidence_per_plan]

        return evidence

    # ----------------------------
    # Plan-level feature extraction
    # ----------------------------
    def extract_plan_features(self, doc_id: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        blob = "\n".join([(e.get("text") or "") for e in (evidence or [])])
        feats = extract_features(blob)
        feats["doc_id"] = doc_id
        return feats

    # ----------------------------
    # Scoring (Risk-aware + intent-aware)
    # ----------------------------
    def compute_rule_score(
        self,
        profile: Dict[str, Any],
        query: str,
        feats: Dict[str, Any],
    ) -> Tuple[float, List[str], List[str]]:
        reasons: List[str] = []
        clinical: List[str] = []

        prefs = parse_query_prefs(query)

        # infer priorities from conditions
        conditions = [c for c in (profile.get("chronic_conditions") or []) if isinstance(c, str)]
        priorities: List[str] = []
        requires_high_tech = False
        requires_psych = False
        for c in conditions:
            rule = DISEASE_RULES.get(c)
            if not rule:
                continue
            pr = rule.get("priority")
            if pr:
                priorities.append(pr)
            if rule.get("requires_high_tech"):
                requires_high_tech = True
            if rule.get("requires_psych_days"):
                requires_psych = True

        # choose max severity priority
        priority_rank = {"critical": 3, "specialised": 2, "moderate": 1, "chronic": 0}
        selected_priority = max(priorities, key=lambda p: priority_rank.get(p, 0), default="chronic")
        weights = DISEASE_PRIORITY.get(selected_priority, DISEASE_PRIORITY["chronic"])

        score = 0.0

        # --- Core clinical weights ---
        if feats.get("high_tech_available"):
            score += float(weights.get("high_tech", 0))
            clinical.append("High-tech hospital access detected in evidence.")
        else:
            if requires_high_tech:
                score -= float(weights.get("high_tech", 0)) * 1.5
                clinical.append("High-tech is required for this condition but not detected in plan evidence.")

        if feats.get("full_inpatient"):
            score += float(weights.get("inpatient", 0))
            clinical.append("Inpatient hospital cover detected in evidence.")
        else:
            score -= float(weights.get("inpatient", 0)) * 0.5
            clinical.append("Limited/unclear inpatient cover in evidence.")

        if feats.get("outpatient"):
            score += float(weights.get("outpatient", 0))
            clinical.append("Outpatient benefits detected in evidence.")

        if feats.get("maternity"):
            score += float(weights.get("maternity", 0))
            clinical.append("Maternity benefits detected in evidence.")

        if requires_psych:
            days = int(feats.get("psychiatric_days", 0) or 0)
            if days >= 100:
                score += float(DISEASE_PRIORITY["moderate"]["psychiatric"]) * 0.75
                clinical.append(f"Psychiatric cover: {days} days.")
            elif days > 0:
                score += float(DISEASE_PRIORITY["moderate"]["psychiatric"]) * 0.35
                clinical.append(f"Some psychiatric cover: {days} days.")
            else:
                score -= float(DISEASE_PRIORITY["moderate"]["psychiatric"]) * 0.75
                clinical.append("Psychiatric cover required but not detected.")

        # --- Excess penalty (scaled by disease severity and intent) ---
        excess = float(feats.get("excess", 0) or 0)
        excess_mult = float(weights.get("excess_mult", 1.0))

        # stronger when explicitly asked
        if prefs.get("low_excess"):
            excess_mult *= 1.75
            reasons.append("User intent: low/cheap excess requested (stronger excess penalty).")

        # penalty (bounded)
        penalty = min(excess / 50.0, 10.0) * excess_mult
        score -= penalty
        reasons.append(f"Excess penalty applied: -{penalty:.2f} (excess=€{int(excess)}).")

        # =========================================================
        # Strong intent enforcement (negative constraints)
        # =========================================================

        # 1) Outpatient only / avoid inpatient (strong override)
        if prefs.get("outpatient_only") or prefs.get("avoid_inpatient"):
            reasons.append("User intent: outpatient-only / avoid inpatient detected.")
            if feats.get("full_inpatient"):
                score -= 25.0
                reasons.append("Penalty: inpatient cover present despite outpatient-only intent (-25).")
            else:
                score += 10.0
                reasons.append("Bonus: no inpatient emphasis aligns with outpatient-only intent (+10).")

            if feats.get("outpatient"):
                score += 12.0
                reasons.append("Bonus: outpatient benefits present (+12).")
            else:
                score -= 12.0
                reasons.append("Penalty: outpatient benefits not detected (-12).")

        # 2) Avoid high-tech
        if prefs.get("avoid_high_tech"):
            reasons.append("User intent: avoid high-tech hospitals detected.")
            if feats.get("high_tech_available"):
                score -= 20.0
                reasons.append("Penalty: high-tech mentioned despite avoid-high-tech intent (-20).")
            else:
                score += 8.0
                reasons.append("Bonus: no high-tech mention aligns with intent (+8).")

        # 3) Want high-tech (enforce a bit stronger)
        if prefs.get("want_high_tech"):
            reasons.append("User intent: high-tech requested.")
            if feats.get("high_tech_available"):
                score += 10.0
                reasons.append("Bonus: high-tech detected (+10).")
            else:
                score -= 15.0
                reasons.append("Penalty: high-tech requested but not detected (-15).")

        # 4) Want maternity (strengthen if explicitly requested)
        if prefs.get("want_maternity"):
            reasons.append("User intent: maternity requested.")
            if feats.get("maternity"):
                score += 10.0
                reasons.append("Bonus: maternity detected (+10).")
            else:
                score -= 10.0
                reasons.append("Penalty: maternity requested but not detected (-10).")

        # ✅ Clamp to avoid negative values in UI
        score = max(0.0, score)

        return float(score), reasons, clinical

    # ----------------------------
    # Recommend
    # ----------------------------
    def recommend(self, profile: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        dense_scores = self.compute_dense_scores(query)
        dense_norm = _normalize_01_dict(dense_scores)

        evidence_map = self.gather_evidence(query)

        scored: List[Dict[str, Any]] = []
        for doc_id in self.plan_ids:
            ev = evidence_map.get(doc_id, [])
            feats = self.extract_plan_features(doc_id, ev)

            rule_score, reasons, clinical = self.compute_rule_score(profile, query, feats)

            retrieval_score_norm = float(dense_norm.get(doc_id, 0.0))

            # retrieval used as tie-breaker (still relevant)
            total_score = rule_score + (retrieval_score_norm * 1.0)

            # ✅ Clamp to avoid negative values in UI
            total_score = max(0.0, total_score)

            # confidence: blend rule magnitude + retrieval
            conf = float(
                min(
                    1.0,
                    max(
                        0.0,
                        (retrieval_score_norm * 0.6)
                        + (min(max(rule_score / 40.0, 0.0), 1.0) * 0.4),
                    ),
                )
            )

            # display name / company
            display_name = ""
            company = ""
            reg = self.registry.get(doc_id)
            if isinstance(reg, dict):
                display_name = reg.get("plan_name") or reg.get("display_name") or ""
                company = reg.get("insurer") or reg.get("company") or ""
            if not display_name:
                display_name = _clean_name_from_doc_id(doc_id)
            if not company:
                company = _infer_company_from_name(display_name)

            highlighted_pdf = None
            pdf_path = None
            if isinstance(reg, dict):
                pdf_path = reg.get("file_path")

            resolved_pdf_path = _resolve_pdf_path(pdf_path, self.cfg.registry_path) if pdf_path else None

            if resolved_pdf_path and ev:
                output_pdf = os.path.join(
                    HIGHLIGHT_DIR,
                    f"{doc_id}_highlighted_{uuid.uuid4().hex}.pdf"
                )
                try:
                    highlighted_pdf = highlight_chunks(
                        pdf_path=resolved_pdf_path,
                        evidence=ev[: self.cfg.evidence_per_plan],
                        output_path=output_pdf,
                    )
                except Exception:
                    highlighted_pdf = None

            scored.append(
                {
                    "doc_id": doc_id,
                    "display_name": display_name,
                    "company": company,
                    "total_score": float(total_score),
                    "rule_score": float(rule_score),
                    "retrieval_score_norm": float(retrieval_score_norm),
                    "confidence": float(conf),
                    "excess": int(feats.get("excess", 0) or 0),
                    "psychiatric_days": int(feats.get("psychiatric_days", 0) or 0),
                    "full_inpatient": bool(feats.get("full_inpatient")),
                    "high_tech": bool(feats.get("high_tech_available")),
                    "outpatient": bool(feats.get("outpatient")),
                    "maternity": bool(feats.get("maternity")),
                    "reasons": reasons,
                    "clinical_summary": clinical,
                    "evidence": ev[: self.cfg.evidence_per_plan],
                    "highlighted_pdf": highlighted_pdf,
                }
            )

        scored.sort(key=lambda x: x["total_score"], reverse=True)
        return scored[: self.cfg.top_n_plans]
