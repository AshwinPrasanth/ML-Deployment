import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Condition rules (same spirit as your notebook) ----
DISEASE_RULES: Dict[str, Dict[str, Any]] = {
    "heart_disease": {
        "keywords": ["cardiac", "angioplasty", "stent", "cardiology", "heart"],
        "weight": 7,
        "requires_high_tech": True,
    },
    "diabetes": {
        "keywords": ["diabetes", "insulin", "endocrinology", "glucose"],
        "weight": 5,
        "requires_outpatient": True,
    },
    "cancer": {
        "keywords": ["oncology", "chemotherapy", "radiotherapy", "cancer", "tumour"],
        "weight": 9,
        "requires_high_tech": True,
    },
    "psychiatric_disorder": {
        "keywords": ["psychiatric", "mental health", "psychology", "psychiatry"],
        "weight": 6,
        "requires_psych": True,
    },
    "neurological_disorder": {
        "keywords": ["neurology", "mri", "ct scan", "brain", "neuro"],
        "weight": 7,
        "requires_high_tech": True,
    },
    "orthopaedic_condition": {
        "keywords": ["orthopaedic", "orthopedic", "joint replacement", "hip", "knee"],
        "weight": 6,
        "requires_hospital": True,
    },
    "pregnancy": {
        "keywords": ["maternity", "obstetric", "pregnancy", "antenatal"],
        "weight": 8,
        "requires_maternity": True,
    },
}


# ---- Helpers ----
def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9€]+", " ", text)
    return text.split()


def _normalize_01(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype("float32")
    mn, mx = float(scores.min()), float(scores.max())
    if mx - mn == 0:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _parse_excess(text: str) -> int:
    t = text.lower()
    # Patterns like: "€300 excess", "excess €300", "€ 300 excess"
    m = re.search(r"€\s?(\d{1,5})\s*excess", t)
    if m:
        return _safe_int(m.group(1), 0)
    m = re.search(r"excess\s*€\s?(\d{1,5})", t)
    if m:
        return _safe_int(m.group(1), 0)
    return 0


def _parse_psych_days(text: str) -> int:
    t = text.lower()
    # Very generic; if your chunks mention "X days" near psychiatric, it’ll be caught sometimes.
    # You can tighten this if you have a consistent phrasing.
    m = re.search(r"(\d+)\s+days", t)
    return _safe_int(m.group(1), 0) if m else 0


def _flag_contains(text: str, include: List[str], exclude: List[str] | None = None) -> bool:
    t = text.lower()
    if exclude:
        for ex in exclude:
            if ex in t:
                return False
    return any(inc in t for inc in include)


def _plan_feature_scan(merged_text: str) -> Dict[str, Any]:
    t = merged_text.lower()
    return {
        "excess": _parse_excess(t),
        "psychiatric_days": _parse_psych_days(t),
        "full_inpatient": _flag_contains(
            t,
            include=["inpatient consultant fees", "in-patient consultant fees", "consultant fees covered"],
            exclude=["not covered"],
        ),
        "high_tech": _flag_contains(t, include=["high-tech hospital", "high tech hospital"], exclude=["not covered"]),
        "outpatient": _flag_contains(t, include=["out-patient", "outpatient", "out patient"], exclude=["not covered"]),
        "maternity": _flag_contains(t, include=["maternity"], exclude=["not covered"]),
    }


def _score_plan(plan: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    if plan.get("full_inpatient"):
        score += 2.5
        reasons.append("Inpatient consultant fees appear covered (+2.5).")
    else:
        reasons.append("Inpatient cover unclear/limited (+0).")

    excess = int(plan.get("excess", 0) or 0)
    # €0 -> +2, €500 -> +1, €1000 -> +0
    add_excess = max(0.0, 2.0 - (excess / 500.0))
    score += add_excess
    reasons.append(f"Excess €{excess} influences affordability (+{add_excess:.2f}).")

    for cond in profile.get("chronic_conditions", []):
        rules = DISEASE_RULES.get(cond)
        if not rules:
            continue

        w = float(rules.get("weight", 0))
        merged = plan.get("_merged_text", "")

        matched_kw = any(kw in merged for kw in rules.get("keywords", []))
        if matched_kw:
            score += w
            reasons.append(f"Matches keywords for {cond} (+{w}).")
        else:
            partial = w * 0.2
            score += partial
            reasons.append(f"No strong keyword hit for {cond} (+{partial:.1f}).")

        if rules.get("requires_high_tech") and plan.get("high_tech"):
            score += 2.0
            reasons.append("High-tech hospital access aligns (+2.0).")
        if rules.get("requires_outpatient") and plan.get("outpatient"):
            score += 1.5
            reasons.append("Outpatient support aligns (+1.5).")
        if rules.get("requires_maternity") and plan.get("maternity"):
            score += 2.0
            reasons.append("Maternity support aligns (+2.0).")

    if "psychiatric_disorder" in profile.get("chronic_conditions", []):
        days = int(plan.get("psychiatric_days", 0) or 0)
        add = min(3.0, days / 20.0)  # 60 days -> +3 cap
        score += add
        reasons.append(f"Psychiatric days {days} contribute (+{add:.2f}).")

    return score, reasons


@dataclass
class EngineConfig:
    chunks_path: str = os.path.join(
        BASE_DIR, "ConstructionJSON", "data", "rag_chunks.jsonl"
    )
    registry_path: str = os.path.join(
        BASE_DIR, "ConstructionJSON", "data", "doc_registry.json"
    )
    faiss_path: str = os.path.join(
        BASE_DIR, "ConstructionJSON", "faiss_index.bin"
    )

    embed_model: str = "all-MiniLM-L6-v2"

    top_k_chunks: int = 25
    alpha_dense: float = 0.65
    beta_bm25: float = 0.35
    top_n_plans: int = 5
    evidence_per_plan: int = 4


class RAGEngine:
    """
    Chunk-level retrieval (FAISS + BM25) -> aggregate to plan(doc_id) -> score plans -> return explainable results.
    Uses YOUR files:
      - ConstructionJSON/data/rag_chunks_engineered_v2.jsonl
      - ConstructionJSON/data/doc_registry.json
      - ConstructionJSON/faiss_index.bin (auto-built if missing)
    """

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg

        # Load chunks
        self.chunks: List[Dict[str, Any]] = []
        with open(cfg.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("doc_type") != "table_of_cover":
                    continue
                # normalize expected fields
                obj.setdefault("doc_id", obj.get("doc_id", "unknown"))
                obj.setdefault("text", obj.get("text", ""))
                self.chunks.append(obj)

        if not self.chunks:
            raise RuntimeError(f"No chunks loaded from {cfg.chunks_path}")

        # Load registry (optional but very helpful for UI)
        self.registry: Dict[str, Any] = {}
        if os.path.exists(cfg.registry_path):
            with open(cfg.registry_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            # ✅ Support both dict and list registry formats
            if isinstance(loaded, dict):
                self.registry = loaded
            elif isinstance(loaded, list):
                # Convert list[{"doc_id": ...}] -> dict keyed by doc_id
                tmp = {}
                for item in loaded:
                    if isinstance(item, dict) and "doc_id" in item:
                        tmp[item["doc_id"]] = item
                self.registry = tmp
            else:
                self.registry = {}


        # Build BM25 over chunk text
        self.chunk_texts = [c.get("text", "") for c in self.chunks]
        tokenized = [_tokenize(t) for t in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

        # Dense model
        self.model = SentenceTransformer(cfg.embed_model)

        # Load or build FAISS index aligned with chunk order
        self.index = self._load_or_build_faiss()

        # Precompute plan merged text + plan features
        self.plan_docs = self._build_plans_from_chunks()

    def _load_or_build_faiss(self) -> faiss.Index:
        # If index exists, load and verify it matches current chunk list
        if os.path.exists(self.cfg.faiss_path):
            index = faiss.read_index(self.cfg.faiss_path)

            # ✅ critical sanity check: index size must match number of chunks
            if index.ntotal == len(self.chunk_texts):
                return index

            # If mismatch, rebuild (otherwise retrieval will be misaligned)
            try:
                os.remove(self.cfg.faiss_path)
            except OSError:
                pass

        # Build and persist (one-time)
        embeddings = self.model.encode(self.chunk_texts, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, self.cfg.faiss_path)
        return index


    def _build_plans_from_chunks(self) -> Dict[str, Dict[str, Any]]:
        by_doc: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(self.chunks):
            by_doc[c.get("doc_id", "unknown")].append(i)

        plans: Dict[str, Dict[str, Any]] = {}
        for doc_id, idxs in by_doc.items():
            merged = " ".join(self.chunks[i].get("text", "") for i in idxs).lower()
            features = _plan_feature_scan(merged)

            reg = self.registry.get(doc_id, {})
            plans[doc_id] = {
                "doc_id": doc_id,
                "display_name": reg.get("plan_name") or reg.get("title") or doc_id,
                "company": reg.get("company") or reg.get("provider") or "",
                "source_pdf": reg.get("pdf") or reg.get("source_pdf") or "",
                "_chunk_indices": idxs,
                "_merged_text": merged,
                **features,
            }
        return plans

    def _hybrid_retrieve_chunks(self, query: str, k: int) -> List[Tuple[int, float]]:
        # BM25 scores
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_scores = _normalize_01(np.array(bm25_scores))

        # Dense scores (FAISS)
        q = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        dense_scores, dense_idx = self.index.search(q, k=min(k, len(self.chunks)))
        dense_idx = dense_idx[0]
        dense_scores = dense_scores[0]

        # Create dense score array for all chunks (sparse -> dense array)
        dense_full = np.zeros(len(self.chunks), dtype="float32")
        for i, s in zip(dense_idx, dense_scores):
            if i >= 0:
                dense_full[int(i)] = float(s)
        dense_full = _normalize_01(dense_full)

        hybrid = self.cfg.alpha_dense * dense_full + self.cfg.beta_bm25 * bm25_scores

        top_idx = np.argsort(hybrid)[::-1][:k]
        return [(int(i), float(hybrid[int(i)])) for i in top_idx]

    def recommend(self, profile: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        # 1) retrieve top chunks
        retrieved = self._hybrid_retrieve_chunks(query, k=self.cfg.top_k_chunks)

        # 2) aggregate chunk scores to doc_id
        doc_scores: Dict[str, float] = defaultdict(float)
        doc_chunks: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

        for idx, score in retrieved:
            doc_id = self.chunks[idx].get("doc_id", "unknown")
            doc_scores[doc_id] += score
            doc_chunks[doc_id].append((idx, score))

        # 3) score plans (rule-based) + include retrieval sum
        scored_plans: List[Dict[str, Any]] = []
        # Normalize retrieval scores across docs so it doesn't dominate
        retrieval_vals = np.array(list(doc_scores.values()), dtype="float32")
        r_min, r_max = float(retrieval_vals.min()), float(retrieval_vals.max())
        def norm_r(x: float) -> float:
            return 0.0 if r_max - r_min == 0 else (x - r_min) / (r_max - r_min)
        for doc_id, retrieval_sum in doc_scores.items():
            plan = self.plan_docs.get(doc_id)
            if not plan:
                continue

            rule_score, reasons = _score_plan(plan, profile)
            retrieval_norm = norm_r(retrieval_sum)

            # Combine (give rule score more power)
            total = (0.75 * rule_score) + (0.25 * retrieval_norm * 10.0)            

            # evidence chunks for this plan: sort by chunk hybrid score
            evid = sorted(doc_chunks[doc_id], key=lambda x: x[1], reverse=True)[: self.cfg.evidence_per_plan]
            evidence_items = []
            for ci, cs in evid:
                c = self.chunks[ci]
                evidence_items.append(
                    {
                        "chunk_score": float(cs),
                        "text": c.get("text", ""),
                        "page": c.get("page", c.get("page_number", None)),
                        "source_pdf": c.get("source_pdf", c.get("pdf", plan.get("source_pdf", ""))),
                        "section": c.get("section", c.get("heading", "")),
                    }
                )

            scored_plans.append(
                {
                    "doc_id": doc_id,
                    "display_name": plan.get("display_name", doc_id),
                    "company": plan.get("company", ""),
                    "total_score": float(total),
                    "rule_score": float(rule_score),
                    "retrieval_score": float(retrieval_sum),
                    "retrieval_score_norm": float(retrieval_norm),
                    "excess": int(plan.get("excess", 0) or 0),
                    "psychiatric_days": int(plan.get("psychiatric_days", 0) or 0),
                    "full_inpatient": bool(plan.get("full_inpatient", False)),
                    "high_tech": bool(plan.get("high_tech", False)),
                    "outpatient": bool(plan.get("outpatient", False)),
                    "maternity": bool(plan.get("maternity", False)),
                    "reasons": reasons,
                    "evidence": evidence_items,
                }
            )

        scored_plans.sort(key=lambda x: x["total_score"], reverse=True)
        return scored_plans[: self.cfg.top_n_plans]
