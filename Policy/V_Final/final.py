# Please run Cell 1 (line 3-265), Cell 2 (from line 266)

# Cell 1

import json
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
RAG_CHUNKS_PATH = "rag_chunks.jsonl"
SUPERSET_PATH = "MASTER_STRUCTURED_SUPERSET_2026-1.jsonl"
URL_METADATA_PATH = "metadata (1).json"
INDEX_PATH = "faiss_multi_provider_index.bin"

# ==========================================
# 1. HELPER: TOKENIZER FOR BM25
# ==========================================
def tokenize(text):
    return re.findall(r'\w+', text.lower())

# ==========================================
# 2. LOAD MODELS
# ==========================================
print("Loading FAISS embedding model...")
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# ==========================================
# 3. LOAD METADATA
# ==========================================
print("Mapping metadata for URLs and citations...")
url_lookup = {}
human_plan_names = {}

try:
    with open(URL_METADATA_PATH, "r") as f:
        meta_data = json.load(f)
        for item in meta_data:
            doc_id = item.get("doc_id", "").lower()
            if doc_id:
                url_lookup[doc_id] = item.get("source_url", "No URL Available")
                human_plan_names[doc_id] = item.get("plan_name", "Unknown Plan")
except FileNotFoundError:
    print(f"Warning: {URL_METADATA_PATH} not found.")

def get_plan_details(doc_id):
    doc_id_lower = doc_id.lower()
    return (human_plan_names.get(doc_id_lower, doc_id.title()), url_lookup.get(doc_id_lower, "No URL Available"))

# ==========================================
# 4. LOAD RAG CHUNKS (For Both FAISS & BM25)
# ==========================================
documents_for_faiss = []
documents_for_bm25 = []
chunk_metadata = []

print(f"Loading granular RAG chunks from {RAG_CHUNKS_PATH}...")
try:
    with open(RAG_CHUNKS_PATH, "r") as f:
        for line in f:
            obj = json.loads(line)

            doc_id = obj.get("doc_id", "")
            provider = obj.get("insurer", "Unknown Provider")
            plan_name = obj.get("plan_name", "")
            chunk_text = obj.get("text", "")

            # FAISS needs enriched context
            enriched_chunk = f"Provider: {provider}. Plan: {plan_name}. Document text: {chunk_text}"
            documents_for_faiss.append(enriched_chunk)

            # BM25 works best on raw text
            documents_for_bm25.append(chunk_text)

            chunk_metadata.append({
                "provider": provider,
                "doc_id": doc_id,
                "plan_name": plan_name,
                "chunk_text": chunk_text
            })
    print(f"Loaded {len(documents_for_faiss)} chunks.")
except FileNotFoundError:
    print(f"Error: {RAG_CHUNKS_PATH} not found.")

# ==========================================
# 5. LOAD STRUCTURED SUPERSET
# ==========================================
structured_data = {}
print("Loading structured data for numeric filtering...")
try:
    with open(SUPERSET_PATH, "r") as f:
        for line in f:
            obj = json.loads(line)
            plan_key = obj.get("plan_name", "").lower()
            structured_data.setdefault(plan_key, []).append(obj)
except FileNotFoundError:
    print(f"Warning: {SUPERSET_PATH} not found.")

# ==========================================
# 6. BUILD HYBRID INDICES (FAISS & BM25)
# ==========================================
print("Building FAISS Index (Dense/Semantic)...")
if documents_for_faiss:
    embeddings = model.encode(documents_for_faiss, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

print("Building BM25 Index (Sparse/Lexical)...")
tokenized_corpus = [tokenize(doc) for doc in documents_for_bm25]
bm25 = BM25Okapi(tokenized_corpus)
print("Hybrid Indexes built successfully!")

# ==========================================
# 7. ADVANCED SCORING & FILTERING LOGIC
# ==========================================
def compute_structured_score(plan_key, query):
    plan_key_lower = plan_key.lower()
    matching_key = next((key for key in structured_data.keys() if key in plan_key_lower or plan_key_lower in key), None)

    if not matching_key: return 0.5, True

    plan_entries = structured_data[matching_key]
    query_lower = query.lower()

    if "maternity" in query_lower and not any(e.get("maternity_flag") for e in plan_entries): return 0.0, False
    if "psychiatric" in query_lower and not any(e.get("psychiatric_flag") for e in plan_entries): return 0.0, False
    if "fertility" in query_lower and not any(e.get("fertility_flag") for e in plan_entries): return 0.0, False

    excess_values = [float(e["excess_amount"]) for e in plan_entries if e.get("excess_amount") is not None]
    avg_excess = np.mean(excess_values) if excess_values else 0
    excess_score = 1 / (1 + avg_excess / 250)

    return (0.7 * excess_score) + 0.15, True

# ==========================================
# 8. HYBRID SEARCH PIPELINE (RRF)
# ==========================================
def hybrid_search(query, k=60):
    query_lower = query.lower()
    mentioned_plan = None

    # 1. Detect explicit plan requests
    unique_plans = set(meta["plan_name"] for meta in chunk_metadata)
    for known_plan in unique_plans:
        clean_known_plan = known_plan.replace("(Table of Cover)", "").strip().lower()
        if clean_known_plan and clean_known_plan in query_lower:
            mentioned_plan = known_plan
            print(f"🎯 Detected explicit request for plan: {mentioned_plan}")
            break

    # 2. Query Rewriting (Strip plan name so search focuses purely on the intent)
    search_query = query
    if mentioned_plan:
        replace_target = mentioned_plan.replace("(Table of Cover)", "").strip()
        search_query = re.sub(replace_target, "", search_query, flags=re.IGNORECASE).strip()

    # --- A. FAISS (Semantic) Search ---
    prefixed_query = "Represent this sentence for searching relevant passages: " + search_query
    query_embedding = model.encode([prefixed_query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")
    faiss_scores, faiss_indices = index.search(query_embedding, k)

    # --- B. BM25 (Lexical) Search ---
    tokenized_query = tokenize(search_query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:k]

    # --- C. Reciprocal Rank Fusion (RRF) ---
    rrf_scores = {}
    RRF_K = 60 # Standard smoothing constant

    # Add FAISS Ranks
    for rank, idx in enumerate(faiss_indices[0]):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + (1.0 / (RRF_K + rank + 1))

    # Add BM25 Ranks
    for rank, idx in enumerate(bm25_indices):
        if bm25_scores[idx] > 0: # Only count if BM25 actually found a match
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1.0 / (RRF_K + rank + 1))

    # --- D. Group by Plan ---
    plan_chunks = {}
    provider_map = {}
    doc_id_map = {}

    for idx, rrf_score in rrf_scores.items():
        meta = chunk_metadata[idx]
        plan_name = meta["plan_name"]

        if plan_name not in plan_chunks:
            plan_chunks[plan_name] = []
            provider_map[plan_name] = meta["provider"]
            doc_id_map[plan_name] = meta["doc_id"]

        plan_chunks[plan_name].append({
            "text": meta["chunk_text"],
            "score": rrf_score
        })

    results = []

    for plan_name, chunks in plan_chunks.items():
        # EXPLICIT PLAN FILTER
        if mentioned_plan and plan_name != mentioned_plan:
            continue

        doc_id = doc_id_map[plan_name]

        # Sort chunks by their combined RRF Score
        chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
        best_rrf_score = chunks[0]["score"]

        # Compute Structured / Math Score
        structured_score, passes_filter = compute_structured_score(plan_name, query)
        if not passes_filter:
            continue

        # Normalize RRF Score for final weighting (max theoretical RRF is ~0.033)
        normalized_rrf = min(best_rrf_score / 0.033, 1.0)
        final_score = (normalized_rrf * 0.7) + (structured_score * 0.3)

        human_name, source_url = get_plan_details(doc_id)
        if human_name == doc_id.title():
            human_name = plan_name

        combined_evidence = "\n\n--- NEXT CHUNK ---\n\n".join([c["text"] for c in chunks[:2]])

        results.append({
            "plan_name": human_name,
            "provider": provider_map[plan_name],
            "source_url": source_url,
            "final_score": round(final_score, 4),
            "hybrid_rrf_score": round(best_rrf_score, 4),
            "structured_score": round(structured_score, 4),
            "evidence": combined_evidence
        })

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    return results[:5]

# ==========================================
# 9. EXECUTION / TEST
# ==========================================
if __name__ == "__main__":
    test_query = "What is the best plan for cardiac treatment?"

    print(f"\nSearching for: '{test_query}'\n")
    top_plans = hybrid_search(test_query)

    for i, r in enumerate(top_plans, 1):
        print(f"Rank {i}: {r['plan_name']} ({r['provider']})")
        print(f"URL: {r['source_url']}")
        print(f"Scores -> Final: {r['final_score']} | Hybrid RRF: {r['hybrid_rrf_score']} | Quantitative: {r['structured_score']}")
        print(f"\nEvidence Preview:\n{r['evidence']}\n")
        print("-" * 60)



# Cell 2


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 10. LLM CONFIGURATION (Qwen-2.5-3B)
# ==========================================
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("\nLoading Expert Broker LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
llm_model.eval()

# ==========================================
# 11. FULL DOCUMENT RETRIEVAL HELPER
# ==========================================
def get_full_document_text(plan_name):
    """
    Simulates 'reading the entire PDF' by finding every chunk
    in the JSONL file that belongs to this specific plan.
    """
    full_text_list = []
    # We look up the doc_id from the global chunk_metadata we loaded earlier
    # to find all chunks sharing that ID.
    target_doc_id = next((m['doc_id'] for m in chunk_metadata if m['plan_name'] == plan_name), None)

    if not target_doc_id:
        return "Plan details not found in source documentation."

    for meta in chunk_metadata:
        if meta['doc_id'] == target_doc_id:
            full_text_list.append(meta['chunk_text'])

    return "\n\n".join(full_text_list)

# ==========================================
# 12. EXPERT RECOMMENDATION GENERATOR
# ==========================================
def generate_recommendation(user_query, user_profile, search_results):
    if not search_results:
        return "I'm sorry, I couldn't find a plan that meets your specific requirements."

    # 1. Take the top-ranked plan
    top_plan = search_results[0]

    # 2. Retrieve the 'Entire PDF' text
    print(f"📖 Reading the entire documentation for: {top_plan['plan_name']}...")
    entire_policy_text = get_full_document_text(top_plan['plan_name'])

    # 3. Build the prompt
    messages = [
        {"role": "system", "content": "You are an expert, empathetic Irish Health Insurance broker. You provide recommendations based on strict actuarial evidence and full policy documentation."},
        {"role": "user", "content": f"""
USER QUESTION: "{user_query}"

PATIENT PROFILE:
- Conditions: {', '.join(user_profile.get('chronic_conditions', []))}
- Expected Annual Inpatient Visits: {user_profile.get('inpatient_visits_yr', 0)}
- Financial Tolerance: {user_profile.get('financial_tolerance', 'medium')}

TOP SEARCH RESULT:
- Plan: {top_plan['plan_name']} ({top_plan['provider']})
- Search Relevance Score: {top_plan['final_score']}
- Key Snippets found: {top_plan['evidence']}

FULL POLICY DOCUMENTATION (Extracted from PDF):
{entire_policy_text[:15000]} # Using first 15k chars to stay within context limits

TASK:
Write a professional recommendation for the user.
1. Acknowledge their condition and medical needs.
2. Explain specifically how the benefits found in the 'FULL POLICY DOCUMENTATION' cover their needs.
3. Explicitly mention the specific hospitals or cardiac programs if mentioned in the text.
4. Conclude why this is the safest medical and financial choice for them.
"""}
    ]

    # 4. Generate with Qwen
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)

    generated_ids = llm_model.generate(
        **model_inputs,
        max_new_tokens=1000,
        temperature=0.3,
        do_sample=True
    )

    # Clean output
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# ==========================================
# 13. UPDATED EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    test_query = "What is the best plan for cardiac treatment?"

    # Define the profile for the LLM to reason about
    current_user = {
        "chronic_conditions": ["cardiac"],
        "inpatient_visits_yr": 2,
        "financial_tolerance": "medium"
    }

    print(f"\nSearching for: '{test_query}'\n")

    # 1. Run your existing Hybrid Search
    top_plans = hybrid_search(test_query)

    # 2. Feed results into the LLM Expert Generator
    if top_plans:
        expert_report = generate_recommendation(test_query, current_user, top_plans)

        print("\n" + "="*60)
        print("🤖 EXPERT BROKER FINAL REPORT")
        print("="*60)
        print(expert_report)
        print("="*60)
        print(f"Source Document: {top_plans[0]['source_url']}")
    else:
        print("No results found.")
