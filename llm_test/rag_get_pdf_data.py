# --- Milvus-backed retrieval tool (matches your insert schema) ---

import os, json
from dotenv import load_dotenv
from typing import Dict, Any, List
from pymilvus import MilvusClient
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5")
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
TEXT_FIELD = os.getenv("MILVUS_TEXT_FIELD", "text")
VECTOR_FIELD = os.getenv("MILVUS_VECTOR_FIELD", "vector")
TOP_K = int(os.getenv("RAG_TOP_K", "10"))
MAX_CHARS = int(os.getenv("RAG_PASSAGE_CHARS", "2000"))
SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.0"))
SEARCH_QUERY_PREFIX = os.getenv("SEARCH_QUERY_PREFIX", "")

# ===== One-time Milvus connection & collection check (NO INSERT) =====
_client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
# raise early if collection missing/misnamed
_client.describe_collection(COLLECTION_NAME)
_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)

def _embed_query(q: str) -> List[float]:
    return _model.encode(f"{SEARCH_QUERY_PREFIX}{q}".strip()).tolist()

def _format_hits(hits: list) -> Dict[str, Any]:
    passages, lines = [], []
    for i, h in enumerate(hits, 1):
        ent = h.get("entity", {})
        txt = (ent.get(TEXT_FIELD, "") or "")[:MAX_CHARS].strip()
        fn, cid, etag = ent.get("filename", ""), ent.get("chunk_id"), ent.get("ETag", "")
        score = float(h.get("distance", 0.0))
        source_id = f"{fn}#{cid}" if fn and cid is not None else (etag or str(ent.get("id","")))
        passages.append({"n": i, "source_id": source_id, "filename": fn, "chunk_id": cid, "etag": etag, "score": score, "text": txt})
        lines.append(f"[{i}] {source_id}\n{txt}")
    return {"passages": passages, "joined_context": "\n\n".join(lines)}

def _search(query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    qv = _embed_query(query)
    res = _client.search(
        collection_name=COLLECTION_NAME,
        data=[qv],
        anns_field=VECTOR_FIELD,
        limit=top_k,
        output_fields=[TEXT_FIELD, "filename", "chunk_id", "ETag"],
        search_params={"metric_type": "COSINE", "params": {}},
    )
    hits = res[0] if res else []
    if SCORE_THRESHOLD > 0 and hits:
        if float(hits[0].get("distance", 0.0)) < SCORE_THRESHOLD:
            return {"passages": [], "joined_context": ""}
    return _format_hits(hits)

@tool("search_pdfs", return_direct=False)
def search_pdfs(query: str, k: int = TOP_K) -> str:
    """Milvus COSINE search over PDF chunks; returns JSON {passages, joined_context}."""
    return json.dumps(_search(query, top_k=k), ensure_ascii=False)

# # Optional: direct function for two-pass fallback (returns dict instead of JSON)
# def rag_search(query: str, k: int = TOP_K) -> Dict[str, Any]:
#     return _search(query, top_k=k)

# def _embed_query(q: str) -> List[float]:
#     """
#     Embed the query with the SAME model/config used for indexing.
#     If you normalized embeddings at index time, do the same here.
#     """
#     q_text = f"{SEARCH_QUERY_PREFIX}{q}".strip()
#     vec = model.encode(q_text)                 # <-- SAME model as convert_to_vectors()
#     # If you normalized to unit length for cosine at indexing, do it here too:
#     # vec = vec / (np.linalg.norm(vec) + 1e-12)
#     return vec.tolist() if hasattr(vec, "tolist") else list(vec)

# def _format_hits(hits: List[dict]) -> Dict[str, Any]:
#     """
#     Convert Milvus hits into a JSON payload:
#       - passages: rich metadata for UI/citation
#       - joined_context: compact text block the LLM can read
#     """
#     passages, lines = [], []
#     for i, hit in enumerate(hits, start=1):
#         # MilvusClient.search returns dicts like:
#         # {"id": ..., "distance": <cosine_sim>, "entity": {"text": "...", "filename": "...", "chunk_id": ..., "ETag": "..."}}
#         entity = hit.get("entity", {})
#         txt  = (entity.get(TEXT_FIELD, "") or "")[:MAX_CHARS].strip()
#         fn   = entity.get("filename", "")
#         cid  = entity.get("chunk_id")
#         etag = entity.get("ETag", "")
#         # For COSINE, Milvus returns similarity in 'distance' (higher ≈ better, up to ~1.0)
#         score = float(hit.get("distance", 0.0))

#         source_id = f"{fn}#{cid}" if fn != "" and cid is not None else (etag or str(entity.get("id", "")))
#         header = f"[{i}] {source_id}"
#         lines.append(f"{header}\n{txt}")

#         passages.append({
#             "n": i,
#             "source_id": source_id,
#             "filename": fn,
#             "chunk_id": cid,
#             "etag": etag,
#             "score": score,
#             "text": txt
#         })

#     return {"passages": passages, "joined_context": "\n\n".join(lines)}

# def _search(query: str, top_k: int = TOP_K) -> Dict[str, Any]:
#     """
#     Embed the query with the SAME encoder, run COSINE search, return formatted results.
#     """
#     q_vec = _embed_query(query)

#     # NOTE: anns_field must be your vector field; output_fields must include your text & meta fields
#     result = _client.search(
#         collection_name=COLLECTION_NAME,
#         data=[q_vec],                         # one query vector
#         anns_field=VECTOR_FIELD,
#         limit=top_k,
#         output_fields=[TEXT_FIELD, "filename", "chunk_id", "ETag"],
#         search_params={"metric_type": "COSINE", "params": {}},
#     )
#     # result is a list (per query); we sent one query
#     hits = result[0] if result else []

#     # Optional gating by score
#     if SCORE_THRESHOLD > 0 and hits:
#         best = float(hits[0].get("distance", 0.0))
#         if best < SCORE_THRESHOLD:
#             return {"passages": [], "joined_context": ""}

#     return _format_hits(hits)

# # ============== LangChain Tool (usable for tool-calling OR manual RAG) ==============
# @tool("search_pdfs", return_direct=False)
# def search_pdfs(query: str, k: int = TOP_K) -> str:
#     """
#     Search Milvus (COSINE) using the SAME Nomic embedder as indexing.
#     Returns JSON with:
#       - 'passages': [{n, source_id, filename, chunk_id, etag, score, text}, ...]
#       - 'joined_context': string with numbered snippets for citations
#     """
#     result = _search(query, top_k=k)
#     return json.dumps(result, ensure_ascii=False)
