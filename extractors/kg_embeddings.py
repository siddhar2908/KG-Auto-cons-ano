"""
extractors/kg_embeddings.py
-----------------------------
Builds FAISS indexes over KG nodes and edges for semantic retrieval.
Adapted from AutoSchemaKG create_graph_index.py.

Uses mxbai-embed-large via Ollama (already available on the system).
Indexes are saved to data/processed/<doc_id>/faiss/ for reuse.

Provides:
  - build_kg_index(doc_id) → builds and saves node + edge indexes
  - search_nodes(query, doc_id, top_k) → semantic node search
  - search_edges(query, doc_id, top_k) → semantic edge/triple search
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Optional

import ollama
import faiss
from loguru import logger
from tqdm import tqdm

from config.settings import PROCESSED_DIR, OLLAMA_EMBED_MODEL
from utils.neo4j_client import run_read


# ─── Ollama embedding wrapper ─────────────────────────────────────────────────

def embed_texts(texts: list[str], model: str = OLLAMA_EMBED_MODEL, batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of texts using Ollama mxbai-embed-large.
    Returns numpy array of shape (N, dim).
    """
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", leave=False):
        batch = texts[i:i + batch_size]
        try:
            response = ollama.embed(model=model, input=batch)
            embeddings = response["embeddings"]
            all_embeddings.extend(embeddings)
        except Exception as e:
            logger.warning(f"Embedding batch {i} failed: {e}. Using zeros.")
            dim = 1024  # mxbai-embed-large dimension
            all_embeddings.extend([np.zeros(dim).tolist()] * len(batch))

    arr = np.array(all_embeddings, dtype=np.float32)
    # L2-normalize for cosine similarity via inner product
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    arr = arr / norms
    return arr


# ─── FAISS index builder ──────────────────────────────────────────────────────

def _build_hnsw_index(embeddings: np.ndarray) -> faiss.Index:
    """Build HNSW flat index (fast, good for <1M vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    # Add in batches
    for i in range(0, embeddings.shape[0], 512):
        index.add(embeddings[i:i + 512])
    return index


# ─── Load KG data from Neo4j ──────────────────────────────────────────────────

def _load_nodes_from_neo4j(doc_id: str) -> tuple[list[str], list[str]]:
    """Returns (node_ids, node_labels) for all Entity nodes in a document."""
    rows = run_read(
        """
        MATCH (d:Document {doc_id: $id})-[:HAS_ENTITY]->(e:Entity)
        RETURN e.entity_id AS eid, e.label AS label
        """,
        {"id": doc_id}
    )
    node_ids    = [r["eid"]   for r in rows]
    node_labels = [r["label"] for r in rows]
    return node_ids, node_labels


def _load_edges_from_neo4j(doc_id: str) -> tuple[list[tuple], list[str]]:
    """Returns (edge_tuples, edge_strings) for all TRIPLE edges in a document."""
    rows = run_read(
        """
        MATCH (h:Entity {doc_id: $id})-[r:TRIPLE]->(t:Entity {doc_id: $id})
        RETURN h.label AS head, r.relation AS rel, t.label AS tail,
               h.entity_id AS hid, t.entity_id AS tid
        """,
        {"id": doc_id}
    )
    edge_tuples  = [(r["hid"], r["tid"]) for r in rows]
    edge_strings = [f"{r['head']} {r['rel']} {r['tail']}" for r in rows]
    return edge_tuples, edge_strings


# ─── Save / load indexes ──────────────────────────────────────────────────────

def _index_dir(doc_id: str) -> Path:
    d = PROCESSED_DIR / doc_id / "faiss"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_index(doc_id: str, index: faiss.Index, items: list, name: str):
    d = _index_dir(doc_id)
    faiss.write_index(index, str(d / f"{name}.index"))
    with open(d / f"{name}_items.pkl", "wb") as f:
        pickle.dump(items, f)


def _load_index(doc_id: str, name: str) -> tuple[Optional[faiss.Index], Optional[list]]:
    d = _index_dir(doc_id)
    idx_path  = d / f"{name}.index"
    item_path = d / f"{name}_items.pkl"
    if not idx_path.exists() or not item_path.exists():
        return None, None
    index = faiss.read_index(str(idx_path))
    with open(item_path, "rb") as f:
        items = pickle.load(f)
    return index, items


# ─── Public API ───────────────────────────────────────────────────────────────

def build_kg_index(doc_id: str, force_rebuild: bool = False) -> dict:
    """
    Build FAISS indexes for nodes and edges of a document's KG.
    Saves to data/processed/<doc_id>/faiss/.
    Returns {"nodes": count, "edges": count}.
    """
    # Check if already built
    d = _index_dir(doc_id)
    if not force_rebuild and (d / "nodes.index").exists() and (d / "edges.index").exists():
        logger.info(f"FAISS indexes already exist for {doc_id}. Use force_rebuild=True to rebuild.")
        _, node_items = _load_index(doc_id, "nodes")
        _, edge_items = _load_index(doc_id, "edges")
        return {"nodes": len(node_items or []), "edges": len(edge_items or [])}

    logger.info(f"Building FAISS indexes for doc_id={doc_id}...")

    # Load from Neo4j
    node_ids, node_labels = _load_nodes_from_neo4j(doc_id)
    edge_tuples, edge_strings = _load_edges_from_neo4j(doc_id)

    if not node_labels:
        logger.warning("No entities found in Neo4j. Run kg_builder first.")
        return {"nodes": 0, "edges": 0}

    logger.info(f"Embedding {len(node_labels)} nodes, {len(edge_strings)} edges...")

    # Embed nodes
    node_embeddings = embed_texts(node_labels)
    node_index = _build_hnsw_index(node_embeddings)
    _save_index(doc_id, node_index, list(zip(node_ids, node_labels)), "nodes")

    # Embed edges (only if edges exist)
    if edge_strings:
        edge_embeddings = embed_texts(edge_strings)
        edge_index = _build_hnsw_index(edge_embeddings)
        _save_index(doc_id, edge_index, list(zip(edge_tuples, edge_strings)), "edges")
    else:
        logger.warning("No edges found to index.")

    logger.success(
        f"FAISS indexes built: {len(node_labels)} nodes, {len(edge_strings)} edges"
    )
    return {"nodes": len(node_labels), "edges": len(edge_strings)}


def search_nodes(
    query: str,
    doc_id: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Semantic search over KG entity nodes.
    Returns list of {entity_id, label, score}.
    """
    index, items = _load_index(doc_id, "nodes")
    if index is None:
        logger.warning(f"No node index for {doc_id}. Run build_kg_index first.")
        return []

    q_emb = embed_texts([query])
    scores, indices = index.search(q_emb, min(top_k, len(items)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(items):
            continue
        entity_id, label = items[idx]
        results.append({"entity_id": entity_id, "label": label, "score": float(score)})

    return results


def search_edges(
    query: str,
    doc_id: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Semantic search over KG triples/edges.
    Returns list of {head_id, tail_id, triple_string, score}.
    """
    index, items = _load_index(doc_id, "edges")
    if index is None:
        logger.warning(f"No edge index for {doc_id}. Run build_kg_index first.")
        return []

    q_emb = embed_texts([query])
    scores, indices = index.search(q_emb, min(top_k, len(items)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(items):
            continue
        (head_id, tail_id), triple_string = items[idx]
        results.append({
            "head_id":       head_id,
            "tail_id":       tail_id,
            "triple_string": triple_string,
            "score":         float(score),
        })

    return results