"""
extractors/kg_builder.py
-------------------------
AutoSchemaKG-inspired KG construction pipeline adapted for:
  - Ollama as the LLM backend
  - DPR/engineering domain (sector-aware prompts)
  - Direct Neo4j write (no GraphML intermediate)
  - mxbai-embed-large for embeddings

Performance optimisations applied:
  1. MERGED PROMPT: entity-relation + event-entity extracted in one LLM call
     instead of two serial calls per chunk → ~2× fewer Ollama round-trips.

  2. BATCHED CONCEPT INDUCTION: all entities + relations on a page are sent
     in a single structured prompt returning JSON, instead of one call per
     entity/relation → reduces concept calls from ~60/page to 1-2/page.

  3. BATCHED NEO4J WRITES: all triples for a page are written with a single
     UNWIND Cypher query for nodes and one for edges, instead of 3 individual
     run_write() calls per triple → eliminates thousands of session round-trips.

  4. PAGE-LEVEL RESULT CACHE: extracted triples are cached to disk so partial
     runs can resume without re-querying Ollama.
"""

import re
import uuid
import json
import json_repair
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from config.settings import NodeLabel, RelType, SECTOR_KEYS, PROCESSED_DIR
from utils.ollama_client import generate, generate_json, get_model_for_task, TaskType
from utils.neo4j_client import run_write, run_read

# ─── Prompts ──────────────────────────────────────────────────────────────────

_TRIPLE_SYSTEM = (
    "You are a helpful assistant who always responds with a valid JSON object. "
    "No explanation, no preamble. Start directly with {."
)

# FIX 1: Single combined prompt — entity-relation triples AND event-entity pairs
# in one LLM call instead of two serial calls.
_COMBINED_EXTRACTION_PROMPT = """You are extracting structured knowledge from a {sector} engineering DPR passage.

Extract TWO things from the passage below:

1. TRIPLES: entities and the relations between them.
   - Entities: specific engineering elements (structures, parameters, materials, locations, standards)
   - Relations: concise verb phrases

2. EVENTS: engineering events/processes and the entities involved.
   - Events: engineering actions, measurements, design decisions, compliance statements

Passage:
{text}

Return ONLY a JSON object with exactly these two keys:
{{
  "triples": [
    {{"Head": "entity noun", "Relation": "verb phrase", "Tail": "entity noun"}},
    ...
  ],
  "events": [
    {{"Event": "engineering action or finding", "Entity": ["entity1", "entity2"]}},
    ...
  ]
}}

Extract all meaningful triples and events. If none found, return empty arrays."""

# FIX 2: Batched concept induction — all entities + relations in one call
_BATCH_CONCEPT_PROMPT = """You are labelling engineering entities and relations from a {sector} DPR with abstract concept types.

For each item below, provide 2-4 short concept labels (1-3 words each) describing its TYPE or CATEGORY.

Items:
{items_json}

Return ONLY a JSON object where each key is the item text and the value is an array of concept labels:
{{
  "item text": ["concept1", "concept2"],
  ...
}}

Guidelines:
- M30 concrete → ["material", "concrete grade"]
- pile foundation → ["foundation type", "structural element"]
- 7.5 m carriageway → ["geometric parameter", "design value"]
- IRC:37 → ["standard", "design code"]
- complies with → ["compliance", "regulatory"]
- has bearing capacity of → ["property specification", "geotechnical property"]

Provide labels for ALL items. No item should be missing from the output."""


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Triple:
    head: str
    relation: str
    tail: str
    triple_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    head_concepts: list[str] = field(default_factory=list)
    relation_concepts: list[str] = field(default_factory=list)
    tail_concepts: list[str] = field(default_factory=list)
    source_page: int = 0
    doc_id: str = ""
    sector: str = ""
    triple_type: str = "entity_relation"


@dataclass
class KGExtractionResult:
    triples: list[Triple] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    all_entities: list[str] = field(default_factory=list)
    all_relations: list[str] = field(default_factory=list)
    concept_map: dict[str, list[str]] = field(default_factory=dict)


# ─── Text chunker ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 6000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks on word boundaries."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    words = text.split()
    chunks, current_words, current_len = [], [], 0

    for word in words:
        wlen = len(word) + 1
        if current_len + wlen > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            overlap_words, overlap_len = [], 0
            for w in reversed(current_words):
                if overlap_len + len(w) + 1 > overlap:
                    break
                overlap_words.insert(0, w)
                overlap_len += len(w) + 1
            current_words = overlap_words
            current_len = overlap_len
        current_words.append(word)
        current_len += wlen

    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


# ─── FIX 1: Combined triple + event extraction (1 LLM call per chunk) ─────────

def _parse_combined_output(
    raw: dict | None,
    doc_id: str,
    sector: str,
    page_num: int,
) -> list[Triple]:
    """Parse the merged extraction JSON into Triple objects."""
    if not isinstance(raw, dict):
        return []

    triples = []

    # Entity-relation triples
    for item in raw.get("triples", []):
        if not isinstance(item, dict):
            continue
        head = str(item.get("Head", "")).strip()
        rel  = str(item.get("Relation", "")).strip()
        tail = str(item.get("Tail", "")).strip()
        if head and rel and tail:
            triples.append(Triple(
                head=head, relation=rel, tail=tail,
                source_page=page_num, doc_id=doc_id, sector=sector,
                triple_type="entity_relation",
            ))

    # Event-entity triples → head participates_in event
    for item in raw.get("events", []):
        if not isinstance(item, dict):
            continue
        event    = str(item.get("Event", "")).strip()
        entities = item.get("Entity", [])
        if event and entities:
            for ent in entities:
                ent = str(ent).strip()
                if ent:
                    triples.append(Triple(
                        head=ent, relation="participates in", tail=event,
                        source_page=page_num, doc_id=doc_id, sector=sector,
                        triple_type="event_entity",
                    ))

    return triples


def extract_triples_from_chunk(
    text: str,
    sector: str,
    doc_id: str,
    page_num: int,
) -> list[Triple]:
    """
    Extract entity-relation and event-entity triples from one text chunk.
    FIX 1: Single combined LLM call (was 2 serial calls).
    """
    prompt = _COMBINED_EXTRACTION_PROMPT.format(sector=sector, text=text[:4500])
    raw = generate_json(prompt, system=_TRIPLE_SYSTEM, model=get_model_for_task(TaskType.EXTRACTION))
    return _parse_combined_output(raw, doc_id, sector, page_num)


# ─── FIX 2: Batched concept induction (1-2 LLM calls per page, was ~60) ───────

def induce_concepts(
    entities: list[str],
    relations: list[str],
    sector: str,
    batch_size: int = 60,
) -> dict[str, list[str]]:
    """
    Generate abstract concept labels for all entities and relations in one batch call.
    FIX 2: Was one LLM call per entity/relation (~60 calls/page).
    Now: 1-2 calls per page using a structured batch prompt.

    batch_size controls how many items go into one prompt (keep ≤80 for context limits).
    """
    concept_map: dict[str, list[str]] = {}

    # Deduplicate and normalise
    unique_entities  = list(dict.fromkeys(e.lower() for e in entities  if e.strip()))[:80]
    unique_relations = list(dict.fromkeys(r.lower() for r in relations if r.strip()))[:40]
    all_items = unique_entities + unique_relations

    if not all_items:
        return concept_map

    # Process in batches (1 prompt per batch_size items)
    for i in range(0, len(all_items), batch_size):
        batch = all_items[i : i + batch_size]
        items_json = json.dumps(batch, ensure_ascii=False)
        prompt = _BATCH_CONCEPT_PROMPT.format(sector=sector, items_json=items_json)

        raw = generate_json(
            prompt,
            model=get_model_for_task(TaskType.CONCEPT),
        )

        if not isinstance(raw, dict):
            # Fallback: assign empty concepts so items aren't missing from map
            for item in batch:
                concept_map[item] = []
            continue

        for item in batch:
            concepts = raw.get(item, [])
            if isinstance(concepts, list):
                # Normalise: lowercase, max 3 words each, cap at 5
                clean = [c.strip().lower() for c in concepts if isinstance(c, str) and c.strip()]
                clean = [c for c in clean if len(c.split()) <= 3]
                concept_map[item] = clean[:5]
            else:
                concept_map[item] = []

    return concept_map


# ─── FIX 3: Batched Neo4j writes (UNWIND — 3 queries total, was 3N queries) ────

def write_triples_to_neo4j(
    triples: list[Triple],
    concept_map: dict[str, list[str]],
    doc_id: str,
    sector: str,
    dry_run: bool = False,
) -> int:
    """
    Write all triples to Neo4j in 3 bulk queries using UNWIND.
    FIX 3: Was 3 individual run_write() calls per triple (3N round-trips).
    Now: 3 UNWIND queries total regardless of triple count.
    """
    if dry_run or not triples:
        return len(triples)

    # ── Build node + edge parameter lists ─────────────────────────────────────

    head_nodes, tail_nodes, edges = [], [], []
    seen_node_ids: set[str] = set()

    for triple in triples:
        head_id   = f"{doc_id}::{triple.head.lower()}"
        tail_id   = f"{doc_id}::{triple.tail.lower()}"
        head_conc = concept_map.get(triple.head.lower(), [])
        tail_conc = concept_map.get(triple.tail.lower(), [])
        rel_conc  = concept_map.get(triple.relation.lower(), [])

        if head_id not in seen_node_ids:
            head_nodes.append({
                "entity_id":  head_id,
                "label":      triple.head,
                "concepts":   head_conc,
                "sector":     sector,
                "doc_id":     doc_id,
                "node_type":  "event" if triple.triple_type == "event_entity" else "entity",
            })
            seen_node_ids.add(head_id)

        if tail_id not in seen_node_ids:
            tail_nodes.append({
                "entity_id":  tail_id,
                "label":      triple.tail,
                "concepts":   tail_conc,
                "sector":     sector,
                "doc_id":     doc_id,
                "node_type":  "entity",
            })
            seen_node_ids.add(tail_id)

        edges.append({
            "head_id":          head_id,
            "tail_id":          tail_id,
            "relation":         triple.relation,
            "relation_concepts": rel_conc,
            "source_page":      triple.source_page,
            "triple_type":      triple.triple_type,
            "triple_id":        triple.triple_id,
            "doc_id":           doc_id,
            "sector":           sector,
        })

    all_nodes = head_nodes + tail_nodes  # already deduped

    # ── Query 1: Upsert all entity nodes in one UNWIND ────────────────────────
    run_write(
        f"""
        UNWIND $nodes AS n
        MERGE (e:{NodeLabel.ENTITY} {{entity_id: n.entity_id}})
        SET e.label     = n.label,
            e.concepts  = n.concepts,
            e.sector    = n.sector,
            e.doc_id    = n.doc_id,
            e.node_type = n.node_type
        WITH e, n
        MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: n.doc_id}})
        MERGE (d)-[:{RelType.HAS_ENTITY}]->(e)
        WITH e, n
        MATCH (s:{NodeLabel.SECTOR} {{name: n.sector}})
        MERGE (e)-[:{RelType.BELONGS_TO}]->(s)
        """,
        {"nodes": all_nodes},
    )

    # ── Query 2: Upsert all TRIPLE edges in one UNWIND ────────────────────────
    run_write(
        f"""
        UNWIND $edges AS edge
        MATCH (h:{NodeLabel.ENTITY} {{entity_id: edge.head_id}})
        MATCH (t:{NodeLabel.ENTITY} {{entity_id: edge.tail_id}})
        MERGE (h)-[r:{RelType.TRIPLE} {{relation: edge.relation, doc_id: edge.doc_id}}]->(t)
        SET r.relation_concepts = edge.relation_concepts,
            r.source_page       = edge.source_page,
            r.triple_type       = edge.triple_type,
            r.triple_id         = edge.triple_id,
            r.sector            = edge.sector
        """,
        {"edges": edges},
    )

    return len(triples)


def write_concept_schema_to_neo4j(
    concept_map: dict[str, list[str]],
    sector: str,
    doc_id: str,
):
    """
    Write unique concept labels as OntologyClass nodes in one UNWIND query.
    FIX 3 (continued): Was one run_write() per concept.
    """
    all_concepts = list({c for concepts in concept_map.values() for c in concepts if c.strip()})
    if not all_concepts:
        return

    run_write(
        f"""
        UNWIND $concepts AS cname
        MERGE (o:{NodeLabel.ONTOLOGY} {{name: cname, sector: $sector}})
        SET o.doc_id     = $doc_id,
            o.is_induced = true
        """,
        {"concepts": all_concepts, "sector": sector, "doc_id": doc_id},
    )


# ─── FIX 4: Page-level disk cache ─────────────────────────────────────────────

def _page_cache_path(doc_id: str, page_num: int) -> Path:
    cache_dir = PROCESSED_DIR / doc_id / "kg_page_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"page_{page_num:05d}.json"


def _load_page_cache(doc_id: str, page_num: int) -> list[dict] | None:
    """Return cached triples_data list for a page, or None if not cached."""
    p = _page_cache_path(doc_id, page_num)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_page_cache(doc_id: str, page_num: int, triples_data: list[dict]):
    """Persist triples_data for a page to disk."""
    p = _page_cache_path(doc_id, page_num)
    p.write_text(json.dumps(triples_data, ensure_ascii=False), encoding="utf-8")


def clear_page_cache(doc_id: str):
    """Delete all per-page cache files for a document (for --force-rebuild)."""
    cache_dir = PROCESSED_DIR / doc_id / "kg_page_cache"
    if cache_dir.exists():
        for f in cache_dir.glob("page_*.json"):
            f.unlink()
        logger.info(f"Cleared page cache for {doc_id}")


# ─── Public API ───────────────────────────────────────────────────────────────

def build_kg_from_page(
    text: str,
    doc_id: str,
    sector: str,
    page_num: int,
    write_to_db: bool = True,
    induce_schema: bool = True,
    use_cache: bool = True,
) -> KGExtractionResult:
    """
    Full KG construction for a single page:
      1. Check disk cache (FIX 4) — skip Ollama if already extracted
      2. Extract triples using merged prompt (FIX 1) — 1 call/chunk not 2
      3. Induce concepts in batch (FIX 2) — 1-2 calls/page not ~60
      4. Write to Neo4j using UNWIND (FIX 3) — 2 queries not 3N

    Returns KGExtractionResult with all extracted data.
    """
    result = KGExtractionResult()

    if not text or len(text.strip()) < 50:
        return result

    # ── FIX 4: Try cache first ─────────────────────────────────────────────
    if use_cache:
        cached = _load_page_cache(doc_id, page_num)
        if cached is not None:
            # Reconstruct result from cache (skip all Ollama calls)
            triples = []
            concept_map = {}
            for t in cached:
                triple = Triple(
                    head=t["head"], relation=t["relation"], tail=t["tail"],
                    triple_type=t.get("triple_type", "entity_relation"),
                    source_page=page_num, doc_id=doc_id, sector=sector,
                    head_concepts=t.get("head_concepts", []),
                    tail_concepts=t.get("tail_concepts", []),
                    relation_concepts=t.get("relation_concepts", []),
                )
                triples.append(triple)
                concept_map.update({
                    triple.head.lower():     triple.head_concepts,
                    triple.tail.lower():     triple.tail_concepts,
                    triple.relation.lower(): triple.relation_concepts,
                })
            result.triples      = triples
            result.all_entities = list(set([t.head for t in triples] + [t.tail for t in triples]))
            result.all_relations = list(set(t.relation for t in triples))
            result.concept_map  = concept_map

            if write_to_db and triples:
                write_triples_to_neo4j(triples, concept_map, doc_id, sector)
                if induce_schema:
                    write_concept_schema_to_neo4j(concept_map, sector, doc_id)
            return result

    # ── FIX 1: Chunk → 1 combined LLM call per chunk (not 2) ──────────────
    chunks = chunk_text(text, chunk_size=5000, overlap=150)
    all_triples: list[Triple] = []
    for chunk in chunks:
        triples = extract_triples_from_chunk(chunk, sector, doc_id, page_num)
        all_triples.extend(triples)

    result.triples       = all_triples
    result.all_entities  = list(set([t.head for t in all_triples] + [t.tail for t in all_triples]))
    result.all_relations = list(set(t.relation for t in all_triples))

    logger.debug(
        f"Page {page_num + 1}: {len(all_triples)} triples, "
        f"{len(result.all_entities)} entities"
    )

    if not all_triples:
        _save_page_cache(doc_id, page_num, [])
        return result

    # ── FIX 2: Batch concept induction (1-2 calls, not ~60) ───────────────
    concept_map: dict[str, list[str]] = {}
    if induce_schema:
        concept_map = induce_concepts(
            result.all_entities[:80],
            result.all_relations[:40],
            sector,
        )
        result.concept_map = concept_map

    # Attach concepts back to triple objects
    for triple in all_triples:
        triple.head_concepts     = concept_map.get(triple.head.lower(), [])
        triple.tail_concepts     = concept_map.get(triple.tail.lower(), [])
        triple.relation_concepts = concept_map.get(triple.relation.lower(), [])

    # ── FIX 4: Save to disk cache before Neo4j write ───────────────────────
    triples_data = [
        {
            "head":              t.head,
            "relation":         t.relation,
            "tail":             t.tail,
            "triple_type":      t.triple_type,
            "head_concepts":    t.head_concepts,
            "relation_concepts": t.relation_concepts,
            "tail_concepts":    t.tail_concepts,
            "source_page":      page_num,
            "doc_id":           doc_id,
            "sector":           sector,
        }
        for t in all_triples
    ]
    _save_page_cache(doc_id, page_num, triples_data)

    # ── FIX 3: UNWIND batch write (2 queries, not 3N) ─────────────────────
    if write_to_db:
        written = write_triples_to_neo4j(all_triples, concept_map, doc_id, sector)
        if induce_schema:
            write_concept_schema_to_neo4j(concept_map, sector, doc_id)
        logger.debug(f"Page {page_num + 1}: wrote {written} triples to Neo4j")

    return result


# ─── Neo4j stats ──────────────────────────────────────────────────────────────

def get_kg_stats(doc_id: str) -> dict:
    """Return triple/entity/concept counts for a document from Neo4j."""
    entity_count = run_read(
        f"MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $id}})-[:{RelType.HAS_ENTITY}]->(e:{NodeLabel.ENTITY}) RETURN count(e) AS cnt",
        {"id": doc_id}
    )
    triple_count = run_read(
        f"MATCH (e:{NodeLabel.ENTITY} {{doc_id: $id}})-[r:{RelType.TRIPLE}]->() RETURN count(r) AS cnt",
        {"id": doc_id}
    )
    concept_count = run_read(
        f"MATCH (o:{NodeLabel.ONTOLOGY} {{doc_id: $id}}) RETURN count(o) AS cnt",
        {"id": doc_id}
    )
    return {
        "entities": entity_count[0]["cnt"] if entity_count else 0,
        "triples":  triple_count[0]["cnt"] if triple_count else 0,
        "concepts": concept_count[0]["cnt"] if concept_count else 0,
    }