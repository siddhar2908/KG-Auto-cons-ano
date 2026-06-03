"""
extractors/kg_builder.py
-------------------------
AutoSchemaKG-inspired KG construction pipeline adapted for:
  - Ollama as the LLM backend (replaces OpenAI LLMGenerator)
  - DPR/engineering domain (sector-aware prompts)
  - Direct Neo4j write (no GraphML intermediate)
  - mxbai-embed-large for embeddings (already available)

Pipeline per page/chunk:
  1. Triple extraction  → (Head, Relation, Tail) triples
  2. Event extraction   → (Event, [Entities]) pairs
  3. Concept induction  → abstract concept labels per entity/relation
  4. Neo4j write        → nodes + edges with concept labels

The triples ARE the KG — facts and rules both become triples.
Concept labels are the ontology schema layer on top.

This replaces both fact_extractor.py and ontology_generator.py
for the main DPR extraction flow.
"""

import re
import uuid
import json
import json_repair
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from loguru import logger

from config.settings import NodeLabel, RelType, SECTOR_KEYS
from utils.ollama_client import generate, generate_json, get_model_for_task, TaskType
from utils.neo4j_client import run_write, run_read

# ─── Prompts (adapted from AutoSchemaKG triple_extraction_prompt.py) ─────────
# We keep the same structure but add sector context for engineering domain

_TRIPLE_SYSTEM = (
    "You are a helpful assistant who always responds with a valid JSON array. "
    "No explanation, no preamble. Start directly with [."
)

_ENTITY_RELATION_PROMPT = """Given a passage from a {sector} engineering DPR, extract all important entities and the relations between them.
Entities should be specific engineering elements (structures, parameters, materials, locations, standards).
Relations should be concise verbs or phrases capturing the connection.

Output ONLY a JSON array:
[
    {{"Head": "entity noun", "Relation": "verb phrase", "Tail": "entity noun"}},
    ...
]

Passage:
{text}"""

_EVENT_ENTITY_PROMPT = """From this {sector} engineering DPR passage, extract engineering events/processes and the entities involved.
An event is a specific engineering action, measurement, design decision, or compliance statement.

Output ONLY a JSON array:
[
    {{"Event": "engineering action or finding as a sentence", "Entity": ["entity1", "entity2"]}},
    ...
]

Passage:
{text}"""

_CONCEPT_ENTITY_PROMPT = """Given this engineering ENTITY from a {sector} DPR, provide 2-4 abstract concept labels (1-2 words each) representing its TYPE or CATEGORY.

Format: concept1, concept2, concept3
No explanation. Just comma-separated labels.

Examples:
ENTITY: M30 concrete
Answer: material, concrete grade, structural material

ENTITY: pile foundation
Answer: foundation type, substructure, structural element

ENTITY: 7.5 m carriageway width
Answer: geometric parameter, road dimension, design value

ENTITY: IRC:37
Answer: standard, design code, regulatory document

ENTITY: {entity}
Answer:"""

_CONCEPT_RELATION_PROMPT = """Given this engineering RELATION, provide 2-4 abstract concept labels (1-2 words each) for its TYPE.

Format: concept1, concept2, concept3
No explanation.

Examples:
RELATION: has bearing capacity of
Answer: property specification, structural parameter, geotechnical property

RELATION: complies with
Answer: compliance, regulatory, standard reference

RELATION: requires
Answer: dependency, prerequisite, design requirement

RELATION: {relation}
Answer:"""


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
    triple_type: str = "entity_relation"  # "entity_relation" | "event_entity"


@dataclass
class KGExtractionResult:
    triples: list[Triple] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    all_entities: list[str] = field(default_factory=list)
    all_relations: list[str] = field(default_factory=list)
    concept_map: dict[str, list[str]] = field(default_factory=dict)


# ─── Text chunker (from AutoSchemaKG TextChunker) ────────────────────────────

def chunk_text(text: str, chunk_size: int = 6000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks on word boundaries."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    words = text.split()
    chunks = []
    current_words = []
    current_len = 0

    for word in words:
        wlen = len(word) + 1
        if current_len + wlen > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            # Overlap: keep last N chars worth of words
            overlap_words = []
            overlap_len = 0
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


# ─── Triple extraction ────────────────────────────────────────────────────────

def _parse_triples(raw: list | None, triple_type: str, doc_id: str, sector: str, page: int) -> list[Triple]:
    """Parse raw LLM output into Triple objects."""
    if not isinstance(raw, list):
        return []

    triples = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        if triple_type == "entity_relation":
            head = str(item.get("Head", "")).strip()
            rel  = str(item.get("Relation", "")).strip()
            tail = str(item.get("Tail", "")).strip()
            if head and rel and tail:
                triples.append(Triple(
                    head=head, relation=rel, tail=tail,
                    source_page=page, doc_id=doc_id, sector=sector,
                    triple_type="entity_relation"
                ))

        elif triple_type == "event_entity":
            event = str(item.get("Event", "")).strip()
            entities = item.get("Entity", [])
            if event and entities:
                # Create one triple per entity: entity -[participates_in]-> event
                for ent in entities:
                    ent = str(ent).strip()
                    if ent:
                        triples.append(Triple(
                            head=ent, relation="participates in", tail=event,
                            source_page=page, doc_id=doc_id, sector=sector,
                            triple_type="event_entity"
                        ))

    return triples


def extract_triples_from_chunk(
    text: str,
    sector: str,
    doc_id: str,
    page_num: int,
) -> list[Triple]:
    """Extract entity-relation and event-entity triples from one text chunk."""
    all_triples = []

    # Entity-relation triples
    prompt1 = _ENTITY_RELATION_PROMPT.format(sector=sector, text=text[:4000])
    raw1 = generate_json(prompt1, system=_TRIPLE_SYSTEM, model=get_model_for_task(TaskType.EXTRACTION))
    all_triples.extend(_parse_triples(raw1, "entity_relation", doc_id, sector, page_num))

    # Event-entity triples
    prompt2 = _EVENT_ENTITY_PROMPT.format(sector=sector, text=text[:4000])
    raw2 = generate_json(prompt2, system=_TRIPLE_SYSTEM, model=get_model_for_task(TaskType.EXTRACTION))
    all_triples.extend(_parse_triples(raw2, "event_entity", doc_id, sector, page_num))

    return all_triples


# ─── Concept induction ────────────────────────────────────────────────────────

def _parse_concepts(raw_text: str) -> list[str]:
    """Parse comma-separated concept labels from LLM output."""
    if not raw_text:
        return []
    concepts = [c.strip().lower() for c in raw_text.split(",") if c.strip()]
    # Filter: max 3 words per concept, no empty
    concepts = [c for c in concepts if c and len(c.split()) <= 3]
    return concepts[:5]  # cap at 5 concepts per entity


def induce_concepts(
    entities: list[str],
    relations: list[str],
    sector: str,
    batch_size: int = 20,
) -> dict[str, list[str]]:
    """
    For each unique entity and relation, generate abstract concept labels.
    Returns concept_map: {entity_or_relation: [concept1, concept2, ...]}
    Batches to minimize LLM calls.
    """
    concept_map: dict[str, list[str]] = {}

    # Deduplicate
    unique_entities  = list(set(e.lower() for e in entities if e.strip()))[:100]
    unique_relations = list(set(r.lower() for r in relations if r.strip()))[:50]

    # Process entities in batches
    for i in range(0, len(unique_entities), batch_size):
        batch = unique_entities[i:i + batch_size]
        for entity in batch:
            prompt = _CONCEPT_ENTITY_PROMPT.format(sector=sector, entity=entity)
            raw = generate(prompt, temperature=0.1, model=get_model_for_task(TaskType.CONCEPT))
            concept_map[entity] = _parse_concepts(raw)

    # Process relations in batches
    for i in range(0, len(unique_relations), batch_size):
        batch = unique_relations[i:i + batch_size]
        for relation in batch:
            prompt = _CONCEPT_RELATION_PROMPT.format(relation=relation)
            raw = generate(prompt, temperature=0.1, model=get_model_for_task(TaskType.CONCEPT))
            concept_map[relation] = _parse_concepts(raw)

    return concept_map


# ─── Neo4j writer ─────────────────────────────────────────────────────────────

def _neo4j_label_from_concepts(concepts: list[str]) -> str:
    """
    Pick the most specific concept label to use as Neo4j additional label.
    Returns a clean label string safe for Cypher (alphanumeric + underscore).
    """
    if not concepts:
        return ""
    label = concepts[0]
    # Sanitize: title-case, remove non-alphanum
    label = re.sub(r"[^a-zA-Z0-9\s]", "", label).title().replace(" ", "_")
    return label if label else ""


def write_triples_to_neo4j(
    triples: list[Triple],
    concept_map: dict[str, list[str]],
    doc_id: str,
    sector: str,
    dry_run: bool = False,
) -> int:
    """
    Write triples to Neo4j as nodes + edges.
    Node structure:
        (:Entity {id, label, concepts, sector, doc_id})
    Edge structure:
        -[:TRIPLE {relation, relation_concepts, source_page, doc_id}]->

    Returns count of triples written.
    """
    if dry_run:
        return len(triples)

    written = 0
    for triple in triples:
        head_id   = f"{doc_id}::{triple.head.lower()}"
        tail_id   = f"{doc_id}::{triple.tail.lower()}"
        head_conc = concept_map.get(triple.head.lower(), [])
        tail_conc = concept_map.get(triple.tail.lower(), [])
        rel_conc  = concept_map.get(triple.relation.lower(), [])

        # Merge head node
        run_write(
            f"""
            MERGE (h:Entity {{entity_id: $hid}})
            SET h.label    = $head,
                h.concepts = $hconc,
                h.sector   = $sector,
                h.doc_id   = $doc_id,
                h.node_type = $ntype
            WITH h
            MATCH (d:Document {{doc_id: $doc_id}})
            MERGE (d)-[:HAS_ENTITY]->(h)
            WITH h
            MATCH (s:Sector {{name: $sector_name}})
            MERGE (h)-[:BELONGS_TO]->(s)
            """,
            {
                "hid":         head_id,
                "head":        triple.head,
                "hconc":       head_conc,
                "sector":      sector,
                "doc_id":      doc_id,
                "ntype":       "event" if triple.triple_type == "event_entity" else "entity",
                "sector_name": sector,
            }
        )

        # Merge tail node
        run_write(
            f"""
            MERGE (t:Entity {{entity_id: $tid}})
            SET t.label    = $tail,
                t.concepts = $tconc,
                t.sector   = $sector,
                t.doc_id   = $doc_id,
                t.node_type = $ntype
            WITH t
            MATCH (d:Document {{doc_id: $doc_id}})
            MERGE (d)-[:HAS_ENTITY]->(t)
            """,
            {
                "tid":    tail_id,
                "tail":   triple.tail,
                "tconc":  tail_conc,
                "sector": sector,
                "doc_id": doc_id,
                "ntype":  "entity",
            }
        )

        # Merge edge (triple)
        run_write(
            f"""
            MATCH (h:Entity {{entity_id: $hid}})
            MATCH (t:Entity {{entity_id: $tid}})
            MERGE (h)-[r:TRIPLE {{
                relation: $relation,
                doc_id:   $doc_id
            }}]->(t)
            SET r.relation_concepts = $rconc,
                r.source_page       = $page,
                r.triple_type       = $ttype,
                r.triple_id         = $tid_r,
                r.sector            = $sector
            """,
            {
                "hid":      head_id,
                "tid":      tail_id,
                "relation": triple.relation,
                "rconc":    rel_conc,
                "page":     triple.source_page,
                "ttype":    triple.triple_type,
                "tid_r":    triple.triple_id,
                "doc_id":   doc_id,
                "sector":   sector,
            }
        )
        written += 1

    return written


# ─── Also write concept nodes as OntologyClass ───────────────────────────────

def write_concept_schema_to_neo4j(
    concept_map: dict[str, list[str]],
    sector: str,
    doc_id: str,
):
    """
    Write unique concept labels as OntologyClass nodes.
    This creates the schema layer — the induced ontology.
    """
    all_concepts = set()
    for concepts in concept_map.values():
        all_concepts.update(concepts)

    for concept in all_concepts:
        if not concept.strip():
            continue
        run_write(
            f"""
            MERGE (o:OntologyClass {{name: $name, sector: $sector}})
            SET o.ontology_id = $oid,
                o.doc_id      = $doc_id,
                o.is_induced  = true
            """,
            {
                "name":   concept,
                "sector": sector,
                "oid":    str(uuid.uuid4()),
                "doc_id": doc_id,
            }
        )


# ─── Public API ───────────────────────────────────────────────────────────────

def build_kg_from_page(
    text: str,
    doc_id: str,
    sector: str,
    page_num: int,
    write_to_db: bool = True,
    induce_schema: bool = True,
) -> KGExtractionResult:
    """
    Full KG construction for a single page:
    1. Extract triples (entity-relation + event-entity)
    2. Induce concepts (schema)
    3. Write to Neo4j

    Returns KGExtractionResult with all extracted data.
    """
    result = KGExtractionResult()

    if not text or len(text.strip()) < 50:
        return result

    # Step 1: Chunk if needed and extract triples
    chunks = chunk_text(text, chunk_size=5000, overlap=150)
    all_triples = []
    for chunk in chunks:
        triples = extract_triples_from_chunk(chunk, sector, doc_id, page_num)
        all_triples.extend(triples)

    result.triples = all_triples
    result.all_entities  = list(set(
        [t.head for t in all_triples] + [t.tail for t in all_triples]
    ))
    result.all_relations = list(set(t.relation for t in all_triples))

    logger.debug(
        f"Page {page_num + 1}: {len(all_triples)} triples, "
        f"{len(result.all_entities)} entities, "
        f"{len(result.all_relations)} relations"
    )

    if not all_triples:
        return result

    # Step 2: Concept induction (schema layer)
    concept_map = {}
    if induce_schema:
        concept_map = induce_concepts(
            result.all_entities[:40],   # limit per page to keep runtime reasonable
            result.all_relations[:20],
            sector,
        )
        result.concept_map = concept_map

    # Attach concepts back to triples
    for triple in all_triples:
        triple.head_concepts     = concept_map.get(triple.head.lower(), [])
        triple.tail_concepts     = concept_map.get(triple.tail.lower(), [])
        triple.relation_concepts = concept_map.get(triple.relation.lower(), [])

    # Step 3: Write to Neo4j
    if write_to_db:
        written = write_triples_to_neo4j(all_triples, concept_map, doc_id, sector)
        if induce_schema:
            write_concept_schema_to_neo4j(concept_map, sector, doc_id)
        logger.debug(f"Page {page_num + 1}: wrote {written} triples to Neo4j")

    return result


def get_kg_stats(doc_id: str) -> dict:
    """Return triple/entity/concept counts for a document from Neo4j."""
    entity_count = run_read(
        "MATCH (d:Document {doc_id: $id})-[:HAS_ENTITY]->(e:Entity) RETURN count(e) AS cnt",
        {"id": doc_id}
    )
    triple_count = run_read(
        "MATCH (e:Entity {doc_id: $id})-[r:TRIPLE]->() RETURN count(r) AS cnt",
        {"id": doc_id}
    )
    concept_count = run_read(
        "MATCH (o:OntologyClass {doc_id: $id}) RETURN count(o) AS cnt",
        {"id": doc_id}
    )
    return {
        "entities":  entity_count[0]["cnt"] if entity_count else 0,
        "triples":   triple_count[0]["cnt"] if triple_count else 0,
        "concepts":  concept_count[0]["cnt"] if concept_count else 0,
    }