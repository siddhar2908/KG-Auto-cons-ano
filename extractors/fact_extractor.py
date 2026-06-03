"""
extractors/fact_extractor.py
-----------------------------
Extracts structured engineering facts from DPR text using Ollama LLM.

Each "fact" is a structured record:
    {
        "fact_id": <uuid>,
        "fact_type": "parameter" | "measurement" | "material" | "cost" | "schedule" | "assumption",
        "subject": "pier foundation",
        "attribute": "bearing capacity",
        "value": "250",
        "unit": "kN/m²",
        "context": "surrounding sentence",
        "source_page": 14,
        "confidence": 0.9,
        "sector": "bridges",
        "doc_id": "...",
    }

Facts are chunked by page and extracted with a sector-aware prompt.
Results are written directly to Neo4j.
"""

import uuid
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from config.settings import CHUNK_SIZE, SECTOR_KEYS, NodeLabel, RelType
from utils.ollama_client import generate_json
from utils.neo4j_client import run_write

# ─── Sector-specific extraction hints ─────────────────────────────────────────
# These are injected into the prompt to guide the LLM on what to look for.
# Fully config-driven — no sector logic hardcoded in extraction code.

SECTOR_EXTRACTION_HINTS: dict[str, str] = {
    "rail": (
        "Focus on: track geometry (gauge, gradient, curvature), axle loads, "
        "speed limits, ballast depth, rail section, sleeper spacing, signal block lengths, "
        "station platforms, earthwork quantities, formation width."
    ),
    "bridges": (
        "Focus on: span lengths, number of spans, deck width, carriageway width, "
        "foundation type, soil bearing capacity (SBC), pile dimensions, "
        "design flood level (HFL/DFL), scour depth, live loads (IRC class), "
        "material grades (concrete, steel), seismic zone."
    ),
    "tunnels": (
        "Focus on: tunnel length, diameter/cross-section dimensions, overburden depth, "
        "rock mass rating (RMR/Q-value), support system (shotcrete thickness, bolt spacing), "
        "lining thickness, portal dimensions, ventilation requirements, groundwater inflow."
    ),
    "metro": (
        "Focus on: corridor length, number of stations, station depth/height, "
        "headway, design speed, rolling stock capacity, viaduct span, "
        "depressed/elevated/underground ratio, ridership projections, fare."
    ),
    "mobility": (
        "Focus on: freight volume, commodity type, route length, modal split, "
        "vehicle counts, logistics park area, warehouse capacity, connection to NH/rail."
    ),
    "highways": (
        "Focus on: carriageway width, number of lanes, pavement composition "
        "(DBM/BC/GSB thickness), subgrade CBR, design traffic (MSA), "
        "formation width, ROW, gradient, curve radius, cross-drainage structures."
    ),
    "ports": (
        "Focus on: berth length, draft depth, cargo capacity, quay wall type, "
        "fender system, dredging depth, equipment (cranes, reach stackers), "
        "hinterland connectivity, navigational channel dimensions."
    ),
    "airports": (
        "Focus on: runway length and width, PCN/ACN, pavement type, "
        "taxiway dimensions, apron area, terminal capacity (MPPA), "
        "approach category (CAT I/II/III), wind rose, RESA dimensions."
    ),
}

# ─── Extraction prompt ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a senior infrastructure engineer extracting structured facts 
from a Detailed Project Report (DPR). You extract ONLY facts explicitly stated in the text — 
never infer or assume values. Each fact must have a numeric or clearly measurable value."""

def _build_extraction_prompt(text: str, sector: str, page_num: int, doc_id: str) -> str:
    sector_key = SECTOR_KEYS.get(sector, "")
    hints = SECTOR_EXTRACTION_HINTS.get(sector_key, "Focus on engineering parameters, quantities, and costs.")

    return f"""Extract engineering facts from the following DPR text (page {page_num + 1}, sector: {sector}).

{hints}

TEXT:
\"\"\"
{text}
\"\"\"

Return a JSON array of fact objects. Each object MUST have these exact keys:
- "fact_type": one of ["parameter", "measurement", "material", "cost", "schedule", "assumption", "design_value"]
- "subject": the engineering element this fact describes (e.g. "pile foundation", "carriageway")
- "attribute": the specific property (e.g. "length", "bearing capacity", "thickness")
- "value": the numeric or categorical value as a string (e.g. "250", "M30", "Class AA")
- "unit": unit of measurement (e.g. "kN/m²", "mm", "km") — use empty string if not applicable
- "context": verbatim sentence or phrase from the text containing this fact (≤ 200 chars)
- "confidence": float 0.0–1.0 (how clearly is this stated in the text?)

Rules:
- Only include facts EXPLICITLY stated in the text
- Numeric values must be extractable
- If no clear facts exist, return []
- Do not repeat the same fact twice
- Ignore administrative or procedural text"""


# ─── Neo4j writer ─────────────────────────────────────────────────────────────

def _write_facts_to_neo4j(facts: list[dict], doc_id: str, sector: str, page_num: int):
    """Upsert extracted facts as Fact nodes connected to the Document node."""
    for fact in facts:
        fact_id = str(uuid.uuid4())
        run_write(
            f"""
            MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})
            MERGE (f:{NodeLabel.FACT} {{fact_id: $fact_id}})
            SET f.fact_type  = $fact_type,
                f.subject    = $subject,
                f.attribute  = $attribute,
                f.value      = $value,
                f.unit       = $unit,
                f.context    = $context,
                f.confidence = $confidence,
                f.source_page = $page_num,
                f.sector     = $sector,
                f.doc_id     = $doc_id
            MERGE (d)-[:{RelType.HAS_FACT}]->(f)
            WITH f
            MATCH (s:{NodeLabel.SECTOR} {{name: $sector_name}})
            MERGE (f)-[:{RelType.BELONGS_TO}]->(s)
            """,
            {
                "doc_id":    doc_id,
                "fact_id":   fact_id,
                "fact_type": fact.get("fact_type", "parameter"),
                "subject":   fact.get("subject", ""),
                "attribute": fact.get("attribute", ""),
                "value":     str(fact.get("value", "")),
                "unit":      fact.get("unit", ""),
                "context":   fact.get("context", "")[:400],
                "confidence": float(fact.get("confidence", 0.5)),
                "page_num":  page_num,
                "sector":    sector,
                "sector_name": sector,
            }
        )


# ─── Table facts ─────────────────────────────────────────────────────────────

def _write_table_facts_to_neo4j(table_rows: list[dict], doc_id: str, sector: str, page_num: int):
    """
    Convert extracted table rows into Fact nodes.
    Each row becomes a separate fact with the row data stored as JSON string.
    """
    import json
    for i, row in enumerate(table_rows):
        fact_id = str(uuid.uuid4())
        run_write(
            f"""
            MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})
            MERGE (f:{NodeLabel.FACT} {{fact_id: $fact_id}})
            SET f.fact_type   = 'table_row',
                f.subject     = 'table',
                f.attribute   = 'row_data',
                f.value       = $row_json,
                f.unit        = '',
                f.context     = 'Extracted from table',
                f.confidence  = 0.85,
                f.source_page = $page_num,
                f.row_index   = $row_index,
                f.sector      = $sector,
                f.doc_id      = $doc_id
            MERGE (d)-[:{RelType.HAS_FACT}]->(f)
            """,
            {
                "doc_id":    doc_id,
                "fact_id":   fact_id,
                "row_json":  json.dumps(row),
                "page_num":  page_num,
                "row_index": i,
                "sector":    sector,
            }
        )


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_facts_from_page(
    text: str,
    doc_id: str,
    sector: str,
    page_num: int,
    write_to_db: bool = True,
) -> list[dict]:
    """
    Extract engineering facts from a single page of text.
    Optionally writes results to Neo4j.
    Returns list of fact dicts.
    """
    if not text or len(text.strip()) < 30:
        return []

    # Truncate to chunk size (rough token approximation: 1 token ≈ 4 chars)
    max_chars = CHUNK_SIZE * 4
    text_chunk = text[:max_chars]

    prompt = _build_extraction_prompt(text_chunk, sector, page_num, doc_id)
    facts = generate_json(prompt, system=_SYSTEM_PROMPT)

    if not isinstance(facts, list):
        logger.debug(f"Page {page_num + 1}: no facts extracted (got {type(facts).__name__})")
        return []

    # Filter out facts with empty value or subject
    facts = [
        f for f in facts
        if f.get("subject") and f.get("value") and str(f.get("value")).strip()
    ]

    logger.debug(f"Page {page_num + 1}: extracted {len(facts)} facts")

    if write_to_db and facts:
        _write_facts_to_neo4j(facts, doc_id, sector, page_num)

    return facts


def write_table_facts(
    table_rows: list[dict],
    doc_id: str,
    sector: str,
    page_num: int,
):
    """Write table rows extracted by table_extractor as Fact nodes."""
    if table_rows:
        _write_table_facts_to_neo4j(table_rows, doc_id, sector, page_num)
        logger.debug(f"Page {page_num + 1}: wrote {len(table_rows)} table row facts")