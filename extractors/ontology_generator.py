"""
extractors/ontology_generator.py
----------------------------------
Dynamically generates a domain ontology from extracted DPR facts.
No hardcoded ontology — generated per-document per-sector.

Workflow:
1. Sample representative facts from Neo4j for this document/sector
2. Prompt Ollama to propose an OWL-compatible ontology (classes + relationships)
3. Write ontology nodes and dependency edges to Neo4j
4. Return the ontology schema for reference

The generated ontology is used by the validation engine to:
- Create semantic node labels (beyond just "Fact")
- Define DEPENDS_ON relationships (which parameters need other parameters)
- Enable context-aware validation checks
"""

import uuid
from loguru import logger

from config.settings import NodeLabel, RelType
from utils.ollama_client import generate_json
from utils.neo4j_client import run_write, run_read


# ─── Ontology generation prompt ───────────────────────────────────────────────

_ONTO_SYSTEM = """You are an ontology engineer creating a domain ontology for infrastructure 
engineering DPR validation. Generate a practical, minimal ontology focused on enabling 
dependency checking and validation — not an academic exercise.
IMPORTANT: Respond with a single JSON object starting with { not an array."""


def _build_ontology_prompt(facts_sample: list[dict], sector: str) -> str:
    facts_text = "\n".join(
        f"- {f['subject']} | {f['attribute']} | {f['value']} {f['unit']}"
        for f in facts_sample[:50]  # sample up to 50 facts
    )

    return f"""Create a domain ontology for a {sector} DPR based on these extracted facts:

FACTS (subject | attribute | value+unit):
{facts_text}

Return a JSON object with this exact structure:
{{
    "classes": [
        {{
            "name": "ClassName",
            "description": "what this class represents",
            "example_subjects": ["pile foundation", "abutment"]
        }}
    ],
    "object_properties": [
        {{
            "name": "propertyName",
            "domain": "SourceClass",
            "range": "TargetClass",
            "description": "relationship meaning"
        }}
    ],
    "dependencies": [
        {{
            "parameter": "foundation type",
            "requires": "soil bearing capacity",
            "reason": "foundation selection depends on SBC",
            "validation_rule": "if shallow foundation then SBC >= 100 kN/m²"
        }}
    ],
    "sector_specific_entities": [
        {{
            "entity_type": "Bridge",
            "key_parameters": ["span", "carriageway width", "HFL", "scour depth"],
            "mandatory_checks": ["hydrology inputs present", "foundation type vs SBC consistent"]
        }}
    ]
}}

Rules:
- Classes should map to real engineering entities in {sector} projects
- Dependencies are the most important part — define which parameters require other parameters
- Keep it practical: 5-12 classes, 5-10 dependencies
- Use engineering terminology specific to {sector}"""


# ─── Write ontology to Neo4j ──────────────────────────────────────────────────

def _write_ontology_to_neo4j(ontology: dict, doc_id: str, sector: str):
    """
    Persist the generated ontology as OntologyClass nodes and DEPENDS_ON edges.
    """
    # Write class nodes
    for cls in ontology.get("classes", []):
        cls_id = str(uuid.uuid4())
        run_write(
            f"""
            MERGE (o:{NodeLabel.ONTOLOGY} {{
                name: $name,
                sector: $sector,
                doc_id: $doc_id
            }})
            SET o.ontology_id  = $cls_id,
                o.description  = $description,
                o.examples     = $examples
            """,
            {
                "name":        cls["name"],
                "sector":      sector,
                "doc_id":      doc_id,
                "cls_id":      cls_id,
                "description": cls.get("description", ""),
                "examples":    cls.get("example_subjects", []),
            }
        )

    # Write DEPENDS_ON relationships between parameter concepts
    for dep in ontology.get("dependencies", []):
        run_write(
            f"""
            MERGE (a:{NodeLabel.ONTOLOGY} {{name: $param, sector: $sector}})
              ON CREATE SET a.ontology_id = $aid, a.description = $param
            MERGE (b:{NodeLabel.ONTOLOGY} {{name: $requires, sector: $sector}})
              ON CREATE SET b.ontology_id = $bid, b.description = $requires
            MERGE (a)-[dep:{RelType.DEPENDS_ON}]->(b)
            SET dep.reason          = $reason,
                dep.validation_rule = $rule,
                dep.doc_id          = $doc_id
            """,
            {
                "param":    dep.get("parameter", ""),
                "requires": dep.get("requires", ""),
                "reason":   dep.get("reason", ""),
                "rule":     dep.get("validation_rule", ""),
                "doc_id":   doc_id,
                "sector":   sector,
                "aid":      str(uuid.uuid4()),
                "bid":      str(uuid.uuid4()),
            }
        )

    # Write sector entity types with mandatory checks
    for entity in ontology.get("sector_specific_entities", []):
        run_write(
            f"""
            MERGE (e:{NodeLabel.ONTOLOGY} {{
                name: $entity_type,
                sector: $sector
            }})
            SET e.ontology_id      = $eid,
                e.key_parameters   = $key_params,
                e.mandatory_checks = $checks,
                e.is_entity_type   = true
            """,
            {
                "entity_type": entity.get("entity_type", ""),
                "sector":      sector,
                "eid":         str(uuid.uuid4()),
                "key_params":  entity.get("key_parameters", []),
                "checks":      entity.get("mandatory_checks", []),
            }
        )

    logger.success(
        f"Ontology written: {len(ontology.get('classes', []))} classes, "
        f"{len(ontology.get('dependencies', []))} dependencies"
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_ontology(doc_id: str, sector: str) -> dict:
    """
    Sample facts from Neo4j for this doc, generate ontology via Ollama,
    persist to Neo4j, and return the ontology dict.
    """
    # Sample representative facts from this document
    facts = run_read(
        f"""
        MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
        WHERE f.fact_type <> 'table_row'
        RETURN f.subject AS subject, f.attribute AS attribute,
               f.value AS value, f.unit AS unit
        ORDER BY f.confidence DESC
        LIMIT 60
        """,
        {"doc_id": doc_id}
    )

    if not facts:
        logger.warning(f"No facts found for doc_id={doc_id}. Generating minimal ontology.")
        facts = []

    logger.info(f"Generating ontology from {len(facts)} facts for sector: {sector}")
    prompt = _build_ontology_prompt(facts, sector)
    ontology = generate_json(prompt, system=_ONTO_SYSTEM)

    # LLM may return a list (due to global array enforcement) or invalid structure
    if isinstance(ontology, list):
        # Try to find a dict inside the list
        for item in ontology:
            if isinstance(item, dict) and "classes" in item:
                ontology = item
                break
        else:
            ontology = {"classes": [], "dependencies": [], "sector_specific_entities": []}
    if not isinstance(ontology, dict):
        logger.error("Ontology generation failed — LLM returned invalid structure")
        ontology = {"classes": [], "dependencies": [], "sector_specific_entities": []}
    # Ensure required keys exist
    ontology.setdefault("classes", [])
    ontology.setdefault("dependencies", [])
    ontology.setdefault("sector_specific_entities", [])

    _write_ontology_to_neo4j(ontology, doc_id, sector)
    return ontology


def get_dependencies_for_sector(sector: str) -> list[dict]:
    """
    Retrieve all DEPENDS_ON relationships for a sector from Neo4j.
    Used by the consistency engine.
    Coerces all fields to strings to avoid type errors downstream.
    """
    rows = run_read(
        f"""
        MATCH (a:{NodeLabel.ONTOLOGY})-[dep:{RelType.DEPENDS_ON}]->(b:{NodeLabel.ONTOLOGY})
        WHERE a.sector = $sector
        RETURN a.name AS parameter, b.name AS requires,
               dep.reason AS reason, dep.validation_rule AS rule
        """,
        {"sector": sector}
    )
    # Coerce every field to string — Neo4j can return lists for multi-value props
    cleaned = []
    for row in rows:
        cleaned.append({
            "parameter": str(row.get("parameter") or "").strip(),
            "requires":  str(row.get("requires")  or "").strip(),
            "reason":    str(row.get("reason")     or "").strip(),
            "rule":      str(row.get("rule")       or "").strip(),
        })
    return cleaned