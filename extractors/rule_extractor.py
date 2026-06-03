"""
extractors/rule_extractor.py
-----------------------------
Extracts structured engineering rules from standards / rulebooks
(RDSO, IRC, BIS, AAI, etc.) and loads them into Neo4j as Rule nodes.

Each rule is:
    {
        "rule_id": <uuid>,
        "standard_name": "IRC:37",
        "clause":    "3.2.1",
        "rule_text": "The design CBR shall not be less than 3%",
        "attribute": "CBR",
        "operator":  ">=",
        "threshold": "3",
        "unit":      "%",
        "condition": "subgrade soil",
        "sector":    "highways",
        "severity":  "CRITICAL",
    }

Rules can be extracted from uploaded PDF/DOCX rulebooks or
loaded from structured YAML seed files (for common standards).
"""

import uuid
import re
from pathlib import Path
from loguru import logger

from config.settings import NodeLabel, RelType, SECTOR_KEYS, Severity
from utils.ollama_client import generate_json
from utils.neo4j_client import run_write
from extractors.document_loader import load_document


# ─── Rule extraction prompt ───────────────────────────────────────────────────

_RULE_SYSTEM = """You are extracting engineering compliance rules from infrastructure standards 
and codes. Each rule defines a minimum/maximum/required value or condition that must be satisfied 
in a Detailed Project Report (DPR). Focus on quantitative rules with measurable thresholds."""


def _build_rule_prompt(text: str, standard_name: str, sector: str) -> str:
    return f"""Extract engineering compliance rules from this excerpt of "{standard_name}" 
for the {sector} sector.

A rule is ANY statement that specifies what MUST, SHALL, SHOULD, or MUST NOT be done,
or specifies a required value, limit, frequency, or condition.
This includes:
  - Numeric thresholds ("shall not exceed 7.5 m")
  - Inspection frequencies ("shall be inspected every 6 months")
  - Material requirements ("shall be of grade M30 or higher")
  - Procedural requirements ("shall be tested before commissioning")
  - Prohibition rules ("shall not be used without prior approval")

TEXT:
\"\"\"
{text[:4000]}
\"\"\"

Return a JSON array. Each rule object MUST have:
- "clause": clause or section number as string (e.g. "3.2.1", "Para 4.2") — use "" if not found
- "rule_text": the full rule statement (verbatim or close paraphrase, ≤300 chars)
- "attribute": the engineering element or parameter this rule governs
- "operator": one of [">=", "<=", ">", "<", "==", "in_range", "must_be", "must_not_be", "requires", "every", "before", "after"]
- "threshold": the required value, limit, frequency, or condition as a string
- "unit": unit of measurement — empty string if none
- "condition": when this rule applies — empty string if always applicable
- "severity": "CRITICAL" for safety rules, "HIGH" for structural/operational, "MEDIUM" for maintenance, "LOW" for advisory

Rules:
- Include ALL shall/must/should statements, not just numeric ones
- Do not extract pure definitions or background information
- If truly no rules exist in this text, return []"""


# ─── Neo4j writer ─────────────────────────────────────────────────────────────

def _write_rules_to_neo4j(rules: list[dict], standard_name: str, sector: str):
    for rule in rules:
        rule_id = str(uuid.uuid4())
        run_write(
            f"""
            MERGE (r:{NodeLabel.RULE} {{
                standard_name: $standard_name,
                clause: $clause,
                attribute: $attribute
            }})
            ON CREATE SET r.rule_id    = $rule_id
            SET r.rule_text  = $rule_text,
                r.operator   = $operator,
                r.threshold  = $threshold,
                r.unit       = $unit,
                r.condition  = $condition,
                r.severity   = $severity,
                r.sector     = $sector
            WITH r
            MATCH (s:{NodeLabel.SECTOR} {{name: $sector_name}})
            MERGE (r)-[:{RelType.BELONGS_TO}]->(s)
            """,
            {
                "rule_id":       rule_id,
                "standard_name": standard_name,
                "clause":        rule.get("clause", ""),
                "rule_text":     rule.get("rule_text", "")[:500],
                "attribute":     rule.get("attribute", ""),
                "operator":      rule.get("operator", "must_be"),
                "threshold":     str(rule.get("threshold", "")),
                "unit":          rule.get("unit", ""),
                "condition":     rule.get("condition", ""),
                "severity":      rule.get("severity", Severity.HIGH),
                "sector":        sector,
                "sector_name":   sector,
            }
        )


# ─── Extract from a text block ────────────────────────────────────────────────

def extract_rules_from_text(
    text: str,
    standard_name: str,
    sector: str,
    write_to_db: bool = True,
) -> list[dict]:
    """Extract rules from a text block and optionally persist to Neo4j."""
    if not text or len(text.strip()) < 50:
        return []

    prompt = _build_rule_prompt(text, standard_name, sector)
    rules = generate_json(prompt, system=_RULE_SYSTEM)

    if not isinstance(rules, list):
        return []

    # Filter rules with empty attribute (threshold can be empty for procedural rules)
    rules = [
        r for r in rules
        if r.get("attribute") and str(r.get("attribute")).strip()
        and r.get("rule_text") and len(str(r.get("rule_text")).strip()) > 10
    ]

    logger.debug(f"Extracted {len(rules)} rules from {standard_name}")

    if write_to_db and rules:
        _write_rules_to_neo4j(rules, standard_name, sector)

    return rules


# ─── Extract from a full document (standards PDF/DOCX) ───────────────────────

def extract_rules_from_document(
    doc_path: Path,
    standard_name: str,
    sector: str,
    doc_id: str = None,
) -> list[dict]:
    """
    Load a standards document and extract all rules page-by-page.
    Returns the full list of extracted rules.
    """
    doc_path = Path(doc_path)
    _id = doc_id or doc_path.stem

    logger.info(f"Extracting rules from: {doc_path.name} → standard: {standard_name}")
    doc = load_document(doc_path, _id)

    all_rules = []
    for page in doc.pages:
        if not page.text or len(page.text.strip()) < 50:
            continue
        rules = extract_rules_from_text(
            text=page.text,
            standard_name=standard_name,
            sector=sector,
            write_to_db=True,
        )
        all_rules.extend(rules)
        if rules:
            logger.info(f"  Page {page.page_num + 1}: {len(rules)} rules extracted")

    logger.success(f"Total rules extracted from {standard_name}: {len(all_rules)}")
    return all_rules


# ─── Seed rules from structured YAML ─────────────────────────────────────────

def load_seed_rules_from_yaml(yaml_path: Path):
    """
    Load pre-structured rules from a YAML file and write to Neo4j.
    YAML format:
        - standard_name: "IRC:37"
          sector: "highways"
          rules:
            - clause: "3.2.1"
              rule_text: "..."
              attribute: "subgrade CBR"
              operator: ">="
              threshold: "3"
              unit: "%"
              condition: ""
              severity: "CRITICAL"
    """
    import yaml
    data = yaml.safe_load(yaml_path.read_text())
    total = 0
    for entry in data:
        standard = entry["standard_name"]
        sector = entry["sector"]
        rules = entry.get("rules", [])
        _write_rules_to_neo4j(rules, standard, sector)
        total += len(rules)
        logger.info(f"Loaded {len(rules)} seed rules from {standard}")
    logger.success(f"Total seed rules loaded: {total}")


# ─── Auto-detect standard name from document ─────────────────────────────────

def detect_standard_name(text_sample: str) -> str:
    """
    Use LLM to detect the standard name from the first few pages of a rulebook.
    Falls back to filename-based detection.
    """
    prompt = (
        "What is the name/code of the engineering standard or rulebook described in this text? "
        "Examples: IRC:37, RDSO/SPN/TR/0026, IS:456, AAI Aerodrome Standards. "
        "Return ONLY the standard name/code as a plain string, nothing else.\n\n"
        f"Text:\n\"\"\"\n{text_sample[:2000]}\n\"\"\""
    )
    from utils.ollama_client import generate
    result = generate(prompt).strip().strip('"').strip("'")
    return result if result else "Unknown Standard"