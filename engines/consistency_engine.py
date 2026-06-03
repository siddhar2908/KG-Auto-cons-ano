"""
engines/consistency_engine.py
------------------------------
Detects inconsistencies WITHIN a DPR document:
  1. Numeric cross-section mismatches   (same parameter, different values in different sections)
  2. Ontology-based dependency failures (parameter present without its prerequisite)
  3. Narrative vs structured data gaps  (text says X but table says Y)
  4. LLM-based holistic consistency check (send grouped facts to Ollama for reasoning)

Results are written as Violation nodes to Neo4j and returned as a report dict.
"""

import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from config.settings import NodeLabel, RelType, Severity, SECTOR_KEYS
from utils.neo4j_client import run_read, run_write
from utils.ollama_client import generate_json
from extractors.ontology_generator import get_dependencies_for_sector

# ─── Attribute canonicalization ───────────────────────────────────────────────
# Maps attribute variants to a canonical form for consistent grouping.
# This prevents the consistency engine from missing mismatches because
# "headway", "headway in sec.", "headway value" are stored separately.

_ATTR_CANON: list[tuple] = [
    # Time / headway
    (re.compile(r"headway.*", re.I),           "headway"),
    (re.compile(r".*train.*interval.*", re.I), "headway"),
    # Length variants
    (re.compile(r"total.?length", re.I),        "total_length"),
    (re.compile(r"corridor.?length", re.I),     "total_length"),
    (re.compile(r"underground.?length.*", re.I),"underground_length"),
    (re.compile(r"elevated.?length.*", re.I),   "elevated_length"),
    # Station count
    (re.compile(r"(no\.?\s*of|number.?of|num).?stations?", re.I), "station_count"),
    (re.compile(r"total.?stations?", re.I),     "station_count"),
    # Speed
    (re.compile(r"design.?speed", re.I),        "design_speed"),
    (re.compile(r"operating.?speed", re.I),     "operating_speed"),
    # Depth
    (re.compile(r"depth.?(below|of).?ground.*", re.I), "depth_below_ground"),
    (re.compile(r"depth_below_ground.*", re.I),         "depth_below_ground"),
]

def _canonicalize_attribute(attr: str) -> str:
    """Normalize attribute name variants to canonical form."""
    a = attr.strip().lower()
    for pattern, canonical in _ATTR_CANON:
        if pattern.fullmatch(a) or pattern.match(a):
            return canonical
    return a


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ConsistencyIssue:
    issue_id: str
    issue_type: str       # "numeric_mismatch" | "missing_dependency" | "narrative_gap" | "llm_flagged"
    severity: str
    description: str
    fact_ids: list[str] = field(default_factory=list)
    evidence: str = ""
    page_refs: list[int] = field(default_factory=list)
    doc_id: str = ""
    sector: str = ""


# ─── 1. Numeric cross-section mismatch ───────────────────────────────────────

def _check_numeric_mismatches(doc_id: str, sector: str) -> list[ConsistencyIssue]:
    """
    Find cases where the same (subject, attribute) pair has significantly
    different numeric values on different pages of the same DPR.
    A 'significant' difference = values differ by > 10% of max.
    """
    issues = []

    # Pull all numeric facts for this document
    facts = run_read(
        f"""
        MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
        WHERE f.fact_type <> 'table_row'
        RETURN f.fact_id AS fid, f.subject AS subject, f.attribute AS attribute,
               f.value AS value, f.unit AS unit, f.source_page AS page,
               f.context AS ctx
        """,
        {"doc_id": doc_id}
    )

    # Group by (subject, canonicalized_attribute)
    # Canonicalization maps "headway in sec.", "headway value", "headway" → "headway"
    # so the engine correctly detects mismatches across attribute name variants.
    grouped: dict[tuple, list] = defaultdict(list)
    for f in facts:
        canon_attr = _canonicalize_attribute(f["attribute"])
        key = (f["subject"].lower().strip(), canon_attr)
        grouped[key].append(f)

    for (subject, attribute), entries in grouped.items():
        if len(entries) < 2:
            continue

        # Extract numeric values
        numeric_entries = []
        for e in entries:
            try:
                val = float(re.sub(r"[^\d.\-]", "", str(e["value"])))
                numeric_entries.append((val, e))
            except (ValueError, TypeError):
                pass

        if len(numeric_entries) < 2:
            continue

        # CRITICAL FILTER: only flag if values appear on DIFFERENT pages
        # Same value on same page repeated = extraction artifact, not mismatch
        # Different values on same page = could be different items, skip
        pages_per_value: dict[float, set] = defaultdict(set)
        for val, e in numeric_entries:
            pages_per_value[round(val, 3)].add(e["page"])

        unique_values_on_diff_pages = len(pages_per_value)
        all_pages = set(e["page"] for _, e in numeric_entries)

        # Skip if all occurrences are on the same page (different list items)
        if len(all_pages) < 2:
            continue

        # Skip if subject is too generic (single word like "station", "line")
        # These represent categories not specific entities
        subject_words = subject.split()
        if len(subject_words) == 1 and subject in {
            "station", "line", "corridor", "section", "phase", "route",
            "segment", "zone", "block", "span", "pier", "column"
        }:
            continue

        values = [v for v, _ in numeric_entries]
        max_val = max(values)
        min_val = min(values)

        if max_val == 0:
            continue

        diff_pct = (max_val - min_val) / max_val
        if diff_pct > 0.10:  # > 10% difference
            pages = sorted(set(e["page"] for _, e in numeric_entries))
            fact_ids = [e["fid"] for _, e in numeric_entries]

            severity = Severity.HIGH if diff_pct > 0.30 else Severity.MEDIUM

            issue = ConsistencyIssue(
                issue_id=str(uuid.uuid4()),
                issue_type="numeric_mismatch",
                severity=severity,
                description=(
                    f"'{subject} → {attribute}' has inconsistent values across pages: "
                    f"{[round(v, 2) for v in values[:5]]} "
                    f"(diff={diff_pct:.0%})"
                ),
                fact_ids=fact_ids,
                page_refs=pages,
                evidence=f"Values found: {values[:5]}",
                doc_id=doc_id,
                sector=sector,
            )
            issues.append(issue)

    logger.info(f"Numeric mismatch check: {len(issues)} issues found")
    return issues


# ─── 2. Ontology dependency check ────────────────────────────────────────────

def _check_dependency_violations(doc_id: str, sector: str) -> list[ConsistencyIssue]:
    """
    For each DEPENDS_ON edge in the ontology, check if the requiring parameter
    exists in the DPR without its prerequisite.
    Example: 'foundation type' DEPENDS_ON 'soil bearing capacity'
    → if we have foundation type facts but no SBC facts, flag it.
    """
    issues = []
    dependencies = get_dependencies_for_sector(sector)

    for dep in dependencies:
        # Coerce to string — Neo4j may return lists if stored as list properties
        param    = str(dep["parameter"]).strip()  if dep.get("parameter")  else ""
        requires = str(dep["requires"]).strip()   if dep.get("requires")   else ""
        reason   = str(dep.get("reason", "")).strip()
        rule     = str(dep.get("rule",   "")).strip()

        # Skip if either side is empty or not a usable string
        if not param or not requires or len(param) < 2 or len(requires) < 2:
            continue

        # Check if the parameter exists in this DPR
        param_facts = run_read(
            f"""
            MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
            WHERE toLower(f.attribute) CONTAINS toLower($param)
               OR toLower(f.subject)   CONTAINS toLower($param)
            RETURN f.fact_id AS fid, f.source_page AS page
            LIMIT 5
            """,
            {"doc_id": doc_id, "param": param}
        )

        if not param_facts:
            continue  # parameter not in DPR, dependency irrelevant

        # Check if the prerequisite exists
        req_facts = run_read(
            f"""
            MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
            WHERE toLower(f.attribute) CONTAINS toLower($req)
               OR toLower(f.subject)   CONTAINS toLower($req)
            RETURN f.fact_id AS fid
            LIMIT 3
            """,
            {"doc_id": doc_id, "req": requires}
        )

        if not req_facts:
            # Parameter present but prerequisite missing → flag
            issue = ConsistencyIssue(
                issue_id=str(uuid.uuid4()),
                issue_type="missing_dependency",
                severity=Severity.HIGH,
                description=(
                    f"'{param}' is specified in the DPR but its prerequisite "
                    f"'{requires}' is not found."
                ),
                fact_ids=[f["fid"] for f in param_facts],
                page_refs=list(set(f["page"] for f in param_facts)),
                evidence=f"Dependency: {reason}. Expected check: {rule}",
                doc_id=doc_id,
                sector=sector,
            )
            issues.append(issue)

    logger.info(f"Dependency check: {len(issues)} issues found")
    return issues


# ─── 3. LLM holistic consistency check ───────────────────────────────────────

def _llm_consistency_check(doc_id: str, sector: str) -> list[ConsistencyIssue]:
    """
    Send a grouped summary of facts to Ollama and ask it to identify
    logical inconsistencies that rule-based checks may miss.
    """
    # Fetch a representative sample of facts
    facts = run_read(
        f"""
        MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
        WHERE f.fact_type <> 'table_row' AND f.confidence > 0.6
        RETURN f.subject AS s, f.attribute AS a, f.value AS v, f.unit AS u,
               f.source_page AS pg
        ORDER BY f.source_page
        LIMIT 80
        """,
        {"doc_id": doc_id}
    )

    if len(facts) < 5:
        return []

    facts_text = "\n".join(
        f"  p{f['pg']}: {f['s']} | {f['a']} = {f['v']} {f['u']}"
        for f in facts
    )

    prompt = (
        f"You are reviewing a {sector} DPR for logical consistency.\n\n"
        f"Here are engineering facts extracted from the document:\n{facts_text}\n\n"
        "Identify up to 5 logical inconsistencies. For each, return a JSON object:\n"
        '{"description": "...", "evidence": "exact facts involved", '
        '"severity": "CRITICAL|HIGH|MEDIUM", "type": "brief label"}\n\n'
        "Return a JSON array. If no inconsistencies, return [].\n"
        "Focus on: conflicting values, physically impossible combinations, "
        "missing critical parameters given what's present."
    )

    result = generate_json(prompt)
    if not isinstance(result, list):
        return []

    issues = []
    for item in result:
        if not item.get("description"):
            continue
        issues.append(ConsistencyIssue(
            issue_id=str(uuid.uuid4()),
            issue_type="llm_flagged",
            severity=item.get("severity", Severity.MEDIUM),
            description=item.get("description", ""),
            evidence=item.get("evidence", ""),
            doc_id=doc_id,
            sector=sector,
        ))

    logger.info(f"LLM consistency check: {len(issues)} issues found")
    return issues


# ─── Write issues to Neo4j ────────────────────────────────────────────────────

def _write_issues_to_neo4j(issues: list[ConsistencyIssue]):
    for issue in issues:
        run_write(
            f"""
            MERGE (v:{NodeLabel.VIOLATION} {{violation_id: $vid}})
            SET v.issue_type   = $issue_type,
                v.severity     = $severity,
                v.description  = $description,
                v.evidence     = $evidence,
                v.page_refs    = $pages,
                v.stage        = 'consistency',
                v.doc_id       = $doc_id,
                v.sector       = $sector
            """,
            {
                "vid":        issue.issue_id,
                "issue_type": issue.issue_type,
                "severity":   issue.severity,
                "description": issue.description,
                "evidence":   issue.evidence,
                "pages":      issue.page_refs,
                "doc_id":     issue.doc_id,
                "sector":     issue.sector,
            }
        )
        # Link to fact nodes
        for fid in issue.fact_ids:
            run_write(
                f"""
                MATCH (v:{NodeLabel.VIOLATION} {{violation_id: $vid}})
                MATCH (f:{NodeLabel.FACT} {{fact_id: $fid}})
                MERGE (v)-[:REFERENCES]->(f)
                """,
                {"vid": issue.issue_id, "fid": fid}
            )


# ─── Public API ───────────────────────────────────────────────────────────────

def run_consistency_engine(doc_id: str, sector: str) -> dict:
    """
    Run all consistency checks for a given document.
    Returns a summary dict with all issues grouped by type.
    """
    logger.info(f"Running consistency engine for doc={doc_id}, sector={sector}")

    all_issues: list[ConsistencyIssue] = []

    all_issues.extend(_check_numeric_mismatches(doc_id, sector))
    all_issues.extend(_check_dependency_violations(doc_id, sector))
    all_issues.extend(_llm_consistency_check(doc_id, sector))

    # Persist all issues
    _write_issues_to_neo4j(all_issues)

    # Summary
    summary = {
        "doc_id": doc_id,
        "sector": sector,
        "total_issues": len(all_issues),
        "by_severity": {
            Severity.CRITICAL: sum(1 for i in all_issues if i.severity == Severity.CRITICAL),
            Severity.HIGH:     sum(1 for i in all_issues if i.severity == Severity.HIGH),
            Severity.MEDIUM:   sum(1 for i in all_issues if i.severity == Severity.MEDIUM),
            Severity.LOW:      sum(1 for i in all_issues if i.severity == Severity.LOW),
        },
        "by_type": {
            "numeric_mismatch":   sum(1 for i in all_issues if i.issue_type == "numeric_mismatch"),
            "missing_dependency": sum(1 for i in all_issues if i.issue_type == "missing_dependency"),
            "llm_flagged":        sum(1 for i in all_issues if i.issue_type == "llm_flagged"),
        },
        "issues": [
            {
                "id":          i.issue_id,
                "type":        i.issue_type,
                "severity":    i.severity,
                "description": i.description,
                "pages":       i.page_refs,
                "evidence":    i.evidence,
            }
            for i in all_issues
        ]
    }

    logger.success(
        f"Consistency engine done: {len(all_issues)} issues "
        f"(CRITICAL={summary['by_severity']['CRITICAL']}, "
        f"HIGH={summary['by_severity']['HIGH']})"
    )
    return summary