"""
validators/validation_engine.py
--------------------------------
Core DPR validation: matches extracted DPR facts against loaded rules
using Neo4j graph traversal + LLM-based reasoning.

Three stages:
  1. Graph-based: Cypher queries match facts to rules by (sector, attribute)
                  and apply threshold comparisons
  2. LLM-based:  Flagged matches are sent to Ollama for natural-language
                 explanation and severity confirmation
  3. Completeness: Check for mandatory parameters (from ontology entity types)
                   that are missing from the DPR

Output: ValidationReport written as JSON + Violation nodes in Neo4j.
"""

import uuid
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime

from loguru import logger

from config.settings import NodeLabel, RelType, Severity, OUTPUT_DIR
from utils.neo4j_client import run_read, run_write
from utils.ollama_client import generate_json, generate, get_model_for_task, TaskType


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ValidationFinding:
    finding_id: str
    fact_id: str
    rule_id: str
    fact_value: str
    fact_unit: str
    rule_threshold: str
    rule_unit: str
    rule_operator: str
    attribute: str
    subject: str
    standard_name: str
    clause: str
    rule_text: str
    status: str           # "PASS" | "FAIL" | "WARNING" | "MISSING"
    severity: str
    explanation: str = ""
    source_page: int = 0
    doc_id: str = ""
    sector: str = ""


# ─── Numeric comparison ───────────────────────────────────────────────────────

def _extract_num(s: str) -> Optional[float]:
    try:
        return float(re.sub(r"[^\d.\-]", "", str(s)))
    except (ValueError, TypeError):
        return None


def _compare(fact_val: str, operator: str, threshold: str) -> Optional[bool]:
    """
    Apply comparison operator between fact value and rule threshold.
    Returns True (pass), False (fail), or None (cannot compare — not numeric).
    """
    fv = _extract_num(fact_val)
    tv = _extract_num(threshold)

    if fv is None or tv is None:
        # Non-numeric: do string/category comparison
        if operator == "==":
            return fact_val.strip().lower() == threshold.strip().lower()
        if operator == "must_be":
            return threshold.strip().lower() in fact_val.strip().lower()
        return None  # cannot compare

    ops = {
        ">=": fv >= tv,
        "<=": fv <= tv,
        ">":  fv > tv,
        "<":  fv < tv,
        "==": abs(fv - tv) < 1e-6,
    }
    if operator in ops:
        return ops[operator]

    if operator == "in_range":
        # threshold format: "min-max"
        parts = re.findall(r"[\d.]+", str(threshold))
        if len(parts) >= 2:
            return float(parts[0]) <= fv <= float(parts[1])

    return None  # unknown operator


# ─── Stage 1: Graph-based fact-rule matching ──────────────────────────────────

def _match_facts_to_rules(doc_id: str, sector: str) -> list[ValidationFinding]:
    """
    Cypher: find (Fact, Rule) pairs with matching sector and attribute.
    Uses sector hierarchy — Metro facts are matched against Metro AND
    Rail Infrastructure rules, Bridges against Bridges AND Highways rules, etc.
    """
    from config.settings import get_applicable_sectors
    applicable_sectors = get_applicable_sectors(sector)

    pairs = run_read(
        f"""
        MATCH (f:{NodeLabel.FACT})-[:{RelType.BELONGS_TO}]->(fs:{NodeLabel.SECTOR})
        MATCH (r:{NodeLabel.RULE})-[:{RelType.BELONGS_TO}]->(rs:{NodeLabel.SECTOR})
        WHERE f.doc_id = $doc_id
          AND f.fact_type <> 'table_row'
          AND fs.name = $sector
          AND rs.name IN $applicable_sectors
          AND (
            toLower(f.attribute) CONTAINS toLower(r.attribute)
            OR toLower(r.attribute) CONTAINS toLower(f.attribute)
          )
          AND (r.condition IS NULL OR r.condition = ''
               OR toLower(f.context) CONTAINS toLower(r.condition))
        RETURN f.fact_id     AS fid,
               f.value       AS fval,
               f.unit        AS funit,
               f.attribute   AS fattr,
               f.subject     AS fsubj,
               f.source_page AS fpage,
               r.rule_id     AS rid,
               r.threshold   AS rthresh,
               r.unit        AS runit,
               r.operator    AS rop,
               r.attribute   AS rattr,
               r.standard_name AS std,
               r.clause      AS clause,
               r.rule_text   AS rtext,
               r.severity    AS rsev,
               rs.name       AS rule_sector
        LIMIT 500
        """,
        {"doc_id": doc_id, "sector": sector,
         "applicable_sectors": applicable_sectors}
    )

    findings = []
    for p in pairs:
        result = _compare(p["fval"], p["rop"], p["rthresh"])

        if result is True:
            status = "PASS"
            severity = Severity.INFO
        elif result is False:
            status = "FAIL"
            severity = p["rsev"] or Severity.HIGH
        else:
            # Cannot compare — send to LLM stage
            status = "WARNING"
            severity = Severity.MEDIUM

        findings.append(ValidationFinding(
            finding_id=str(uuid.uuid4()),
            fact_id=p["fid"],
            rule_id=p["rid"],
            fact_value=str(p["fval"]),
            fact_unit=str(p["funit"] or ""),
            rule_threshold=str(p["rthresh"]),
            rule_unit=str(p["runit"] or ""),
            rule_operator=str(p["rop"]),
            attribute=str(p["fattr"]),
            subject=str(p["fsubj"]),
            standard_name=str(p["std"]),
            clause=str(p["clause"]),
            rule_text=str(p["rtext"]),
            status=status,
            severity=severity,
            source_page=int(p["fpage"] or 0),
            doc_id=doc_id,
            sector=sector,
        ))

    logger.info(
        f"Graph matching: {len(findings)} fact-rule pairs evaluated "
        f"(FAIL={sum(1 for f in findings if f.status == 'FAIL')}, "
        f"PASS={sum(1 for f in findings if f.status == 'PASS')})"
    )
    return findings


# ─── Stage 2: LLM reasoning for non-numeric and WARNING findings ──────────────

def _llm_explain_findings(findings: list[ValidationFinding]) -> list[ValidationFinding]:
    """
    For FAIL and WARNING findings, ask Ollama to:
    1. Confirm whether it's a real violation
    2. Provide a plain-language explanation
    3. Suggest corrective action
    """
    needs_llm = [f for f in findings if f.status in ("FAIL", "WARNING")]
    if not needs_llm:
        return findings

    # Batch into groups of 10 to limit LLM calls
    batch_size = 10
    for i in range(0, len(needs_llm), batch_size):
        batch = needs_llm[i:i + batch_size]

        items_text = "\n".join(
            f"{j+1}. [{f.status}] '{f.attribute}' in '{f.subject}': "
            f"DPR value = {f.fact_value} {f.fact_unit}, "
            f"Rule ({f.standard_name} {f.clause}): "
            f"must be {f.rule_operator} {f.rule_threshold} {f.rule_unit}. "
            f"Rule: {f.rule_text}"
            for j, f in enumerate(batch)
        )

        prompt = (
            f"You are validating a {batch[0].sector} DPR against engineering standards.\n\n"
            f"Review these {len(batch)} findings:\n{items_text}\n\n"
            f"For each finding (numbered 1–{len(batch)}), return:\n"
            '{"index": 1, "status": "FAIL|WARNING|PASS", '
            '"severity": "CRITICAL|HIGH|MEDIUM|LOW", '
            '"explanation": "plain English explanation ≤150 chars", '
            '"action": "what the DPR author should do ≤100 chars"}\n\n'
            "Return a JSON array. Confirm or upgrade severity based on engineering judgment."
        )

        results = generate_json(prompt)

        if isinstance(results, list):
            for item in results:
                idx = int(item.get("index", 0)) - 1
                if 0 <= idx < len(batch):
                    f = batch[idx]
                    f.status = item.get("status", f.status)
                    f.severity = item.get("severity", f.severity)
                    f.explanation = (
                        item.get("explanation", "") +
                        (" — " + item.get("action", "") if item.get("action") else "")
                    )

    return findings


# ─── Stage 3: Completeness check against ontology ────────────────────────────

def _check_mandatory_parameters(doc_id: str, sector: str) -> list[ValidationFinding]:
    """
    Check if mandatory parameters defined in sector entity types are present.
    Returns MISSING findings for absent mandatory parameters.
    """
    # Get mandatory checks from ontology entity types
    from config.settings import get_applicable_sectors
    applicable_sectors = get_applicable_sectors(sector)

    entity_types = run_read(
        f"""
        MATCH (e:{NodeLabel.ONTOLOGY})
        WHERE e.is_entity_type = true
          AND e.sector IN $applicable_sectors
        RETURN e.name AS entity, e.key_parameters AS params,
               e.mandatory_checks AS checks, e.sector AS esector
        """,
        {"applicable_sectors": applicable_sectors}
    )

    findings = []
    for et in entity_types:
        for param in (et.get("params") or []):
            # Check if this parameter exists in DPR facts
            found = run_read(
                f"""
                MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
                WHERE toLower(f.attribute) CONTAINS toLower($param)
                   OR toLower(f.subject) CONTAINS toLower($param)
                RETURN count(f) AS cnt
                """,
                {"doc_id": doc_id, "param": param}
            )
            count = found[0]["cnt"] if found else 0

            if count == 0:
                findings.append(ValidationFinding(
                    finding_id=str(uuid.uuid4()),
                    fact_id="",
                    rule_id="",
                    fact_value="MISSING",
                    fact_unit="",
                    rule_threshold="present",
                    rule_unit="",
                    rule_operator="must_be",
                    attribute=param,
                    subject=et.get("entity", ""),
                    standard_name="Ontology / DPR Completeness",
                    clause="",
                    rule_text=f"'{param}' is a mandatory parameter for {et.get('entity', '')} in {sector} DPRs",
                    status="MISSING",
                    severity=Severity.HIGH,
                    explanation=f"Parameter '{param}' is required for {et.get('entity', '')} but not found in DPR",
                    doc_id=doc_id,
                    sector=sector,
                ))

    logger.info(f"Completeness check: {len(findings)} missing mandatory parameters")
    return findings


# ─── Write findings to Neo4j ──────────────────────────────────────────────────

def _write_findings_to_neo4j(findings: list[ValidationFinding]):
    for f in findings:
        run_write(
            f"""
            MERGE (v:{NodeLabel.VIOLATION} {{violation_id: $vid}})
            SET v.status        = $status,
                v.severity      = $severity,
                v.issue_type    = 'validation',
                v.description   = $desc,
                v.explanation   = $explanation,
                v.attribute     = $attribute,
                v.subject       = $subject,
                v.standard_name = $standard,
                v.clause        = $clause,
                v.rule_text     = $rule_text,
                v.fact_value    = $fact_val,
                v.rule_threshold= $threshold,
                v.stage         = 'validation',
                v.source_page   = $page,
                v.doc_id        = $doc_id,
                v.sector        = $sector
            """,
            {
                "vid":         f.finding_id,
                "status":      f.status,
                "severity":    f.severity,
                "desc":        f"{f.status}: {f.attribute} = {f.fact_value} {f.fact_unit} "
                               f"(rule: {f.rule_operator} {f.rule_threshold} {f.rule_unit})",
                "explanation": f.explanation,
                "attribute":   f.attribute,
                "subject":     f.subject,
                "standard":    f.standard_name,
                "clause":      f.clause,
                "rule_text":   f.rule_text,
                "fact_val":    f.fact_value,
                "threshold":   f.rule_threshold,
                "page":        f.source_page,
                "doc_id":      f.doc_id,
                "sector":      f.sector,
            }
        )

        # Link fact → violation, violation → rule
        if f.fact_id:
            run_write(
                f"""
                MATCH (fact:{NodeLabel.FACT} {{fact_id: $fid}})
                MATCH (v:{NodeLabel.VIOLATION} {{violation_id: $vid}})
                MERGE (fact)-[:{RelType.VIOLATES}]->(v)
                """,
                {"fid": f.fact_id, "vid": f.finding_id}
            )


# ─── Report generation ────────────────────────────────────────────────────────

def _build_report(
    doc_id: str,
    sector: str,
    findings: list[ValidationFinding],
    consistency_summary: dict,
    anomaly_summary: dict,
) -> dict:
    fails = [f for f in findings if f.status == "FAIL"]
    passes = [f for f in findings if f.status == "PASS"]
    warnings = [f for f in findings if f.status == "WARNING"]
    missing = [f for f in findings if f.status == "MISSING"]

    overall_score = (
        len(passes) / max(len(findings), 1) * 100
        if findings else 0
    )

    # KG stats
    from extractors.kg_builder import get_kg_stats
    try:
        kg_stats = get_kg_stats(doc_id)
    except Exception:
        kg_stats = {}

    return {
        "doc_id": doc_id,
        "sector": sector,
        "generated_at": datetime.now().isoformat(),
        "kg_stats": kg_stats,
        "overall_score": round(overall_score, 1),
        "verdict": (
            "NO_RULES_LOADED"  if len(findings) == 0 and len(missing) > 0
            else "CRITICAL_ISSUES" if any(f.severity == Severity.CRITICAL for f in fails)
            else "MAJOR_ISSUES" if len(fails) > 5
            else "MINOR_ISSUES" if len(fails) > 0 or len(warnings) > 3
            else "PASS"
        ),
        "summary": {
            "total_checks": len(findings),
            "pass":    len(passes),
            "fail":    len(fails),
            "warning": len(warnings),
            "missing": len(missing),
        },
        "consistency": consistency_summary,
        "anomalies": anomaly_summary,
        "findings": {
            "failures": [
                {
                    "attribute":     f.attribute,
                    "subject":       f.subject,
                    "dpr_value":     f"{f.fact_value} {f.fact_unit}",
                    "rule":          f"{f.rule_operator} {f.rule_threshold} {f.rule_unit}",
                    "standard":      f"{f.standard_name} {f.clause}",
                    "severity":      f.severity,
                    "explanation":   f.explanation,
                    "source_page":   f.source_page,
                }
                for f in sorted(fails, key=lambda x: (x.severity, x.attribute))
            ],
            "missing_parameters": list({
                (f.attribute, f.subject): {
                    "parameter": f.attribute,
                    "entity":    f.subject,
                    "rule_text": f.rule_text,
                }
                for f in missing
            }.values()),
            "warnings": [
                {
                    "attribute":     f.attribute,
                    "explanation":   f.explanation,
                    "source_page":   f.source_page,
                }
                for f in warnings[:20]  # cap at 20 warnings in report
            ],
        }
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def _pre_validation_cleanup(doc_id: str, min_confidence: float = 0.45):
    """
    Filter low-confidence facts and remove duplicates from Neo4j BEFORE validation.
    This runs AFTER the consistency and anomaly engines have already used the
    full unfiltered fact set for their cross-section mismatch and outlier checks.

    What gets removed:
      - Facts with confidence < min_confidence
      - Duplicate (subject, attribute, value) facts — keep highest confidence per page
    """
    # Delete low-confidence facts
    deleted_low_conf = run_write(
        f"""
        MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
        WHERE f.confidence < $min_conf
          AND f.fact_type <> 'table_row'
        DETACH DELETE f
        """,
        {"doc_id": doc_id, "min_conf": min_confidence}
    )

    # For duplicates: keep highest confidence, delete the rest
    # Find duplicate groups (same subject+attribute+value, different pages)
    dupes = run_read(
        f"""
        MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
        WHERE f.fact_type <> 'table_row'
        WITH f.subject AS subj, f.attribute AS attr, f.value AS val,
             collect(f) AS facts
        WHERE size(facts) > 1
        RETURN subj, attr, val, facts
        """,
        {"doc_id": doc_id}
    )

    deleted_dupes = 0
    for group in dupes:
        facts = group["facts"]
        # Sort by confidence descending — keep first, delete rest
        facts_sorted = sorted(facts, key=lambda x: float(x.get("confidence", 0)), reverse=True)
        for f in facts_sorted[1:]:
            run_write(
                f"MATCH (f:{NodeLabel.FACT} {{fact_id: $fid}}) DETACH DELETE f",
                {"fid": f["fact_id"]}
            )
            deleted_dupes += 1

    logger.info(
        f"Pre-validation cleanup: removed {deleted_low_conf} low-confidence facts, "
        f"{deleted_dupes} duplicate facts"
    )
    return {"removed_low_conf": deleted_low_conf, "removed_dupes": deleted_dupes}


def _semantic_fact_rule_matching(doc_id: str, sector: str) -> list[ValidationFinding]:
    """
    Path B: FAISS semantic matching — find Rule nodes that semantically
    match KG triples even when attribute strings don't exactly overlap.
    Falls back gracefully if FAISS indexes don't exist yet.
    """
    findings = []
    try:
        from extractors.kg_embeddings import search_edges
        from config.settings import PROCESSED_DIR
        import os

        # Check if FAISS index exists
        index_path = PROCESSED_DIR / doc_id / "faiss" / "edges.index"
        if not index_path.exists():
            logger.debug("No FAISS edge index found — skipping semantic matching")
            return []

        # Get all rules for this sector
        rules = run_read(
            f"""
            MATCH (r:{NodeLabel.RULE})-[:{RelType.BELONGS_TO}]->(s:{NodeLabel.SECTOR} {{name: $sector}})
            RETURN r.rule_id AS rid, r.attribute AS attr, r.rule_text AS text,
                   r.threshold AS threshold, r.operator AS op, r.unit AS unit,
                   r.standard_name AS std, r.clause AS clause, r.severity AS sev
            """,
            {"sector": sector}
        )

        if not rules:
            return []

        for rule in rules:
            # Search for semantically similar triples
            query = f"{rule.get('attr', '')} {rule.get('text', '')}"
            similar_edges = search_edges(query, doc_id, top_k=5)

            for edge in similar_edges:
                if edge["score"] < 0.75:  # minimum semantic similarity threshold
                    continue

                # Check if this triple-rule pair already found by exact matching
                # by looking for existing finding with same rule
                findings.append(ValidationFinding(
                    finding_id=str(uuid.uuid4()),
                    fact_id="",    # semantic match — no direct fact node
                    rule_id=rule.get("rid", ""),
                    fact_value=edge["triple_string"],
                    fact_unit="",
                    rule_threshold=str(rule.get("threshold", "")),
                    rule_unit=str(rule.get("unit", "")),
                    rule_operator=str(rule.get("op", "")),
                    attribute=str(rule.get("attr", "")),
                    subject="KG triple",
                    standard_name=str(rule.get("std", "")),
                    clause=str(rule.get("clause", "")),
                    rule_text=str(rule.get("text", "")),
                    status="WARNING",   # semantic matches need LLM confirmation
                    severity=Severity.MEDIUM,
                    explanation=f"Semantic match (score={edge['score']:.2f}): {edge['triple_string']}",
                    doc_id=doc_id,
                    sector=sector,
                ))

        logger.info(f"Semantic matching: {len(findings)} candidate matches found")
        return findings

    except Exception as e:
        logger.debug(f"Semantic matching skipped: {e}")
        return []


def run_validation(
    doc_id: str,
    sector: str,
    consistency_summary: dict = None,
    anomaly_summary: dict = None,
    save_report: bool = True,
    min_confidence: float = 0.45,
) -> dict:
    """
    Full validation pipeline for a document.
    Expects facts and rules already loaded in Neo4j.
    Runs filter+dedup cleanup first (after engines have already used full data).
    Returns the validation report dict.
    """
    logger.info(f"Running validation for doc={doc_id}, sector={sector}")

    # Pre-validation cleanup: filter + dedup NOW (after engines used full data)
    cleanup = _pre_validation_cleanup(doc_id, min_confidence)
    logger.info(
        f"Facts remaining after cleanup: "
        f"removed {cleanup['removed_low_conf']} low-conf + "
        f"{cleanup['removed_dupes']} dupes"
    )

    # Stage 1A: Graph-based exact matching (Cypher)
    findings = _match_facts_to_rules(doc_id, sector)

    # Stage 1B: FAISS semantic matching (KG triples vs rules)
    semantic_findings = _semantic_fact_rule_matching(doc_id, sector)
    # Merge — avoid duplicate rule matches
    existing_rule_ids = {f.rule_id for f in findings}
    for sf in semantic_findings:
        if sf.rule_id not in existing_rule_ids:
            findings.append(sf)
            existing_rule_ids.add(sf.rule_id)

    # Stage 2: LLM reasoning for failures and warnings
    if findings:
        findings = _llm_explain_findings(findings)

    # Stage 3: Completeness check
    completeness_findings = _check_mandatory_parameters(doc_id, sector)
    findings.extend(completeness_findings)

    # Persist all findings
    _write_findings_to_neo4j(findings)

    # Build report
    report = _build_report(
        doc_id, sector, findings,
        consistency_summary or {},
        anomaly_summary or {},
    )

    if save_report:
        out_path = OUTPUT_DIR / f"validation_report_{doc_id}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.success(f"Validation report saved: {out_path}")

    logger.success(
        f"Validation complete: score={report['overall_score']:.1f}%, "
        f"verdict={report['verdict']}, "
        f"FAIL={report['summary']['fail']}, PASS={report['summary']['pass']}"
    )
    return report