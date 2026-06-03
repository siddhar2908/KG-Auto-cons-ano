"""
engines/anomaly_engine.py
--------------------------
Detects anomalies in extracted DPR facts:
  1. Statistical outliers  (z-score > threshold on numeric fact groups)
  2. Order-of-magnitude errors  (value × 10 or ÷ 10 from expected range)
  3. Duplicate value patterns   (identical values repeated where variation expected)
  4. Unit mismatch flags        (same attribute with incompatible units)
  5. LLM-based anomaly scan     (send fact groups to Ollama for domain-aware flagging)

Results persisted as Violation nodes in Neo4j with stage='anomaly'.
"""

import re
import uuid
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from loguru import logger

from config.settings import (
    NodeLabel, RelType, Severity,
    ANOMALY_SIGMA_THRESHOLD, ORDER_OF_MAGNITUDE_RATIO, NUMERIC_DUPLICATE_RATIO
)
from utils.neo4j_client import run_read, run_write
from utils.ollama_client import generate_json


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class AnomalyFlag:
    flag_id: str
    anomaly_type: str     # "statistical_outlier" | "oom_error" | "duplicate_values"
                          # | "unit_mismatch" | "llm_flagged"
    severity: str
    description: str
    fact_ids: list[str] = field(default_factory=list)
    attribute: str = ""
    flagged_value: str = ""
    expected_range: str = ""
    doc_id: str = ""
    sector: str = ""


def _extract_numeric(value_str: str) -> float | None:
    """Extract first numeric value from a string. Returns None if not numeric."""
    try:
        cleaned = re.sub(r"[^\d.\-]", "", str(value_str))
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


# ─── 1. Statistical outlier detection ────────────────────────────────────────

def _detect_statistical_outliers(facts: list[dict]) -> list[AnomalyFlag]:
    """
    Group facts by (sector, attribute), compute z-scores.
    Flag any fact with z-score > ANOMALY_SIGMA_THRESHOLD.
    Requires at least 4 facts in a group to be meaningful.
    """
    flags = []

    # Group by (sector, attribute) for meaningful comparison
    grouped: dict[tuple, list] = defaultdict(list)
    for f in facts:
        key = (f.get("sector", ""), f.get("attribute", "").lower().strip())
        num = _extract_numeric(f.get("value", ""))
        if num is not None:
            grouped[key].append((num, f))

    for (sector, attribute), entries in grouped.items():
        if len(entries) < 4:
            continue

        values = [v for v, _ in entries]
        try:
            mean = statistics.mean(values)
            stdev = statistics.stdev(values)
        except statistics.StatisticsError:
            continue

        if stdev == 0:
            continue

        for val, fact in entries:
            z = abs(val - mean) / stdev
            if z > ANOMALY_SIGMA_THRESHOLD:
                severity = Severity.HIGH if z > ANOMALY_SIGMA_THRESHOLD * 1.5 else Severity.MEDIUM
                flags.append(AnomalyFlag(
                    flag_id=str(uuid.uuid4()),
                    anomaly_type="statistical_outlier",
                    severity=severity,
                    description=(
                        f"'{attribute}' value {val} {fact.get('unit', '')} is a statistical outlier "
                        f"(z={z:.1f}, mean={mean:.2f}, σ={stdev:.2f})"
                    ),
                    fact_ids=[fact["fid"]],
                    attribute=attribute,
                    flagged_value=str(val),
                    expected_range=f"{mean - 2*stdev:.2f} – {mean + 2*stdev:.2f}",
                    doc_id=fact.get("doc_id", ""),
                    sector=sector,
                ))

    logger.debug(f"Statistical outlier check: {len(flags)} flags")
    return flags


# ─── 2. Order-of-magnitude error detection ───────────────────────────────────

# Approximate expected ranges for common engineering parameters
# Format: {attribute_keyword: (min, max, unit_hint)}
_KNOWN_RANGES: dict[str, tuple] = {
    "bearing capacity":   (50, 5000, "kN/m²"),
    "sbc":                (50, 5000, "kN/m²"),
    "pile diameter":      (300, 2000, "mm"),
    "pile length":        (5, 60, "m"),
    "span":               (1, 300, "m"),
    "carriageway width":  (5, 30, "m"),
    "formation width":    (5, 60, "m"),
    "rebar diameter":     (8, 50, "mm"),
    "pavement thickness": (20, 500, "mm"),
    "cbr":                (1, 30, "%"),
    "gradient":           (0.1, 10, "%"),
    "design speed":       (20, 200, "km/h"),
    "msas":               (1, 1000, "million"),
    "runway length":      (500, 5000, "m"),
    # Note: depth is intentionally excluded — underground depths are negative
    # (below ground level convention) which would trigger false OOM flags.
    # Depth anomalies are caught by the LLM scan instead.
}

# Attributes that are expected to be negative — skip OOM check for these
_SIGNED_ATTRIBUTES = {
    "depth", "depth below ground", "depth below ground level",
    "depth/height", "depth_below_ground_level", "elevation",
    "rl", "reduced level", "invert level",
}


def _detect_oom_errors(facts: list[dict]) -> list[AnomalyFlag]:
    """
    Check if numeric fact values are outside known reasonable ranges by
    an order-of-magnitude factor (suggests decimal placement / unit errors).
    """
    flags = []

    for f in facts:
        attr = f.get("attribute", "").lower()
        num = _extract_numeric(f.get("value", ""))
        if num is None or num == 0:
            continue

        # Skip signed attributes — negative values are expected and correct
        if any(signed in attr for signed in _SIGNED_ATTRIBUTES):
            continue

        for keyword, (low, high, unit_hint) in _KNOWN_RANGES.items():
            if keyword not in attr:
                continue

            # Check for OOM error: value is 10× above or below range
            if num < low / ORDER_OF_MAGNITUDE_RATIO or num > high * ORDER_OF_MAGNITUDE_RATIO:
                flags.append(AnomalyFlag(
                    flag_id=str(uuid.uuid4()),
                    anomaly_type="oom_error",
                    severity=Severity.HIGH,
                    description=(
                        f"'{attr}' value {num} {f.get('unit', '')} appears to be an "
                        f"order-of-magnitude error. Expected range: {low}–{high} {unit_hint}. "
                        f"Possible decimal placement mistake?"
                    ),
                    fact_ids=[f["fid"]],
                    attribute=attr,
                    flagged_value=str(num),
                    expected_range=f"{low}–{high} {unit_hint}",
                    doc_id=f.get("doc_id", ""),
                    sector=f.get("sector", ""),
                ))
            break  # matched a keyword, don't check others for same fact

    logger.debug(f"OOM error check: {len(flags)} flags")
    return flags


# ─── 3. Duplicate value detection ────────────────────────────────────────────

def _detect_duplicate_values(facts: list[dict]) -> list[AnomalyFlag]:
    """
    Flag attributes where >NUMERIC_DUPLICATE_RATIO of values are identical.
    Suggests copy-paste errors in structured tables.
    Skips attributes with < 4 facts.
    """
    flags = []

    grouped: dict[str, list] = defaultdict(list)
    for f in facts:
        key = f.get("attribute", "").lower().strip()
        if key:
            grouped[key].append(f)

    for attribute, entries in grouped.items():
        if len(entries) < 4:
            continue

        values = [str(f.get("value", "")).strip() for f in entries]
        value_counts = defaultdict(int)
        for v in values:
            value_counts[v] += 1

        most_common_val, most_common_count = max(value_counts.items(), key=lambda x: x[1])
        ratio = most_common_count / len(values)

        if ratio >= NUMERIC_DUPLICATE_RATIO:
            flags.append(AnomalyFlag(
                flag_id=str(uuid.uuid4()),
                anomaly_type="duplicate_values",
                severity=Severity.MEDIUM,
                description=(
                    f"'{attribute}' has identical value '{most_common_val}' in "
                    f"{most_common_count}/{len(values)} occurrences ({ratio:.0%}). "
                    "Possible copy-paste error in data entry."
                ),
                fact_ids=[f["fid"] for f in entries],
                attribute=attribute,
                flagged_value=most_common_val,
                doc_id=entries[0].get("doc_id", ""),
                sector=entries[0].get("sector", ""),
            ))

    logger.debug(f"Duplicate value check: {len(flags)} flags")
    return flags


# ─── 4. Unit mismatch detection ───────────────────────────────────────────────

def _detect_unit_mismatches(facts: list[dict]) -> list[AnomalyFlag]:
    """
    Flag cases where the same attribute appears with incompatible units
    in different facts (e.g. length in both 'm' and 'km' without scaling).
    """
    flags = []

    # Group by attribute
    grouped: dict[str, list] = defaultdict(list)
    for f in facts:
        key = f.get("attribute", "").lower().strip()
        if key and f.get("unit"):
            grouped[key].append(f)

    # Unit families (units that cannot be mixed without explicit conversion)
    _INCOMPATIBLE = [
        {"mm", "cm", "m", "km"},        # length — mixing OK if scaled; flag if not
        {"kn", "mn", "kn/m²", "mpa"},   # force / pressure
        {"%", "decimal", "ratio"},       # fractions
    ]

    for attribute, entries in grouped.items():
        units_used = set(
            f.get("unit", "").lower().strip().replace(" ", "")
            for f in entries
            if f.get("unit")
        )
        if len(units_used) < 2:
            continue

        # Check if any incompatible family has >1 unit from entries
        for family in _INCOMPATIBLE:
            used_from_family = units_used & family
            if len(used_from_family) >= 2:
                flags.append(AnomalyFlag(
                    flag_id=str(uuid.uuid4()),
                    anomaly_type="unit_mismatch",
                    severity=Severity.MEDIUM,
                    description=(
                        f"'{attribute}' uses mixed units: {used_from_family}. "
                        "Verify all values are on the same scale."
                    ),
                    fact_ids=[f["fid"] for f in entries],
                    attribute=attribute,
                    flagged_value=str(used_from_family),
                    doc_id=entries[0].get("doc_id", ""),
                    sector=entries[0].get("sector", ""),
                ))
                break

    logger.debug(f"Unit mismatch check: {len(flags)} flags")
    return flags


# ─── 5. LLM-based anomaly scan ───────────────────────────────────────────────

def _llm_anomaly_scan(facts: list[dict], sector: str, doc_id: str) -> list[AnomalyFlag]:
    """Ask Ollama to identify domain-aware anomalies in the fact set."""
    if len(facts) < 5:
        return []

    facts_text = "\n".join(
        f"  {f.get('attribute','?')} = {f.get('value','?')} {f.get('unit','')} "
        f"(subject: {f.get('subject','?')})"
        for f in facts[:60]
    )

    prompt = (
        f"You are a senior {sector} engineer reviewing DPR data for anomalies.\n\n"
        f"Engineering facts extracted from a {sector} DPR:\n{facts_text}\n\n"
        "Identify up to 5 unusual, improbable, or suspicious values/combinations.\n"
        "For each anomaly return: "
        '{"description": "...", "attribute": "...", "flagged_value": "...", '
        '"reason": "...", "severity": "HIGH|MEDIUM|LOW"}\n\n'
        "Return a JSON array. If no anomalies, return [].\n"
        "Focus on: physically impossible values, unrealistic assumptions for this sector, "
        "suspicious combinations (e.g. very high load on very shallow foundation)."
    )

    result = generate_json(prompt)
    if not isinstance(result, list):
        return []

    flags = []
    for item in result:
        if not item.get("description"):
            continue
        flags.append(AnomalyFlag(
            flag_id=str(uuid.uuid4()),
            anomaly_type="llm_flagged",
            severity=item.get("severity", Severity.MEDIUM),
            description=item.get("description", ""),
            attribute=item.get("attribute", ""),
            flagged_value=item.get("flagged_value", ""),
            expected_range=item.get("reason", ""),
            doc_id=doc_id,
            sector=sector,
        ))

    logger.debug(f"LLM anomaly scan: {len(flags)} flags")
    return flags


# ─── Write flags to Neo4j ─────────────────────────────────────────────────────

def _write_flags_to_neo4j(flags: list[AnomalyFlag]):
    for flag in flags:
        run_write(
            f"""
            MERGE (v:{NodeLabel.VIOLATION} {{violation_id: $vid}})
            SET v.issue_type     = $anomaly_type,
                v.severity       = $severity,
                v.description    = $description,
                v.attribute      = $attribute,
                v.flagged_value  = $flagged_value,
                v.expected_range = $expected_range,
                v.stage          = 'anomaly',
                v.doc_id         = $doc_id,
                v.sector         = $sector
            """,
            {
                "vid":            flag.flag_id,
                "anomaly_type":   flag.anomaly_type,
                "severity":       flag.severity,
                "description":    flag.description,
                "attribute":      flag.attribute,
                "flagged_value":  flag.flagged_value,
                "expected_range": flag.expected_range,
                "doc_id":         flag.doc_id,
                "sector":         flag.sector,
            }
        )
        for fid in flag.fact_ids:
            run_write(
                f"""
                MATCH (v:{NodeLabel.VIOLATION} {{violation_id: $vid}})
                MATCH (f:{NodeLabel.FACT} {{fact_id: $fid}})
                MERGE (v)-[:REFERENCES]->(f)
                """,
                {"vid": flag.flag_id, "fid": fid}
            )


# ─── Public API ───────────────────────────────────────────────────────────────

def run_anomaly_engine(doc_id: str, sector: str) -> dict:
    """Run all anomaly checks for a document. Returns summary dict."""
    logger.info(f"Running anomaly engine for doc={doc_id}, sector={sector}")

    # Fetch engineering facts only — exclude table_row facts
    # Table rows store JSON strings as their value which produces garbage
    # scientific notation when parsed as numbers by the outlier detector
    facts = run_read(
        f"""
        MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
        WHERE f.fact_type <> 'table_row'
        RETURN f.fact_id AS fid, f.subject AS subject, f.attribute AS attribute,
               f.value AS value, f.unit AS unit, f.source_page AS page,
               f.sector AS sector, f.doc_id AS doc_id
        """,
        {"doc_id": doc_id}
    )

    # Fetch table rows separately for table-specific checks only
    table_facts = run_read(
        f"""
        MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
        WHERE f.fact_type = 'table_row'
        RETURN f.fact_id AS fid, f.subject AS subject, f.attribute AS attribute,
               f.value AS value, f.unit AS unit, f.source_page AS page,
               f.sector AS sector, f.doc_id AS doc_id
        """,
        {"doc_id": doc_id}
    )

    all_flags: list[AnomalyFlag] = []
    # Numeric checks on engineering facts only (not table rows)
    all_flags.extend(_detect_statistical_outliers(facts))
    all_flags.extend(_detect_oom_errors(facts))
    all_flags.extend(_detect_duplicate_values(facts))
    all_flags.extend(_detect_unit_mismatches(facts))
    # LLM scan on engineering facts (table rows handled separately if needed)
    all_flags.extend(_llm_anomaly_scan(facts, sector, doc_id))

    _write_flags_to_neo4j(all_flags)

    summary = {
        "doc_id": doc_id,
        "sector": sector,
        "total_flags": len(all_flags),
        "by_type": {
            "statistical_outlier": sum(1 for f in all_flags if f.anomaly_type == "statistical_outlier"),
            "oom_error":           sum(1 for f in all_flags if f.anomaly_type == "oom_error"),
            "duplicate_values":    sum(1 for f in all_flags if f.anomaly_type == "duplicate_values"),
            "unit_mismatch":       sum(1 for f in all_flags if f.anomaly_type == "unit_mismatch"),
            "llm_flagged":         sum(1 for f in all_flags if f.anomaly_type == "llm_flagged"),
        },
        "by_severity": {
            Severity.CRITICAL: sum(1 for f in all_flags if f.severity == Severity.CRITICAL),
            Severity.HIGH:     sum(1 for f in all_flags if f.severity == Severity.HIGH),
            Severity.MEDIUM:   sum(1 for f in all_flags if f.severity == Severity.MEDIUM),
            Severity.LOW:      sum(1 for f in all_flags if f.severity == Severity.LOW),
        },
        "flags": [
            {
                "id":             f.flag_id,
                "type":           f.anomaly_type,
                "severity":       f.severity,
                "description":    f.description,
                "attribute":      f.attribute,
                "flagged_value":  f.flagged_value,
                "expected_range": f.expected_range,
            }
            for f in all_flags
        ]
    }

    logger.success(
        f"Anomaly engine done: {len(all_flags)} flags "
        f"(HIGH={summary['by_severity']['HIGH']}, "
        f"MEDIUM={summary['by_severity']['MEDIUM']})"
    )
    return summary