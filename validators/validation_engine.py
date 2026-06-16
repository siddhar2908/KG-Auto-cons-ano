"""
validators/validation_engine.py
--------------------------------
Core DPR validation: matches extracted DPR facts against loaded rulebook rules
using Neo4j graph traversal + LLM-based reasoning.

Output is a flat list of Compliant / Non-Compliant rows — one per rule checked —
each with a plain-English reason. No consistency/anomaly data is included here;
those engines run separately in run_engines.py.

Three stages:
  1. Graph-based: Cypher queries match facts to rules by (sector, attribute)
                  and apply threshold comparisons → Compliant or Non-Compliant
  2. LLM-based:   Non-compliant and ambiguous matches are sent to Ollama for
                  plain-language explanation + severity confirmation
  3. Completeness: Mandatory parameters missing from the DPR → Non-Compliant

Scoring:
  Each rule check carries a weight based on its severity:
    CRITICAL → 4 pts (max impact if Non-Compliant)
    HIGH     → 3 pts
    MEDIUM   → 2 pts
    LOW      → 1 pt

  Weighted compliance score = Σ(weights of Compliant checks) / Σ(all weights) × 100

  Verdict bands:
    ≥ 90%  → GOOD
    ≥ 75%  → SATISFACTORY
    ≥ 50%  → NEEDS IMPROVEMENT
    < 50%  → POOR

Output: ValidationReport JSON + Violation nodes in Neo4j.
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


# ─── Severity weights for scoring ─────────────────────────────────────────────

SEVERITY_WEIGHTS = {
    Severity.CRITICAL: 4,
    Severity.HIGH:     3,
    Severity.MEDIUM:   2,
    Severity.LOW:      1,
    Severity.INFO:     1,
}

# Verdict thresholds (weighted compliance %)
VERDICT_BANDS = [
    (90, "GOOD"),
    (75, "SATISFACTORY"),
    (50, "NEEDS IMPROVEMENT"),
    (0,  "POOR"),
]


# ─── Data class ───────────────────────────────────────────────────────────────

@dataclass
class ValidationRow:
    """One row in the validation report — one rule check against one DPR fact."""
    row_id:        str
    classification: str          # "Compliant" | "Non-Compliant"
    check_area:    str           # human-readable check name, e.g. "Design Speed"
    category:      str           # rulebook category, e.g. "Track", "Electrification"
    dpr_value:     str           # what the DPR says  (e.g. "160 kmph")
    rule_expected: str           # what the rule expects (e.g. ">= 160 kmph")
    standard:      str           # standard name + clause (e.g. "RDSO §3.2.1")
    severity:      str           # CRITICAL | HIGH | MEDIUM | LOW
    reason:        str           # plain-English human-readable reason (1–2 sentences)
    source_page:   int           # page in DPR where fact was found (0 = unknown)
    weight:        int           # scoring weight derived from severity
    fact_id:       str = ""
    rule_id:       str = ""
    doc_id:        str = ""
    sector:        str = ""


# ─── Numeric comparison ───────────────────────────────────────────────────────

def _extract_num(s: str) -> Optional[float]:
    try:
        return float(re.sub(r"[^\d.\-]", "", str(s)))
    except (ValueError, TypeError):
        return None


def _compare(fact_val: str, operator: str, threshold: str) -> Optional[bool]:
    """
    Apply operator between fact value and rule threshold.
    Returns True (compliant), False (non-compliant), None (cannot compare).
    """
    fv = _extract_num(fact_val)
    tv = _extract_num(threshold)

    if fv is None or tv is None:
        if operator == "==":
            return fact_val.strip().lower() == threshold.strip().lower()
        if operator == "must_be":
            return threshold.strip().lower() in fact_val.strip().lower()
        return None

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
        parts = re.findall(r"[\d.]+", str(threshold))
        if len(parts) >= 2:
            return float(parts[0]) <= fv <= float(parts[1])

    return None


def _describe_operator(operator: str, threshold: str, unit: str) -> str:
    """Convert rule operator + threshold into human-readable expected value."""
    unit_str = f" {unit}" if unit else ""
    op_words = {
        ">=":          f"at least {threshold}{unit_str}",
        "<=":          f"no more than {threshold}{unit_str}",
        ">":           f"greater than {threshold}{unit_str}",
        "<":           f"less than {threshold}{unit_str}",
        "==":          f"exactly {threshold}{unit_str}",
        "in_range":    f"within range {threshold}{unit_str}",
        "must_be":     f"must be '{threshold}'",
        "must_not_be": f"must not be '{threshold}'",
        "requires":    f"requires '{threshold}'",
        "every":       f"every {threshold}{unit_str}",
        "before":      f"before {threshold}",
        "after":       f"after {threshold}",
    }
    return op_words.get(operator, f"{operator} {threshold}{unit_str}")


# ─── Stage 1: Graph-based fact-rule matching ──────────────────────────────────

def _embed_texts_local(texts: list[str]):
    """
    Embed texts using mxbai-embed-large via Ollama.
    Returns L2-normalised float32 numpy array (N, dim).
    Falls back to zero vectors if Ollama embed is unavailable.
    """
    import numpy as np
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    try:
        import ollama as _ollama
        from config.settings import OLLAMA_EMBED_MODEL
        response = _ollama.embed(model=OLLAMA_EMBED_MODEL, input=texts)
        arr = np.array(response["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return arr / norms
    except Exception as e:
        logger.debug(f"Embedding unavailable for validation: {e}")
        import numpy as np
        return np.zeros((len(texts), 1024), dtype=np.float32)


def _match_facts_to_rules(doc_id: str, sector: str) -> list[ValidationRow]:
    """
    Two-pass fact-rule matching.

    Pass 1 — Deterministic Cypher (exact substring):
      Finds (Fact, Rule) pairs where attribute strings overlap via CONTAINS.
      Fast and precise for exact/partial name matches.

    Pass 2 — Semantic embedding (mxbai-embed-large, cosine ≥ 0.82):
      For every Rule NOT matched in Pass 1, embed the rule attribute text
      and compare against all Fact attribute+subject embeddings.
      Catches synonyms: "formation width" ↔ "carriageway width",
      "design axle load" ↔ "axle loading", "HFL" ↔ "high flood level".
      Threshold 0.82 is intentionally tight to avoid spurious matches.

    Only one row per (fact_id, rule_id) pair — Pass 2 never duplicates Pass 1.
    """
    import numpy as np
    from config.settings import get_applicable_sectors
    applicable_sectors = get_applicable_sectors(sector)

    # ── Pass 1: deterministic Cypher ──────────────────────────────────────────
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
               r.severity    AS rsev
        LIMIT 500
        """,
        {"doc_id": doc_id, "sector": sector, "applicable_sectors": applicable_sectors}
    )

    rows = []
    for p in pairs:
        result = _compare(p["fval"], p["rop"], p["rthresh"])
        severity = p["rsev"] or Severity.HIGH
        weight = SEVERITY_WEIGHTS.get(severity, 2)

        dpr_display = f"{p['fval']} {p['funit']}".strip()
        rule_display = _describe_operator(p["rop"], p["rthresh"], p["runit"] or "")

        if result is True:
            classification = "Compliant"
            reason = (
                f"The DPR reports {p['fattr']} as {dpr_display}, "
                f"which meets the requirement of {rule_display} "
                f"per {p['std']} {p['clause']}."
            )
        elif result is False:
            classification = "Non-Compliant"
            reason = (
                f"The DPR reports {p['fattr']} as {dpr_display}, "
                f"but the standard requires {rule_display} "
                f"per {p['std']} {p['clause']}. This needs to be corrected."
            )
        else:
            # Cannot compare numerically — will be resolved by LLM in stage 2
            classification = "Non-Compliant"
            reason = (
                f"The DPR value for {p['fattr']} is '{dpr_display}'. "
                f"The rule requires {rule_display} per {p['std']} {p['clause']}. "
                f"This could not be automatically verified and needs manual review."
            )

        rows.append(ValidationRow(
            row_id=str(uuid.uuid4()),
            classification=classification,
            check_area=str(p["rattr"]).title(),
            category=str(p["std"]),
            dpr_value=dpr_display,
            rule_expected=rule_display,
            standard=f"{p['std']} {p['clause']}".strip(),
            severity=severity,
            reason=reason,
            source_page=int(p["fpage"] or 0),
            weight=weight,
            fact_id=str(p["fid"]),
            rule_id=str(p["rid"]),
            doc_id=doc_id,
            sector=sector,
        ))

    matched_pairs = {(p["fid"], p["rid"]) for p in pairs}
    compliant_count   = sum(1 for r in rows if r.classification == "Compliant")
    noncompliant_count = sum(1 for r in rows if r.classification == "Non-Compliant")
    logger.info(
        f"Graph matching pass 1: {len(rows)} fact-rule pairs "
        f"(Compliant={compliant_count}, Non-Compliant={noncompliant_count})"
    )

    # ── Pass 2: semantic embedding for unmatched rules ─────────────────────────
    # Load all rules for this sector, then only process rules Pass 1 missed.
    all_rules = run_read(
        f"""
        MATCH (r:{NodeLabel.RULE})-[:{RelType.BELONGS_TO}]->(rs:{NodeLabel.SECTOR})
        WHERE rs.name IN $applicable_sectors
        RETURN r.rule_id AS rid, r.attribute AS rattr, r.threshold AS rthresh,
               r.unit AS runit, r.operator AS rop, r.standard_name AS std,
               r.clause AS clause, r.severity AS rsev, r.rule_text AS rtext,
               r.condition AS rcond
        """,
        {"applicable_sectors": applicable_sectors}
    )
    matched_rule_ids = {p["rid"] for p in pairs}
    unmatched_rules  = [r for r in all_rules if r["rid"] not in matched_rule_ids]

    if unmatched_rules:
        # Load all facts for the document (attribute + subject as search text)
        all_facts = run_read(
            f"""
            MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
            WHERE f.fact_type <> 'table_row'
            RETURN f.fact_id AS fid, f.attribute AS fattr, f.subject AS fsubj,
                   f.value AS fval, f.unit AS funit, f.source_page AS fpage,
                   f.confidence AS fconf
            LIMIT 2000
            """,
            {"doc_id": doc_id}
        )

        if all_facts:
            # Embed all fact texts (attribute + subject concatenated)
            fact_texts = [
                f"{f.get('fattr','')} {f.get('fsubj','')}".strip()
                for f in all_facts
            ]
            fact_embs = _embed_texts_local(fact_texts)  # (N_facts, 1024)

            # Embed all unmatched rule attributes
            rule_texts = [str(r.get("rattr", "")) for r in unmatched_rules]
            rule_embs  = _embed_texts_local(rule_texts)  # (N_rules, 1024)

            SEMANTIC_THRESHOLD = 0.82  # tight — avoids spurious matches

            for r_idx, rule in enumerate(unmatched_rules):
                r_emb = rule_embs[r_idx]
                sims  = fact_embs @ r_emb  # cosine similarity (L2-normed vectors)
                best_fact_idx = int(np.argmax(sims))
                best_sim      = float(sims[best_fact_idx])

                if best_sim < SEMANTIC_THRESHOLD:
                    continue  # no semantic match found

                fact = all_facts[best_fact_idx]
                pair_key = (fact["fid"], rule["rid"])
                if pair_key in matched_pairs:
                    continue  # already covered by Pass 1

                matched_pairs.add(pair_key)
                severity = rule.get("rsev") or Severity.HIGH
                weight   = SEVERITY_WEIGHTS.get(severity, 2)
                dpr_display  = f"{fact.get('fval','')} {fact.get('funit','')}".strip()
                rule_display = _describe_operator(
                    str(rule.get("rop", "")), str(rule.get("rthresh", "")),
                    str(rule.get("runit", "")),
                )
                result = _compare(fact.get("fval", ""), rule.get("rop", ""), str(rule.get("rthresh", "")))

                if result is True:
                    classification = "Compliant"
                    reason = (
                        f"The DPR reports {fact.get('fattr','')} as {dpr_display}, "
                        f"which meets the requirement of {rule_display} per "
                        f"{rule.get('std','')} {rule.get('clause','')}. "
                        f"(Matched semantically, similarity={best_sim:.2f})"
                    )
                elif result is False:
                    classification = "Non-Compliant"
                    reason = (
                        f"The DPR reports {fact.get('fattr','')} as {dpr_display}, "
                        f"but the standard requires {rule_display} per "
                        f"{rule.get('std','')} {rule.get('clause','')}. "
                        f"(Matched semantically, similarity={best_sim:.2f})"
                    )
                else:
                    classification = "Non-Compliant"
                    reason = (
                        f"The DPR contains a semantically similar parameter "
                        f"'{fact.get('fattr','')} ({fact.get('fsubj','')})' "
                        f"(similarity={best_sim:.2f} to rule attribute "
                        f"'{rule.get('rattr','')}'), "
                        f"but the value could not be automatically compared. "
                        f"Manual review needed for {rule.get('std','')} "
                        f"{rule.get('clause','')}."
                    )

                rows.append(ValidationRow(
                    row_id=str(uuid.uuid4()),
                    classification=classification,
                    check_area=str(rule.get("rattr", "")).title(),
                    category=str(rule.get("std", "")),
                    dpr_value=dpr_display,
                    rule_expected=rule_display,
                    standard=f"{rule.get('std','')} {rule.get('clause','')}".strip(),
                    severity=severity,
                    reason=reason,
                    source_page=int(fact.get("fpage") or 0),
                    weight=weight,
                    fact_id=str(fact["fid"]),
                    rule_id=str(rule["rid"]),
                    doc_id=doc_id,
                    sector=sector,
                ))

            sem_count = len(rows) - len(matched_pairs) + len(unmatched_rules)
            logger.info(
                f"Graph matching pass 2 (semantic): "
                f"{sum(1 for r in rows if 'similarity=' in r.reason)} new pairs added"
            )

    return rows


# ─── Stage 2: LLM enrichment for Non-Compliant and ambiguous rows ─────────────

def _llm_enrich_rows(rows: list[ValidationRow]) -> list[ValidationRow]:
    """
    For Non-Compliant rows, ask Ollama to:
      1. Confirm whether it is truly non-compliant or just ambiguous
      2. Write a richer, more specific plain-English reason
      3. Confirm or upgrade severity

    Compliant rows keep their auto-generated reason (no LLM cost needed).
    """
    needs_llm = [r for r in rows if r.classification == "Non-Compliant"]
    if not needs_llm:
        return rows

    batch_size = 10
    updated: dict[str, dict] = {}  # row_id → {classification, severity, reason}

    for i in range(0, len(needs_llm), batch_size):
        batch = needs_llm[i : i + batch_size]

        items_text = "\n".join(
            f"{j+1}. Check: '{r.check_area}' | "
            f"DPR value: {r.dpr_value} | "
            f"Rule requires: {r.rule_expected} | "
            f"Standard: {r.standard} | "
            f"Rule text: {r.reason}"
            for j, r in enumerate(batch)
        )

        prompt = (
            f"You are a senior {batch[0].sector} infrastructure engineer reviewing a DPR "
            f"for compliance against engineering standards.\n\n"
            f"Review these {len(batch)} compliance checks:\n{items_text}\n\n"
            f"For each check (numbered 1–{len(batch)}), decide:\n"
            "  - Is this genuinely Non-Compliant, or was it a false flag?\n"
            "  - Write a single clear reason (1–2 sentences) in plain English that an "
            "    appraisal officer can understand. Be specific about what's wrong and "
            "    what needs to be done. Do not use jargon like 'FAIL' or 'violation'.\n\n"
            'Return a JSON array, each item: '
            '{"index": 1, '
            '"classification": "Compliant" or "Non-Compliant", '
            '"severity": "CRITICAL|HIGH|MEDIUM|LOW", '
            '"reason": "plain English reason ≤200 chars"}\n\n'
            "Only override classification to Compliant if you are confident the check "
            "was incorrectly flagged (e.g. unit mismatch in extraction). "
            "Upgrade severity to CRITICAL only for safety-critical issues."
        )

        results = generate_json(prompt)
        if not isinstance(results, list):
            continue

        for item in results:
            idx = int(item.get("index", 0)) - 1
            if 0 <= idx < len(batch):
                r = batch[idx]
                new_cls = item.get("classification", r.classification)
                new_sev = item.get("severity", r.severity)
                new_reason = item.get("reason", "").strip()
                updated[r.row_id] = {
                    "classification": new_cls if new_cls in ("Compliant", "Non-Compliant") else r.classification,
                    "severity":       new_sev if new_sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW) else r.severity,
                    "reason":         new_reason if new_reason else r.reason,
                }

    # Apply updates
    for r in rows:
        if r.row_id in updated:
            upd = updated[r.row_id]
            r.classification = upd["classification"]
            r.severity       = upd["severity"]
            r.weight         = SEVERITY_WEIGHTS.get(r.severity, 2)
            r.reason         = upd["reason"]

    return rows


# ─── Stage 3: Completeness check (semantic + deterministic) ──────────────────

# Engineering synonym map: common alternative phrasings for mandatory parameters.
# Checked BEFORE embedding so deterministic synonyms are free (no Ollama call).
# Add entries here as you discover false-negatives in production.
_PARAM_SYNONYMS: dict[str, list[str]] = {
    "embankment volume":   ["embankment qty", "embankment quantity", "filling quantity",
                            "earthwork filling", "bank quantity", "earth filling volume"],
    "cutting volume":      ["cutting qty", "cutting quantity", "excavation volume",
                            "earthwork cutting", "excavation quantity"],
    "max bank height":     ["maximum bank height", "max embankment height", "bank height",
                            "maximum fill height", "embankment height"],
    "max cutting depth":   ["maximum cutting depth", "cutting depth", "max excavation depth",
                            "maximum excavation depth"],
    "bridge count":        ["number of bridges", "no of bridges", "total bridges",
                            "bridges count", "bridge quantity"],
    "curve percentage":    ["percentage of curves", "curved track", "curve length",
                            "horizontal curves", "curvature"],
    "total number of stations": ["number of stations", "no of stations", "station count",
                                 "total stations"],
}

# Semantic similarity threshold for completeness (slightly lower than fact-rule
# matching — we just need to confirm presence, not match values)
_COMPLETENESS_SEM_THRESHOLD = 0.78


def _param_present_deterministic(param: str, all_fact_texts: list[str]) -> bool:
    """
    Deterministic check: is param (or any synonym) present in any fact text?
    Checks: exact substring match across attribute+subject texts.
    """
    param_lower = param.lower()
    synonyms    = [s.lower() for s in _PARAM_SYNONYMS.get(param, [])]
    search_terms = [param_lower] + synonyms

    # Split multi-word param and check if ALL words appear in the same fact text
    param_words = set(param_lower.split())

    for text in all_fact_texts:
        text_lower = text.lower()
        # Direct substring match (any synonym)
        if any(term in text_lower for term in search_terms):
            return True
        # All-words match: "embankment" and "volume" both in same fact text
        if len(param_words) > 1 and all(w in text_lower for w in param_words):
            return True

    return False


def _param_present_semantic(
    param: str,
    fact_embs,          # np.ndarray (N, 1024)
    param_emb,          # np.ndarray (1024,)
    all_fact_texts: list[str],
    threshold: float = _COMPLETENESS_SEM_THRESHOLD,
) -> tuple[bool, float, str]:
    """
    Semantic check: is param semantically present in the fact set?
    Returns (found: bool, best_score: float, matched_text: str).
    """
    import numpy as np
    if fact_embs.shape[0] == 0:
        return False, 0.0, ""
    sims     = fact_embs @ param_emb
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    matched  = all_fact_texts[best_idx] if best_idx < len(all_fact_texts) else ""
    return best_sim >= threshold, best_sim, matched


def _check_mandatory_parameters(doc_id: str, sector: str) -> list[ValidationRow]:
    """
    Semantic + deterministic completeness check.

    For each mandatory parameter in the sector ontology:
      Step 1 — Deterministic: substring match + synonym lookup across all fact texts.
                If found → skip (parameter is present).
      Step 2 — Semantic: embed the parameter name and compare against all fact
                attribute+subject embeddings (mxbai-embed-large, cosine ≥ 0.78).
                If found → skip (parameter is present under a different name).
      Step 3 — Only if BOTH steps miss → raise Non-Compliant.

    Deduplication: same parameter across multiple applicable sectors (e.g.
    Rail Infrastructure + Metro both requiring "embankment volume") is only
    flagged once. Sector hierarchy duplication is suppressed by a seen_params set.
    """
    import numpy as np
    from config.settings import get_applicable_sectors
    applicable_sectors = get_applicable_sectors(sector)

    entity_types = run_read(
        f"""
        MATCH (e:{NodeLabel.ONTOLOGY})
        WHERE e.is_entity_type = true
          AND e.sector IN $applicable_sectors
        RETURN e.name AS entity, e.key_parameters AS params, e.sector AS esector
        """,
        {"applicable_sectors": applicable_sectors}
    )

    # Load all fact attribute+subject texts once (avoid N Cypher calls)
    all_facts = run_read(
        f"""
        MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
        WHERE f.fact_type <> 'table_row'
        RETURN f.fact_id AS fid, f.attribute AS fattr, f.subject AS fsubj,
               f.source_page AS fpage
        LIMIT 5000
        """,
        {"doc_id": doc_id}
    )
    all_fact_texts = [
        f"{f.get('fattr','')} {f.get('fsubj','')}".strip()
        for f in all_facts
    ]

    # Embed all fact texts once (single Ollama call for entire completeness check)
    fact_embs = _embed_texts_local(all_fact_texts) if all_fact_texts else np.zeros((0,1024), dtype=np.float32)

    # Collect all unique params to embed in one batch
    all_params: list[str] = []
    seen_params: set[str] = set()
    param_to_entity: dict[str, str] = {}
    for et in entity_types:
        for param in (et.get("params") or []):
            pkey = param.lower().strip()
            if pkey not in seen_params:
                seen_params.add(pkey)
                all_params.append(param)
                param_to_entity[pkey] = et.get("entity", "")

    # Embed all params in one batch
    param_embs = _embed_texts_local(all_params) if all_params else np.zeros((0,1024), dtype=np.float32)

    rows = []
    seen_flagged: set[str] = set()   # prevent duplicate Non-Compliant rows

    for p_idx, param in enumerate(all_params):
        pkey = param.lower().strip()
        entity_name = param_to_entity.get(pkey, "")

        if pkey in seen_flagged:
            continue  # already flagged by a parent sector — skip duplicate

        # Step 1: deterministic substring + synonym check
        if _param_present_deterministic(param, all_fact_texts):
            logger.debug(f"Completeness OK (deterministic): '{param}'")
            continue

        # Step 2: semantic embedding check
        p_emb = param_embs[p_idx]
        sem_found, best_sim, matched_text = _param_present_semantic(
            param, fact_embs, p_emb, all_fact_texts
        )
        if sem_found:
            logger.debug(
                f"Completeness OK (semantic): '{param}' matched "
                f"'{matched_text[:60]}' (sim={best_sim:.3f})"
            )
            continue

        # Step 3: genuinely missing — raise Non-Compliant
        seen_flagged.add(pkey)
        reason_detail = (
            f"Best semantic match was '{matched_text[:50]}' "
            f"(similarity={best_sim:.2f}) — below threshold {_COMPLETENESS_SEM_THRESHOLD}."
            if best_sim > 0 else "No semantically similar text found in the document."
        )
        rows.append(ValidationRow(
            row_id=str(uuid.uuid4()),
            classification="Non-Compliant",
            check_area=str(param).title(),
            category=f"{entity_name} – Completeness",
            dpr_value="Not found in DPR",
            rule_expected="Parameter must be present",
            standard="DPR Completeness / Ontology",
            severity=Severity.HIGH,
            reason=(
                f"The parameter '{param}' is mandatory for {entity_name} in a "
                f"{sector} DPR, but it was not found — neither by keyword search "
                f"nor by semantic similarity. {reason_detail}"
            ),
            source_page=0,
            weight=SEVERITY_WEIGHTS[Severity.HIGH],
            doc_id=doc_id,
            sector=sector,
        ))

    logger.info(
        f"Completeness check: {len(all_params)} params checked, "
        f"{len(rows)} flagged as missing"
    )
    return rows


# ─── Stage 4: Semantic matching (FAISS) ───────────────────────────────────────

def _semantic_fact_rule_matching(doc_id: str, sector: str) -> list[ValidationRow]:
    """
    FAISS semantic matching — find Rule nodes that semantically match KG triples
    even when attribute strings don't exactly overlap.
    Falls back gracefully if FAISS indexes don't exist.
    """
    rows = []
    try:
        from extractors.kg_embeddings import search_edges
        from config.settings import PROCESSED_DIR

        index_path = PROCESSED_DIR / doc_id / "faiss" / "edges.index"
        if not index_path.exists():
            logger.debug("No FAISS edge index found — skipping semantic matching")
            return []

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
            query = f"{rule.get('attr', '')} {rule.get('text', '')}"
            similar_edges = search_edges(query, doc_id, top_k=5)

            for edge in similar_edges:
                if edge["score"] < 0.75:
                    continue
                severity = rule.get("sev") or Severity.MEDIUM
                rule_display = _describe_operator(
                    str(rule.get("op", "")),
                    str(rule.get("threshold", "")),
                    str(rule.get("unit", "")),
                )
                rows.append(ValidationRow(
                    row_id=str(uuid.uuid4()),
                    classification="Non-Compliant",
                    check_area=str(rule.get("attr", "")).title(),
                    category=str(rule.get("std", "")),
                    dpr_value=edge["triple_string"],
                    rule_expected=rule_display,
                    standard=f"{rule.get('std', '')} {rule.get('clause', '')}".strip(),
                    severity=severity,
                    reason=(
                        f"A semantically similar statement was found in the DPR "
                        f"(similarity score: {edge['score']:.2f}), but it could not be "
                        f"automatically verified against the rule. Manual review is needed "
                        f"to confirm compliance with {rule.get('std', '')} {rule.get('clause', '')}."
                    ),
                    source_page=0,
                    weight=SEVERITY_WEIGHTS.get(severity, 2),
                    rule_id=str(rule.get("rid", "")),
                    doc_id=doc_id,
                    sector=sector,
                ))

        logger.info(f"Semantic matching: {len(rows)} candidate rows found")
        return rows

    except Exception as e:
        logger.debug(f"Semantic matching skipped: {e}")
        return []


# ─── Write rows to Neo4j ──────────────────────────────────────────────────────

def _write_rows_to_neo4j(rows: list[ValidationRow]):
    for r in rows:
        run_write(
            f"""
            MERGE (v:{NodeLabel.VIOLATION} {{violation_id: $vid}})
            SET v.classification = $cls,
                v.status         = $status,
                v.severity       = $severity,
                v.check_area     = $check_area,
                v.category       = $category,
                v.dpr_value      = $dpr_value,
                v.rule_expected  = $rule_expected,
                v.standard       = $standard,
                v.reason         = $reason,
                v.weight         = $weight,
                v.issue_type     = 'validation',
                v.stage          = 'validation',
                v.source_page    = $page,
                v.doc_id         = $doc_id,
                v.sector         = $sector
            """,
            {
                "vid":          r.row_id,
                "cls":          r.classification,
                "status":       "PASS" if r.classification == "Compliant" else "FAIL",
                "severity":     r.severity,
                "check_area":   r.check_area,
                "category":     r.category,
                "dpr_value":    r.dpr_value,
                "rule_expected": r.rule_expected,
                "standard":     r.standard,
                "reason":       r.reason,
                "weight":       r.weight,
                "page":         r.source_page,
                "doc_id":       r.doc_id,
                "sector":       r.sector,
            }
        )
        if r.fact_id:
            run_write(
                f"""
                MATCH (fact:{NodeLabel.FACT} {{fact_id: $fid}})
                MATCH (v:{NodeLabel.VIOLATION} {{violation_id: $vid}})
                MERGE (fact)-[:{RelType.VIOLATES}]->(v)
                """,
                {"fid": r.fact_id, "vid": r.row_id}
            )


# ─── Scoring ──────────────────────────────────────────────────────────────────

def _compute_score(rows: list[ValidationRow]) -> dict:
    """
    Weighted compliance score.

    Each row has a weight (1–4) based on severity.
    Score = Σ(weights of Compliant rows) / Σ(all weights) × 100

    Also breaks down score by category (rulebook section).
    """
    if not rows:
        return {
            "weighted_score": 0.0,
            "verdict": "NO_CHECKS_RUN",
            "total_weight": 0,
            "compliant_weight": 0,
            "non_compliant_weight": 0,
            "by_severity": {},
            "by_category": {},
        }

    total_weight      = sum(r.weight for r in rows)
    compliant_weight  = sum(r.weight for r in rows if r.classification == "Compliant")
    non_compliant_w   = total_weight - compliant_weight
    weighted_score    = round(compliant_weight / total_weight * 100, 1) if total_weight else 0.0

    # Verdict band
    verdict = "POOR"
    for threshold, label in VERDICT_BANDS:
        if weighted_score >= threshold:
            verdict = label
            break

    # Breakdown by severity
    by_severity = {}
    for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW):
        sev_rows = [r for r in rows if r.severity == sev]
        if sev_rows:
            sev_compliant = sum(1 for r in sev_rows if r.classification == "Compliant")
            by_severity[sev] = {
                "total":         len(sev_rows),
                "compliant":     sev_compliant,
                "non_compliant": len(sev_rows) - sev_compliant,
                "weight":        sum(r.weight for r in sev_rows),
                "compliant_weight": sum(r.weight for r in sev_rows if r.classification == "Compliant"),
            }

    # Breakdown by category (rulebook section / standard)
    from collections import defaultdict
    cat_data: dict[str, dict] = defaultdict(lambda: {"total": 0, "compliant": 0, "non_compliant": 0})
    for r in rows:
        cat = r.category or "Uncategorised"
        cat_data[cat]["total"] += 1
        if r.classification == "Compliant":
            cat_data[cat]["compliant"] += 1
        else:
            cat_data[cat]["non_compliant"] += 1

    by_category = {}
    for cat, data in cat_data.items():
        total = data["total"]
        compliant = data["compliant"]
        by_category[cat] = {
            "total":         total,
            "compliant":     compliant,
            "non_compliant": data["non_compliant"],
            "category_score": round(compliant / total * 100, 1) if total else 0.0,
        }

    return {
        "weighted_score":       weighted_score,
        "verdict":              verdict,
        "total_weight":         total_weight,
        "compliant_weight":     compliant_weight,
        "non_compliant_weight": non_compliant_w,
        "by_severity":          by_severity,
        "by_category":          by_category,
    }


# ─── Report builder ───────────────────────────────────────────────────────────

def _build_report(
    doc_id: str,
    sector: str,
    rows: list[ValidationRow],
) -> dict:
    """
    Build the final validation report dict.

    Structure:
      doc_id, sector, generated_at
      score:
        weighted_score (0–100)
        verdict (GOOD / SATISFACTORY / NEEDS IMPROVEMENT / POOR)
        total_checks, compliant_count, non_compliant_count
        by_severity, by_category
      results: [
        { classification, check_area, category, dpr_value, rule_expected,
          standard, severity, weight, reason, source_page }
        ...sorted: Non-Compliant first, then by severity weight desc
      ]
    """
    compliant     = [r for r in rows if r.classification == "Compliant"]
    non_compliant = [r for r in rows if r.classification == "Non-Compliant"]

    score = _compute_score(rows)

    # KG stats
    from extractors.kg_builder import get_kg_stats
    try:
        kg_stats = get_kg_stats(doc_id)
    except Exception:
        kg_stats = {}

    # Sort: Non-Compliant first (CRITICAL → LOW), then Compliant
    sev_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3, Severity.INFO: 4}
    sorted_rows = sorted(
        non_compliant, key=lambda r: sev_order.get(r.severity, 5)
    ) + sorted(
        compliant, key=lambda r: sev_order.get(r.severity, 5)
    )

    return {
        "doc_id":       doc_id,
        "sector":       sector,
        "generated_at": datetime.now().isoformat(),
        "kg_stats":     kg_stats,
        "score": {
            "weighted_score":       score["weighted_score"],
            "verdict":              score["verdict"],
            "total_checks":         len(rows),
            "compliant_count":      len(compliant),
            "non_compliant_count":  len(non_compliant),
            "compliant_weight":     score["compliant_weight"],
            "non_compliant_weight": score["non_compliant_weight"],
            "total_weight":         score["total_weight"],
            "by_severity":          score["by_severity"],
            "by_category":          score["by_category"],
        },
        "results": [
            {
                "classification": r.classification,
                "check_area":     r.check_area,
                "category":       r.category,
                "dpr_value":      r.dpr_value,
                "rule_expected":  r.rule_expected,
                "standard":       r.standard,
                "severity":       r.severity,
                "weight":         r.weight,
                "reason":         r.reason,
                "source_page":    r.source_page,
            }
            for r in sorted_rows
        ],
    }


# ─── Pre-validation cleanup ───────────────────────────────────────────────────

def _pre_validation_cleanup(doc_id: str, min_confidence: float = 0.45):
    """
    Filter low-confidence facts and remove duplicates from Neo4j BEFORE validation.
    Runs AFTER the consistency and anomaly engines have already used the full fact set.
    """
    deleted_low_conf = run_write(
        f"""
        MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
        WHERE f.confidence < $min_conf
          AND f.fact_type <> 'table_row'
        DETACH DELETE f
        """,
        {"doc_id": doc_id, "min_conf": min_confidence}
    )

    dupes = run_read(
        f"""
        MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
        WHERE f.fact_type <> 'table_row'
        WITH f.subject AS subj, f.attribute AS attr, f.value AS val, collect(f) AS facts
        WHERE size(facts) > 1
        RETURN subj, attr, val, facts
        """,
        {"doc_id": doc_id}
    )

    deleted_dupes = 0
    for group in dupes:
        facts = group["facts"]
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


# ─── Public API ───────────────────────────────────────────────────────────────

def run_validation(
    doc_id: str,
    sector: str,
    save_report: bool = True,
    min_confidence: float = 0.45,
) -> dict:
    """
    Full validation pipeline for a document.
    Expects facts (from DPR) and rules (from rulebook) already loaded in Neo4j.

    Returns a validation report dict with:
      - Weighted compliance score and verdict
      - Flat list of Compliant / Non-Compliant rows with plain-English reasons
    """
    logger.info(f"Running validation for doc={doc_id}, sector={sector}")

    # Cleanup: filter low-confidence + dedup facts (after engines have run)
    _pre_validation_cleanup(doc_id, min_confidence)

    # Stage 1: Graph-based exact matching
    rows = _match_facts_to_rules(doc_id, sector)

    # Stage 1B: FAISS semantic matching (skipped if no index)
    semantic_rows = _semantic_fact_rule_matching(doc_id, sector)
    existing_rule_ids = {r.rule_id for r in rows}
    for sr in semantic_rows:
        if sr.rule_id not in existing_rule_ids:
            rows.append(sr)
            existing_rule_ids.add(sr.rule_id)

    # Stage 2: LLM enrichment for Non-Compliant rows
    if rows:
        rows = _llm_enrich_rows(rows)

    # Stage 3: Completeness check
    completeness_rows = _check_mandatory_parameters(doc_id, sector)
    rows.extend(completeness_rows)

    # Persist to Neo4j
    _write_rows_to_neo4j(rows)

    # Build report
    report = _build_report(doc_id, sector, rows)

    if save_report:
        out_path = OUTPUT_DIR / f"validation_report_{doc_id}.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.success(f"Validation report saved: {out_path}")

    logger.success(
        f"Validation complete: score={report['score']['weighted_score']}%, "
        f"verdict={report['score']['verdict']}, "
        f"Compliant={report['score']['compliant_count']}, "
        f"Non-Compliant={report['score']['non_compliant_count']}"
    )
    return report