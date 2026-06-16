#!/usr/bin/env python3
"""
run_push.py
-----------
STEP 1B: Filter → Deduplicate → Normalize → Push to Neo4j → Generate Ontology

Reads raw JSON from data/processed/<doc_id>/ (output of run_extraction.py),
cleans the data, and writes it to Neo4j.

Separation from extraction means:
  - You can inspect/edit raw JSON before DB push
  - Re-push with different normalization without re-running LLM
  - Dedup logic is explicit and auditable

Filter rules:
  - Drop facts with confidence < MIN_CONFIDENCE
  - Drop facts with empty or whitespace value/subject
  - Drop table rows that are pure headers (all values identical to keys)

Dedup rules:
  - Same (doc_id, subject, attribute, value) → keep highest confidence
  - Unit-normalized duplicates → keep canonical form

Normalization:
  - Length:   mm/cm/m/km → metres (canonical)
  - Force:    kN/MN      → kN (canonical)
  - Area:     sqm/m²/ha  → m² (canonical)
  - Numbers:  "7,500" → 7500, "approx 7.5" → 7.5
  - Strings:  strip extra whitespace, title-case subjects

Usage:
    python run_push.py --doc-id <id>
    python run_push.py --from-state
    python run_push.py --doc-id <id> --min-confidence 0.6
    python run_push.py --doc-id <id> --skip-ontology
    python run_push.py --doc-id <id> --dry-run   (shows what would be pushed, no DB write)
"""

import sys
import re
import uuid
import json
import argparse
from pathlib import Path
from collections import defaultdict

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import NodeLabel, RelType, PROCESSED_DIR
from utils.neo4j_client import init_schema, run_write
from extractors.ontology_generator import generate_ontology

console = Console()

# ─── Constants ────────────────────────────────────────────────────────────────

MIN_CONFIDENCE    = 0.45   # facts below this are dropped
MIN_VALUE_LENGTH  = 1      # value must have at least 1 non-whitespace char
MIN_SUBJECT_LENGTH = 2


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DPR Validation — Step 1B: Normalize → Neo4j (filter/dedup happens AFTER engines)",
    )
    parser.add_argument("--doc-id",         type=str,  help="Document ID from run_extraction.py")
    parser.add_argument("--rulebook-id",    type=str,  help="Rulebook doc ID to also push rules")
    parser.add_argument("--from-state",     action="store_true", help="Load IDs from output/.extraction_state.json")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--skip-ontology",  action="store_true")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Show what would be pushed without writing to Neo4j")
    return parser.parse_args()


# ─── Load raw JSON ────────────────────────────────────────────────────────────

def load_raw(doc_id: str) -> dict:
    doc_dir = PROCESSED_DIR / doc_id
    if not doc_dir.exists():
        console.print(f"[red]No processed data for doc_id={doc_id}. Run run_extraction.py first.[/red]")
        sys.exit(1)

    def _load(fname):
        p = doc_dir / fname
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    return {
        "metadata":   _load("metadata.json") or {},
        "facts":      _load("facts_raw.json") or [],
        "tables":     _load("tables_raw.json") or {},
        "rules":      _load("rules_raw.json") or [],
    }


# ─── Normalization helpers ────────────────────────────────────────────────────

# Unit conversion table: (pattern, canonical_unit, multiplier)
_UNIT_NORMS = [
    # Length → metres
    (re.compile(r"^mm$",          re.I), "m",   0.001),
    (re.compile(r"^cms?$",        re.I), "m",   0.01),
    (re.compile(r"^metres?$",     re.I), "m",   1.0),
    (re.compile(r"^kms?$",        re.I), "m",   1000.0),
    # Force / load → kN
    (re.compile(r"^mn$",          re.I), "kN",  1000.0),
    (re.compile(r"^kilo.?newtons?$", re.I), "kN", 1.0),
    # Pressure → kN/m²
    (re.compile(r"^mpa$",         re.I), "kN/m²", 1000.0),
    (re.compile(r"^n/mm2$",       re.I), "kN/m²", 1000.0),
    (re.compile(r"^kpa$",         re.I), "kN/m²", 1.0),
    # Area → m²
    (re.compile(r"^sqm$",         re.I), "m²",  1.0),
    (re.compile(r"^sq\.?\s*m$",   re.I), "m²",  1.0),
    (re.compile(r"^hectares?$",   re.I), "m²",  10000.0),
    (re.compile(r"^ha$",          re.I), "m²",  10000.0),
]

# Unit canonicalization — case/spelling variants → canonical form
# Applied BEFORE numeric conversion to clean up inconsistent LLM output
_UNIT_CANON = [
    # Speed variants → km/h
    (re.compile(r"^kmph$",        re.I), "km/h"),
    (re.compile(r"^km/hr$",       re.I), "km/h"),
    (re.compile(r"^kph$",         re.I), "km/h"),
    # Time variants → canonical
    (re.compile(r"^seconds?$",    re.I), "s"),
    (re.compile(r"^sec\.?$",      re.I), "s"),
    (re.compile(r"^secs?\.?$",    re.I), "s"),
    (re.compile(r"^minutes?$",    re.I), "min"),
    (re.compile(r"^mins?\.?$",    re.I), "min"),
    # Power variants → kW
    (re.compile(r"^kilowatts?$",  re.I), "kW"),
    (re.compile(r"^KW$"),          "kW"),
    # Distance variants
    (re.compile(r"^Km$"),          "km"),
    (re.compile(r"^R\.\s*Km\.?$", re.I), "km"),
    # Currency
    (re.compile(r"^Rs\.?$",       re.I), "Rs."),
    (re.compile(r"^INR$",         re.I), "Rs."),
    # Volume variants → m³
    (re.compile(r"^Cu\.?m$",      re.I), "m³"),
    (re.compile(r"^Cum$",         re.I), "m³"),
    (re.compile(r"^cu\.?m$",      re.I), "m³"),
    # Area variants → m²
    (re.compile(r"^Sq\.?m\.?$",   re.I), "m²"),
    (re.compile(r"^Sqm\.?$",      re.I), "m²"),
    (re.compile(r"^sq\.m$",       re.I), "m²"),
    # Case variants
    (re.compile(r"^Km$"),                "km"),
    (re.compile(r"^Ha$"),                "ha"),
    (re.compile(r"^Nos?\.?$",     re.I), "nos"),
    (re.compile(r"^Hrs?\.?$",     re.I), "hr"),
]

def _canonicalize_unit(unit: str) -> str:
    """Normalize unit spelling/case variants to canonical form."""
    u = unit.strip()
    for pattern, canonical in _UNIT_CANON:
        if pattern.match(u):
            return canonical
    return u

def _normalize_unit_and_value(value_str: str, unit_str: str) -> tuple[str, str]:
    """Normalize value+unit to canonical form. Returns (normalized_value, canonical_unit)."""
    # Clean numeric string: remove commas, "approx", "~", "about"
    v = re.sub(r"[,\s]", "", str(value_str))
    v = re.sub(r"(?i)(approx\.?|approximately|about|~|±\d*)", "", v).strip()

    # Canonicalize unit spelling first
    unit = _canonicalize_unit(unit_str.strip())

    # Try numeric conversion
    try:
        num = float(v)
        for pattern, canonical, multiplier in _UNIT_NORMS:
            if pattern.match(unit):
                num = round(num * multiplier, 6)
                # Remove trailing zeros
                v = f"{num:g}"
                unit = canonical
                break
        else:
            v = f"{num:g}"  # at least normalize the number format
    except (ValueError, TypeError):
        pass  # non-numeric value, keep as-is

    return v, unit


def _normalize_fact(fact: dict) -> dict:
    """Normalize a single fact dict in-place. Returns the fact."""
    # Normalize value + unit
    val, unit = _normalize_unit_and_value(
        fact.get("value", ""),
        fact.get("unit", "")
    )
    fact["value"] = val
    fact["unit"]  = unit

    # Normalize subject and attribute: strip, lower-case attribute, title-case subject
    fact["subject"]   = str(fact.get("subject", "")).strip()
    fact["attribute"] = str(fact.get("attribute", "")).strip().lower()

    # Clamp confidence
    try:
        fact["confidence"] = max(0.0, min(1.0, float(fact.get("confidence", 0.5))))
    except (TypeError, ValueError):
        fact["confidence"] = 0.5

    return fact


# ─── Filter ───────────────────────────────────────────────────────────────────

def filter_facts(facts: list[dict], min_confidence: float) -> tuple[list[dict], dict]:
    """Filter out low-quality facts. Returns (kept, stats)."""
    kept = []
    stats = defaultdict(int)

    for f in facts:
        # Empty value or subject
        if not str(f.get("value", "")).strip():
            stats["empty_value"] += 1
            continue
        if len(str(f.get("subject", "")).strip()) < MIN_SUBJECT_LENGTH:
            stats["empty_subject"] += 1
            continue
        # Low confidence
        try:
            conf = float(f.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0
        if conf < min_confidence:
            stats["low_confidence"] += 1
            continue
        # Value is just punctuation or single char
        if re.match(r"^[\.\-\:\,\s]+$", str(f.get("value", ""))):
            stats["punctuation_only"] += 1
            continue

        kept.append(f)
        stats["kept"] += 1

    return kept, dict(stats)


def _is_null_value(v) -> bool:
    """Check if a table cell value is effectively null/empty/NaN."""
    if v is None:
        return True
    s = str(v).strip()
    return s in ("", "None", "NaN", "nan", "NULL", "null", "-", "N/A", "n/a")


def filter_table_rows(tables: dict) -> tuple[dict, int]:
    """
    Remove table rows that are:
    - Pure headers (all values equal their keys)
    - All-null/NaN rows (carry no engineering information)
    - Rows where >80% of values are null (near-empty rows from merged cells)
    """
    cleaned = {}
    removed = 0
    for page, rows in tables.items():
        good_rows = []
        for row in rows:
            if not row:
                continue
            vals = list(row.values())
            keys = list(row.keys())

            # Skip header rows (all values equal their keys)
            str_vals = [str(v).strip() for v in vals if not _is_null_value(v)]
            str_keys = [str(k).strip() for k in keys]
            if str_vals == str_keys:
                removed += 1
                continue

            # Skip rows where all values are null/NaN
            if all(_is_null_value(v) for v in vals):
                removed += 1
                continue

            # Skip rows where >80% of values are null (merged cell artifacts)
            null_count = sum(1 for v in vals if _is_null_value(v))
            if len(vals) > 0 and null_count / len(vals) > 0.8:
                removed += 1
                continue

            good_rows.append(row)
        if good_rows:
            cleaned[page] = good_rows
    return cleaned, removed


# ─── Dedup ────────────────────────────────────────────────────────────────────

def dedup_facts(facts: list[dict]) -> tuple[list[dict], int]:
    """
    Remove duplicate facts.
    Duplicate = same (doc_id, subject_lower, attribute_lower, value_normalized).
    When duplicates exist, keep highest confidence.
    """
    seen: dict[tuple, dict] = {}
    for f in facts:
        key = (
            f.get("doc_id", ""),
            f.get("subject", "").lower().strip(),
            f.get("attribute", "").lower().strip(),
            f.get("value", "").strip(),
        )
        if key not in seen:
            seen[key] = f
        else:
            # Keep higher confidence
            if float(f.get("confidence", 0)) > float(seen[key].get("confidence", 0)):
                seen[key] = f

    deduped = list(seen.values())
    removed = len(facts) - len(deduped)
    return deduped, removed


# ─── Neo4j push ───────────────────────────────────────────────────────────────

def push_document_node(metadata: dict, dry_run: bool):
    if dry_run:
        return
    run_write(
        f"""
        MERGE (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})
        SET d.filename        = $filename,
            d.doc_type        = $doc_type,
            d.total_pages     = $pages,
            d.sector          = $sector,
            d.sector_confidence = $conf,
            d.is_scanned      = $scanned,
            d.page_range      = $page_range,
            d.source_path     = $src
        WITH d
        MATCH (s:{NodeLabel.SECTOR} {{name: $sector_name}})
        MERGE (d)-[:{RelType.BELONGS_TO}]->(s)
        """,
        {
            "doc_id":    metadata["doc_id"],
            "filename":  metadata.get("filename", ""),
            "doc_type":  metadata.get("doc_type", "pdf"),
            "pages":     metadata.get("total_pages", 0),
            "sector":    metadata.get("sector", ""),
            "conf":      metadata.get("sector_confidence", 0),
            "scanned":   metadata.get("is_scanned", False),
            "page_range": metadata.get("page_range", ""),
            "src":       metadata.get("filename", ""),
            "sector_name": metadata.get("sector", ""),
        }
    )


def push_facts(facts: list[dict], dry_run: bool) -> int:
    if dry_run:
        return len(facts)
    pushed = 0
    for fact in facts:
        fact_id = str(uuid.uuid4())
        run_write(
            f"""
            MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})
            MERGE (f:{NodeLabel.FACT} {{fact_id: $fact_id}})
            SET f.fact_type   = $fact_type,
                f.subject     = $subject,
                f.attribute   = $attribute,
                f.value       = $value,
                f.unit        = $unit,
                f.context     = $context,
                f.confidence  = $confidence,
                f.source_page = $page,
                f.sector      = $sector,
                f.doc_id      = $doc_id
            MERGE (d)-[:{RelType.HAS_FACT}]->(f)
            WITH f
            MATCH (s:{NodeLabel.SECTOR} {{name: $sector_name}})
            MERGE (f)-[:{RelType.BELONGS_TO}]->(s)
            """,
            {
                "doc_id":      fact.get("doc_id", ""),
                "fact_id":     fact_id,
                "fact_type":   fact.get("fact_type", "parameter"),
                "subject":     fact.get("subject", ""),
                "attribute":   fact.get("attribute", ""),
                "value":       str(fact.get("value", "")),
                "unit":        fact.get("unit", ""),
                "context":     str(fact.get("context", ""))[:400],
                "confidence":  float(fact.get("confidence", 0.5)),
                "page":        int(fact.get("source_page", 0)),
                "sector":      fact.get("sector", ""),
                "sector_name": fact.get("sector", ""),
            }
        )
        pushed += 1
    return pushed


def push_table_facts(tables: dict, doc_id: str, sector: str, dry_run: bool) -> int:
    if dry_run:
        return sum(len(v) for v in tables.values())
    pushed = 0
    for page_str, rows in tables.items():
        for i, row in enumerate(rows):
            # Deterministic fact_id based on content — prevents duplicates on re-run
            # Using page + row_index + doc_id means same row always gets same ID
            import hashlib
            row_json = json.dumps(row, sort_keys=True)
            fact_id = "tr_" + hashlib.md5(
                f"{doc_id}_{page_str}_{i}_{row_json}".encode()
            ).hexdigest()[:16]

            run_write(
                f"""
                MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})
                MERGE (f:{NodeLabel.FACT} {{fact_id: $fact_id}})
                SET f.fact_type   = 'table_row',
                    f.subject     = 'table',
                    f.attribute   = 'row_data',
                    f.value       = $row_json,
                    f.unit        = '',
                    f.confidence  = 0.85,
                    f.source_page = $page,
                    f.row_index   = $idx,
                    f.sector      = $sector,
                    f.doc_id      = $doc_id
                MERGE (d)-[:{RelType.HAS_FACT}]->(f)
                """,
                {
                    "doc_id":   doc_id,
                    "fact_id":  fact_id,
                    "row_json": row_json,
                    "page":     int(page_str),
                    "idx":      i,
                    "sector":   sector,
                }
            )
            pushed += 1
    return pushed


def push_rules(rules: list[dict], dry_run: bool) -> int:
    if dry_run:
        return len(rules)
    pushed = 0
    for rule in rules:
        run_write(
            f"""
            MERGE (r:{NodeLabel.RULE} {{
                standard_name: $std,
                clause:        $clause,
                attribute:     $attribute
            }})
            ON CREATE SET r.rule_id = $rule_id
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
                "rule_id":   str(uuid.uuid4()),
                "std":       rule.get("standard_name", ""),
                "clause":    rule.get("clause", ""),
                "attribute": rule.get("attribute", ""),
                "rule_text": str(rule.get("rule_text", ""))[:500],
                "operator":  rule.get("operator", "must_be"),
                "threshold": str(rule.get("threshold", "")),
                "unit":      rule.get("unit", ""),
                "condition": rule.get("condition", ""),
                "severity":  rule.get("severity", "HIGH"),
                "sector":    rule.get("sector", ""),
                "sector_name": rule.get("sector", ""),
            }
        )
        pushed += 1
    return pushed


# ─── Stats table printer ──────────────────────────────────────────────────────

def print_stats(title: str, rows: list[tuple], color: str = "cyan"):
    t = Table(title=title, border_style=color)
    t.add_column("Step", style="bold")
    t.add_column("Count", justify="right")
    for label, count in rows:
        t.add_row(label, str(count))
    console.print(t)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    doc_id = args.doc_id
    rulebook_id = args.rulebook_id

    # Auto-load from state if no explicit doc_id given
    state_file = Path("output/.extraction_state.json")
    rulebook_ids = []  # may have multiple rulebooks

    if (doc_id is None and rulebook_id is None) and state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        doc_id = state.get("dpr", {}).get("doc_id")

        # Handle both "rulebook" (singular, old) and "rulebooks" (list, new)
        if "rulebooks" in state and isinstance(state["rulebooks"], list):
            rulebook_ids = [rb["doc_id"] for rb in state["rulebooks"] if rb.get("doc_id")]
        elif "rulebook" in state and state["rulebook"].get("doc_id"):
            rulebook_ids = [state["rulebook"]["doc_id"]]

        if doc_id:
            console.print(
                f"📂 Auto-loaded: dpr=[cyan]{doc_id}[/cyan], "
                f"rulebooks=[cyan]{len(rulebook_ids)}[/cyan]"
            )
    elif rulebook_id:
        rulebook_ids = [rulebook_id]

    if not doc_id and not rulebook_ids:
        console.print("[red]No document found. Run extraction pipeline first or pass --doc-id explicitly.[/red]")
        sys.exit(1)

    if args.dry_run:
        console.print("[yellow]DRY RUN — no data will be written to Neo4j[/yellow]")
    else:
        console.print("🔌 Connecting to Neo4j...")
        init_schema()

    total_pushed = {"facts": 0, "table_rows": 0, "rules": 0}

    # ── Push ALL rulebook rules
    if rulebook_ids:
        console.rule("[bold]Pushing Rulebooks[/bold]")
        for rb_id in rulebook_ids:
            try:
                rb_data = load_raw(rb_id)
                rules_raw = rb_data["rules"]
                meta = rb_data["metadata"]
                std_name = meta.get("standard_name", rb_id)
                sector_rb = meta.get("sector", "")
                console.print(
                    f"   [{std_name}] {len(rules_raw)} rules "
                    f"(sector: [green]{sector_rb}[/green])"
                )
                if rules_raw and not args.dry_run:
                    push_document_node(meta, args.dry_run)
                    pushed = push_rules(rules_raw, args.dry_run)
                    total_pushed["rules"] += pushed
                    console.print(f"     → Pushed [green]{pushed}[/green] rules")
            except Exception as e:
                console.print(f"   [yellow]Warning: could not push {rb_id}: {e}[/yellow]")

    # ── Push DPR facts
    if doc_id:
        console.rule("[bold]Pushing DPR Facts[/bold]")
        raw = load_raw(doc_id)
        metadata = raw["metadata"]
        sector   = metadata.get("sector", "")
        facts_raw   = raw["facts"]
        tables_raw  = raw["tables"]

        console.print(f"   Sector: [green]{sector}[/green]")
        console.print(f"   Raw facts: {len(facts_raw)}  |  Table pages: {len(tables_raw)}")

        # Step 1: Normalize units only
        # NOTE: We do NOT filter or dedup here.
        # The consistency engine needs ALL occurrences of every fact across all pages
        # to detect cross-section mismatches (e.g. pile diameter = 750mm on p23,
        # 800mm on p67). Dedup before engines would hide these mismatches entirely.
        # Filter and dedup happen in run_validation.py AFTER engines have run.
        facts_norm = [_normalize_fact(f) for f in facts_raw]

        # Filter tables only — remove pure header rows (no data content)
        # This is safe before engines since headers carry no engineering values
        tables_clean, tables_removed = filter_table_rows(tables_raw)

        print_stats("Fact Processing Pipeline", [
            ("Raw extracted",          len(facts_raw)),
            ("After unit normalization", len(facts_norm)),
            ("Table rows raw",         sum(len(v) for v in tables_raw.values())),
            ("Table header rows removed", tables_removed),
            ("Table rows to push",     sum(len(v) for v in tables_clean.values())),
            ("[yellow]Filter/dedup deferred to run_validation.py[/yellow]", 0),
        ])

        if not args.dry_run:
            # Push document node
            push_document_node(metadata, args.dry_run)
            # Push ALL normalized facts — every page occurrence preserved
            pushed_facts = push_facts(facts_norm, args.dry_run)
            pushed_tables = push_table_facts(tables_clean, doc_id, sector, args.dry_run)
            total_pushed["facts"]      = pushed_facts
            total_pushed["table_rows"] = pushed_tables
            console.print(f"\n✅ Pushed to Neo4j: [green]{pushed_facts}[/green] facts "
                          f"(all occurrences preserved for consistency engine), "
                          f"[green]{pushed_tables}[/green] table rows")

            # Save normalized (pre-engine) JSON for inspection
            norm_path = PROCESSED_DIR / doc_id / "facts_normalized.json"
            norm_path.write_text(json.dumps(facts_norm, indent=2, ensure_ascii=False), encoding="utf-8")
            console.print(f"   Normalized facts saved: [cyan]{norm_path}[/cyan]")

            # Ontology generation
            if not args.skip_ontology and len(facts_norm) > 0:
                console.print("\n🧠 Generating ontology...")
                ontology = generate_ontology(doc_id, sector)
                console.print(
                    f"   Ontology: {len(ontology.get('classes', []))} classes, "
                    f"{len(ontology.get('dependencies', []))} dependencies"
                )

    # Update state
    state_file = Path("output/.extraction_state.json")
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    else:
        state = {}
    state["push"] = {"doc_id": doc_id, "rulebook_ids": rulebook_ids, "pushed": total_pushed}
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    console.print(Panel(
        f"Facts pushed:      {total_pushed['facts']}\n"
        f"Table rows pushed: {total_pushed['table_rows']}\n"
        f"Rules pushed:      {total_pushed['rules']}",
        title="[bold green]Push Complete[/bold green]",
        border_style="green",
    ))
    console.print(f"\nNext step: [bold]python run_engines.py --doc-id {doc_id}[/bold]")


if __name__ == "__main__":
    main()