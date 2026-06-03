#!/usr/bin/env python3
"""
run_engines.py
--------------
STEP 2: Consistency + Anomaly engine checks on extracted data in Neo4j.

Usage:
    python run_engines.py --doc-id <id>
    python run_engines.py --doc-id <id> --skip-consistency
    python run_engines.py --doc-id <id> --skip-anomaly
    python run_engines.py --doc-id <id> --sector "Bridges"  # override stored sector

What this does:
    1. Pulls extracted facts from Neo4j for the given document
    2. Runs consistency engine:
        - Numeric cross-section mismatches
        - Missing ontology dependencies
        - LLM holistic review
    3. Runs anomaly detection engine:
        - Statistical outliers (z-score)
        - Order-of-magnitude errors
        - Duplicate value patterns
        - Unit mismatches
        - LLM domain-aware scan
    4. Writes all violations/flags to Neo4j
    5. Saves engine reports to output/

Run after run_extraction.py. Run before run_validation.py.
"""

import sys
import argparse
import json
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import NodeLabel, Severity
from utils.neo4j_client import init_schema, run_read
from engines.consistency_engine import run_consistency_engine
from engines.anomaly_engine import run_anomaly_engine

console = Console()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DPR Validation System — Step 2: Consistency + Anomaly Engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_engines.py --doc-id abc12345
  python run_engines.py --doc-id abc12345 --skip-anomaly
  python run_engines.py --doc-id abc12345 --sector "Bridges"
        """
    )
    parser.add_argument("--doc-id", type=str, help="Document ID from run_extraction.py")
    parser.add_argument("--sector", type=str, help="Override sector (auto-loaded from Neo4j if not provided)")
    parser.add_argument("--skip-consistency", action="store_true", help="Skip consistency engine")
    parser.add_argument("--skip-anomaly", action="store_true", help="Skip anomaly engine")
    parser.add_argument("--from-state", action="store_true",
                        help="Load doc-id and sector from output/.extraction_state.json")
    return parser.parse_args()


# ─── Load doc metadata from Neo4j ────────────────────────────────────────────

def load_doc_metadata(doc_id: str) -> dict:
    """Retrieve stored document metadata from Neo4j."""
    rows = run_read(
        f"""
        MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})
        RETURN d.sector AS sector, d.filename AS filename,
               d.total_pages AS pages, d.sector_confidence AS conf
        """,
        {"doc_id": doc_id}
    )
    if not rows:
        return {}
    return rows[0]


# ─── Print engine results summary ────────────────────────────────────────────

def print_summary_table(title: str, summary: dict, color: str = "cyan"):
    table = Table(title=title, border_style=color, show_header=True)
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right")

    if "by_type" in summary:
        for k, v in summary["by_type"].items():
            table.add_row(k.replace("_", " ").title(), str(v))
        table.add_section()

    if "by_severity" in summary:
        for sev, count in summary["by_severity"].items():
            color_map = {
                Severity.CRITICAL: "red",
                Severity.HIGH: "orange1",
                Severity.MEDIUM: "yellow",
                Severity.LOW: "green",
                Severity.INFO: "dim",
            }
            c = color_map.get(sev, "white")
            table.add_row(f"[{c}]{sev}[/{c}]", f"[{c}]{count}[/{c}]")

    console.print(table)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load from state file if requested
    doc_id = args.doc_id
    sector = args.sector

    if args.from_state or doc_id is None:
        state_file = Path("output/.extraction_state.json")
        if state_file.exists():
            state = json.loads(state_file.read_text(encoding="utf-8"))
            dpr_state = state.get("dpr", {})
            doc_id = doc_id or dpr_state.get("doc_id")
            sector = sector or dpr_state.get("sector")
            console.print(f"📂 Loaded from state: doc_id=[cyan]{doc_id}[/cyan], sector=[green]{sector}[/green]")
        else:
            console.print("[red]No --doc-id provided and no state file found.[/red]")
            sys.exit(1)

    if not doc_id:
        console.print("[red]--doc-id is required[/red]")
        sys.exit(1)

    # Connect to Neo4j
    console.print("🔌 Connecting to Neo4j...")
    init_schema()

    # Load stored metadata
    meta = load_doc_metadata(doc_id)
    if not meta:
        console.print(f"[red]Document '{doc_id}' not found in Neo4j. Run run_extraction.py first.[/red]")
        sys.exit(1)

    if sector is None:
        sector = meta.get("sector", "")
    if not sector:
        console.print("[red]Sector unknown. Provide --sector explicitly.[/red]")
        sys.exit(1)

    console.print(
        f"\n📋 Document: [cyan]{meta.get('filename', doc_id)}[/cyan] | "
        f"Sector: [green]{sector}[/green] | "
        f"Pages: {meta.get('pages', '?')}"
    )

    # Verify facts exist
    fact_count_result = run_read(
        f"MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $id}})-[:HAS_FACT]->(f) "
        f"RETURN count(f) AS cnt",
        {"id": doc_id}
    )
    fact_count = fact_count_result[0]["cnt"] if fact_count_result else 0
    console.print(f"📊 Facts in Neo4j: [bold]{fact_count}[/bold]")

    if fact_count == 0:
        console.print("[yellow]Warning: No facts found. Run run_extraction.py first.[/yellow]")

    engine_results = {"doc_id": doc_id, "sector": sector}

    # ── Consistency engine
    if not args.skip_consistency:
        console.rule("[bold]Consistency Engine[/bold]")
        consistency_summary = run_consistency_engine(doc_id, sector)
        engine_results["consistency"] = consistency_summary
        print_summary_table("Consistency Issues", consistency_summary, "blue")

        # Show top issues
        issues = consistency_summary.get("issues", [])
        if issues:
            console.print(f"\n[bold]Top issues:[/bold]")
            for issue in sorted(issues, key=lambda x: x.get("severity", ""), reverse=True)[:5]:
                sev = issue.get("severity", "")
                sev_color = {"CRITICAL": "red", "HIGH": "orange1", "MEDIUM": "yellow"}.get(sev, "white")
                console.print(
                    f"  [{sev_color}][{sev}][/{sev_color}] "
                    f"{issue.get('description', '')[:100]}"
                )
    else:
        console.print("[dim]Consistency engine: skipped[/dim]")

    # ── Anomaly engine
    if not args.skip_anomaly:
        console.rule("[bold]Anomaly Detection Engine[/bold]")
        anomaly_summary = run_anomaly_engine(doc_id, sector)
        engine_results["anomaly"] = anomaly_summary
        print_summary_table("Anomaly Flags", anomaly_summary, "yellow")

        # Show top anomalies
        flags = anomaly_summary.get("flags", [])
        if flags:
            console.print(f"\n[bold]Top anomalies:[/bold]")
            for flag in sorted(flags, key=lambda x: x.get("severity", ""), reverse=True)[:5]:
                sev = flag.get("severity", "")
                sev_color = {"CRITICAL": "red", "HIGH": "orange1", "MEDIUM": "yellow"}.get(sev, "white")
                console.print(
                    f"  [{sev_color}][{sev}][/{sev_color}] "
                    f"{flag.get('description', '')[:100]}"
                )
    else:
        console.print("[dim]Anomaly engine: skipped[/dim]")

    # Save results
    out_path = Path(f"output/engine_results_{doc_id}.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(engine_results, indent=2, default=str), encoding="utf-8")

    # Update state file
    state_file = Path("output/.extraction_state.json")
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    else:
        state = {}
    state["engines"] = engine_results
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    console.print(Panel(
        f"Consistency: {engine_results.get('consistency', {}).get('total_issues', 'skipped')} issues\n"
        f"Anomalies: {engine_results.get('anomaly', {}).get('total_flags', 'skipped')} flags\n"
        f"Report: output/engine_results_{doc_id}.json",
        title="[bold green]Engines Complete[/bold green]",
        border_style="green",
    ))

    console.print(
        f"\nNext step: [bold]python run_validation.py --doc-id {doc_id}[/bold]"
    )


if __name__ == "__main__":
    main()