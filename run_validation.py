#!/usr/bin/env python3
"""
run_validation.py
-----------------
STEP 3: DPR validation — KG-based fact vs rule matching + LLM reasoning.

Usage:
    python run_validation.py --doc-id <id>
    python run_validation.py --doc-id <id> --sector "Bridges"
    python run_validation.py --doc-id <id> --print-report
    python run_validation.py --from-state  (auto-loads from output/.extraction_state.json)

What this does:
    1. Matches DPR fact nodes against Rule nodes in Neo4j (same sector, same attribute)
    2. Applies operator/threshold comparison (>=, <=, ==, in_range, must_be)
    3. Sends FAIL and WARNING findings to Ollama for natural-language explanation
    4. Checks completeness: mandatory parameters from ontology must be present
    5. Pulls consistency + anomaly summaries from previous step
    6. Generates final ValidationReport (JSON + console summary)
    7. Writes Violation nodes to Neo4j with VIOLATES / COMPLIES_WITH edges

Output:
    output/validation_report_<doc_id>.json
    Console: scored summary with PASS/FAIL verdict

Run after run_engines.py.
"""

import sys
import argparse
import json
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import NodeLabel, Severity
from utils.neo4j_client import init_schema, run_read
from validators.validation_engine import run_validation

console = Console()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DPR Validation System — Step 3: Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_validation.py --doc-id abc12345
  python run_validation.py --from-state
  python run_validation.py --doc-id abc12345 --print-report
        """
    )
    parser.add_argument("--doc-id", type=str, help="Document ID")
    parser.add_argument("--sector", type=str, help="Override sector")
    parser.add_argument("--print-report", action="store_true", help="Print full findings to console")
    parser.add_argument("--from-state", action="store_true",
                        help="Load doc-id and sector from output/.extraction_state.json")
    return parser.parse_args()


# ─── Report printer ───────────────────────────────────────────────────────────

def print_report(report: dict):
    console.print("\n")

    # Verdict banner
    verdict = report.get("verdict", "UNKNOWN")
    score = report.get("overall_score", 0)
    verdict_color = {
        "PASS": "bold green",
        "MINOR_ISSUES": "yellow",
        "MAJOR_ISSUES": "orange1",
        "CRITICAL_ISSUES": "bold red",
    }.get(verdict, "white")

    console.print(Panel(
        f"[{verdict_color}]{verdict}[/{verdict_color}]   "
        f"Score: [{verdict_color}]{score:.1f}%[/{verdict_color}]",
        title=f"Validation Result — {report.get('sector', '')} DPR",
        border_style=verdict_color.replace("bold ", ""),
    ))

    # Summary table
    summary = report.get("summary", {})
    t = Table(title="Validation Summary", border_style="cyan")
    t.add_column("Check", style="bold")
    t.add_column("Count", justify="right")
    t.add_row("[green]PASS[/green]",    f"[green]{summary.get('pass', 0)}[/green]")
    t.add_row("[red]FAIL[/red]",        f"[red]{summary.get('fail', 0)}[/red]")
    t.add_row("[yellow]WARNING[/yellow]", f"[yellow]{summary.get('warning', 0)}[/yellow]")
    t.add_row("[orange1]MISSING[/orange1]", f"[orange1]{summary.get('missing', 0)}[/orange1]")
    console.print(t)

    # Consistency + Anomaly
    cons = report.get("consistency", {})
    anom = report.get("anomalies", {})
    if cons or anom:
        t2 = Table(title="Engine Results", border_style="blue")
        t2.add_column("Engine")
        t2.add_column("Issues", justify="right")
        t2.add_column("Critical", justify="right")
        t2.add_column("High", justify="right")
        if cons:
            sev = cons.get("by_severity", {})
            t2.add_row(
                "Consistency",
                str(cons.get("total_issues", 0)),
                f"[red]{sev.get('CRITICAL', 0)}[/red]",
                f"[orange1]{sev.get('HIGH', 0)}[/orange1]",
            )
        if anom:
            sev = anom.get("by_severity", {})
            t2.add_row(
                "Anomaly Detection",
                str(anom.get("total_flags", 0)),
                f"[red]{sev.get('CRITICAL', 0)}[/red]",
                f"[orange1]{sev.get('HIGH', 0)}[/orange1]",
            )
        console.print(t2)

    # Failures
    findings = report.get("findings", {})
    failures = findings.get("failures", [])
    if failures:
        console.print(f"\n[bold red]❌ Validation Failures ({len(failures)}):[/bold red]")
        fail_table = Table(border_style="red", show_lines=True)
        fail_table.add_column("Attribute", style="bold", min_width=20)
        fail_table.add_column("DPR Value", min_width=12)
        fail_table.add_column("Rule", min_width=16)
        fail_table.add_column("Standard", min_width=12)
        fail_table.add_column("Severity", min_width=8)
        fail_table.add_column("Explanation", min_width=30)

        for f in failures[:20]:  # cap at 20 for readability
            sev = f.get("severity", "")
            sev_color = {"CRITICAL": "red", "HIGH": "orange1", "MEDIUM": "yellow", "LOW": "green"}.get(sev, "white")
            fail_table.add_row(
                f.get("attribute", ""),
                f.get("dpr_value", ""),
                f.get("rule", ""),
                f"{f.get('standard', '')} §{f.get('clause', '')}",
                f"[{sev_color}]{sev}[/{sev_color}]",
                f.get("explanation", "")[:80],
            )
        console.print(fail_table)

    # Missing parameters
    missing = findings.get("missing_parameters", [])
    if missing:
        console.print(f"\n[bold orange1]⚠ Missing Mandatory Parameters ({len(missing)}):[/bold orange1]")
        for m in missing[:10]:
            console.print(
                f"  • [yellow]{m.get('parameter', '')}[/yellow] "
                f"({m.get('entity', '')}): {m.get('rule_text', '')[:80]}"
            )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    doc_id = args.doc_id
    sector = args.sector
    consistency_summary = {}
    anomaly_summary = {}

    # Auto-load from state if no doc_id given
    state_file = Path("output/.extraction_state.json")
    if doc_id is None and state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        dpr_state = state.get("dpr", {})
        doc_id = dpr_state.get("doc_id")
        sector = sector or dpr_state.get("sector")
        eng = state.get("engines", {})
        consistency_summary = eng.get("consistency", {})
        anomaly_summary     = eng.get("anomaly", {})
        if doc_id:
            console.print(f"📂 Auto-loaded: doc_id=[cyan]{doc_id}[/cyan], sector=[green]{sector}[/green]")

    if not doc_id:
        console.print("[red]No document found. Run the full pipeline first or pass --doc-id explicitly.[/red]")
        sys.exit(1)

    # Connect
    console.print("🔌 Connecting to Neo4j...")
    init_schema()

    # Load metadata if sector not set
    if not sector:
        meta = run_read(
            f"MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $id}}) RETURN d.sector AS sector",
            {"id": doc_id}
        )
        sector = meta[0]["sector"] if meta else None

    if not sector:
        console.print("[red]Sector unknown. Use --sector.[/red]")
        sys.exit(1)

    # Check rules exist
    rule_count_result = run_read(
        f"MATCH (r:{NodeLabel.RULE})-[:BELONGS_TO]->(:Sector {{name: $sector}}) RETURN count(r) AS cnt",
        {"sector": sector}
    )
    rule_count = rule_count_result[0]["cnt"] if rule_count_result else 0
    console.print(f"📚 Rules in Neo4j for {sector}: [bold]{rule_count}[/bold]")

    if rule_count == 0:
            "[yellow]Warning: No rules found for this sector. "
            "Load rulebooks first with run_extraction.py --rulebook.[/yellow]"

    # Load engine summaries from file if not in state
    engine_file = Path(f"output/engine_results_{doc_id}.json")
    if (not consistency_summary or not anomaly_summary) and engine_file.exists():
        eng = json.loads(engine_file.read_text(encoding="utf-8"))
        consistency_summary = consistency_summary or eng.get("consistency", {})
        anomaly_summary = anomaly_summary or eng.get("anomaly", {})

    console.rule("[bold]Validation Engine[/bold]")
    console.print(f"🔍 Validating doc=[cyan]{doc_id}[/cyan] against {rule_count} rules...")

    report = run_validation(
        doc_id=doc_id,
        sector=sector,
        consistency_summary=consistency_summary,
        anomaly_summary=anomaly_summary,
        save_report=True,
    )

    # Always print summary
    print_report(report)

    # Print full findings if requested
    if args.print_report:
        full_path = Path(f"output/validation_report_{doc_id}.json")
        console.print(f"\n[dim]Full report: {full_path}[/dim]")

    console.print(
        f"\n✅ [bold green]Validation complete.[/bold green] "
        f"Report: [cyan]output/validation_report_{doc_id}.json[/cyan]"
    )


if __name__ == "__main__":
    main()