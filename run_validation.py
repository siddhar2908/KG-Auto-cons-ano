#!/usr/bin/env python3
"""
run_validation.py
-----------------
STEP 3: DPR validation — rulebook compliance check.

Usage:
    python run_validation.py --doc-id <id>
    python run_validation.py --doc-id <id> --sector "Bridges"
    python run_validation.py --doc-id <id> --print-report
    python run_validation.py --from-state

What this does:
    1. Matches DPR fact nodes against Rule nodes in Neo4j (same sector, same attribute)
    2. Compares values using operator/threshold rules (>=, <=, ==, in_range, must_be)
    3. Sends Non-Compliant findings to Ollama for plain-English explanation
    4. Checks completeness: mandatory parameters from ontology must be present
    5. Computes a weighted compliance score (CRITICAL rules worth 4×, HIGH 3×, etc.)
    6. Generates a Compliant / Non-Compliant report (JSON + console summary)
    7. Writes Violation nodes to Neo4j

Output:
    output/validation_report_<doc_id>.json
    Console: scored summary with verdict band (GOOD / SATISFACTORY / NEEDS IMPROVEMENT / POOR)

Run after run_engines.py (or directly after run_extraction.py if engines not needed).
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
        description="DPR Validation System — Step 3: Rulebook Compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_validation.py --doc-id abc12345
  python run_validation.py --from-state
  python run_validation.py --doc-id abc12345 --print-report
        """
    )
    parser.add_argument("--doc-id",       type=str, help="Document ID")
    parser.add_argument("--sector",       type=str, help="Override sector")
    parser.add_argument("--print-report", action="store_true",
                        help="Print full Compliant/Non-Compliant table to console")
    parser.add_argument("--from-state",   action="store_true",
                        help="Load doc-id and sector from output/.extraction_state.json")
    return parser.parse_args()


# ─── Report printer ───────────────────────────────────────────────────────────

def _verdict_color(verdict: str) -> str:
    return {
        "GOOD":               "bold green",
        "SATISFACTORY":       "green",
        "NEEDS IMPROVEMENT":  "yellow",
        "POOR":               "bold red",
        "NO_CHECKS_RUN":      "dim",
    }.get(verdict, "white")


def print_report(report: dict):
    console.print("\n")
    score   = report.get("score", {})
    verdict = score.get("verdict", "UNKNOWN")
    wscore  = score.get("weighted_score", 0.0)
    sector  = report.get("sector", "")
    vc      = _verdict_color(verdict)

    # ── Verdict banner
    console.print(Panel(
        f"[{vc}]{verdict}[/{vc}]   "
        f"Weighted Compliance Score: [{vc}]{wscore:.1f}%[/{vc}]",
        title=f"Validation Result — {sector} DPR",
        border_style=vc.replace("bold ", ""),
        subtitle=(
            f"[dim]Compliant: {score.get('compliant_count',0)}  |  "
            f"Non-Compliant: {score.get('non_compliant_count',0)}  |  "
            f"Total checks: {score.get('total_checks',0)}[/dim]"
        ),
    ))

    # ── Score breakdown by severity
    by_sev = score.get("by_severity", {})
    if by_sev:
        sev_table = Table(title="Score Breakdown by Severity", border_style="cyan", show_lines=False)
        sev_table.add_column("Severity",       style="bold", min_width=10)
        sev_table.add_column("Total Checks",   justify="right", min_width=12)
        sev_table.add_column("Compliant",      justify="right", min_width=10)
        sev_table.add_column("Non-Compliant",  justify="right", min_width=14)
        sev_table.add_column("Weight/Check",   justify="right", min_width=12)

        sev_color_map = {
            Severity.CRITICAL: "red",
            Severity.HIGH:     "orange1",
            Severity.MEDIUM:   "yellow",
            Severity.LOW:      "green",
        }
        for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW):
            if sev not in by_sev:
                continue
            d   = by_sev[sev]
            sc  = sev_color_map.get(sev, "white")
            nc  = d.get("non_compliant", 0)
            co  = d.get("compliant", 0)
            wt  = d.get("weight", 0) // max(d.get("total", 1), 1)
            sev_table.add_row(
                f"[{sc}]{sev}[/{sc}]",
                str(d.get("total", 0)),
                f"[green]{co}[/green]",
                f"[red]{nc}[/red]" if nc else "0",
                f"{wt}",
            )
        console.print(sev_table)

    # ── Score breakdown by category
    by_cat = score.get("by_category", {})
    if by_cat:
        cat_table = Table(title="Score by Category (Rulebook Section)", border_style="blue", show_lines=False)
        cat_table.add_column("Category",       style="bold", min_width=30)
        cat_table.add_column("Checks",         justify="right", min_width=8)
        cat_table.add_column("Compliant",      justify="right", min_width=10)
        cat_table.add_column("Non-Compliant",  justify="right", min_width=14)
        cat_table.add_column("Category Score", justify="right", min_width=14)

        sorted_cats = sorted(by_cat.items(), key=lambda x: x[1].get("category_score", 100))
        for cat, d in sorted_cats:
            cat_score = d.get("category_score", 0.0)
            score_color = "green" if cat_score >= 75 else "yellow" if cat_score >= 50 else "red"
            cat_table.add_row(
                cat[:40],
                str(d.get("total", 0)),
                f"[green]{d.get('compliant', 0)}[/green]",
                f"[red]{d.get('non_compliant', 0)}[/red]" if d.get("non_compliant", 0) else "0",
                f"[{score_color}]{cat_score:.1f}%[/{score_color}]",
            )
        console.print(cat_table)

    # ── Full results table (only if --print-report)
    results = report.get("results", [])
    if results:
        non_comp = [r for r in results if r["classification"] == "Non-Compliant"]
        comp     = [r for r in results if r["classification"] == "Compliant"]

        if non_comp:
            console.print(f"\n[bold red]Non-Compliant Checks ({len(non_comp)}):[/bold red]")
            _print_results_table(non_comp, border_color="red")

        if comp:
            console.print(f"\n[bold green]Compliant Checks ({len(comp)}):[/bold green]")
            _print_results_table(comp[:20], border_color="green")  # cap at 20 for readability
            if len(comp) > 20:
                console.print(f"[dim]  ... and {len(comp) - 20} more compliant checks.[/dim]")


def _print_results_table(rows: list[dict], border_color: str):
    t = Table(border_style=border_color, show_lines=True)
    t.add_column("Classification",   min_width=14,  style="bold")
    t.add_column("Check Area",       min_width=22)
    t.add_column("Category",         min_width=18)
    t.add_column("DPR Value",        min_width=14)
    t.add_column("Rule Requires",    min_width=18)
    t.add_column("Severity",         min_width=9)
    t.add_column("Score Weight",     min_width=10, justify="right")
    t.add_column("Standard",         min_width=16)
    t.add_column("Page",             min_width=5, justify="right")
    t.add_column("Reason",           min_width=40)

    sev_color_map = {
        Severity.CRITICAL: "red",
        Severity.HIGH:     "orange1",
        Severity.MEDIUM:   "yellow",
        Severity.LOW:      "green",
    }

    for r in rows:
        cls        = r["classification"]
        cls_color  = "green" if cls == "Compliant" else "red"
        sev        = r.get("severity", "")
        sc         = sev_color_map.get(sev, "white")
        pg         = str(r.get("source_page", "")) or "—"

        t.add_row(
            f"[{cls_color}]{cls}[/{cls_color}]",
            r.get("check_area", "")[:30],
            r.get("category",   "")[:25],
            r.get("dpr_value",  "")[:18],
            r.get("rule_expected", "")[:22],
            f"[{sc}]{sev}[/{sc}]",
            str(r.get("weight", "")),
            r.get("standard", "")[:20],
            pg,
            r.get("reason", "")[:80],
        )

    console.print(t)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    doc_id = args.doc_id
    sector = args.sector

    # Auto-load from state
    state_file = Path("output/.extraction_state.json")
    if (doc_id is None or args.from_state) and state_file.exists():
        state    = json.loads(state_file.read_text(encoding="utf-8"))
        dpr_state = state.get("dpr", {})
        doc_id   = doc_id or dpr_state.get("doc_id")
        sector   = sector or dpr_state.get("sector")
        if doc_id:
            console.print(
                f"📂 Auto-loaded: doc_id=[cyan]{doc_id}[/cyan], "
                f"sector=[green]{sector}[/green]"
            )

    if not doc_id:
        console.print(
            "[red]No document found. "
            "Run the full pipeline first or pass --doc-id explicitly.[/red]"
        )
        sys.exit(1)

    # Connect to Neo4j
    console.print("🔌 Connecting to Neo4j...")
    init_schema()

    # Resolve sector from Neo4j if not provided
    if not sector:
        meta = run_read(
            f"MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $id}}) RETURN d.sector AS sector",
            {"id": doc_id}
        )
        sector = meta[0]["sector"] if meta else None

    if not sector:
        console.print("[red]Sector unknown. Use --sector.[/red]")
        sys.exit(1)

    # Check rules are loaded
    rule_count_result = run_read(
        f"MATCH (r:{NodeLabel.RULE})-[:BELONGS_TO]->(:Sector {{name: $sector}}) "
        f"RETURN count(r) AS cnt",
        {"sector": sector}
    )
    rule_count = rule_count_result[0]["cnt"] if rule_count_result else 0
    console.print(f"📚 Rules in Neo4j for {sector}: [bold]{rule_count}[/bold]")

    if rule_count == 0:
        console.print(
            "[yellow]⚠ No rules found for this sector. "
            "Load rulebooks first with run_extraction.py --rulebook.[/yellow]"
        )

    console.rule("[bold]Validation Engine — Rulebook Compliance[/bold]")
    console.print(f"🔍 Validating doc=[cyan]{doc_id}[/cyan] against {rule_count} rules...")

    report = run_validation(
        doc_id=doc_id,
        sector=sector,
        save_report=True,
    )

    # Always print score summary + category table
    print_report(report)

    # Print full results table only if --print-report
    if not args.print_report:
        # Show just the top 5 non-compliant checks as a quick preview
        non_comp = [r for r in report.get("results", []) if r["classification"] == "Non-Compliant"]
        if non_comp:
            console.print(f"\n[bold]Top Non-Compliant Checks (preview):[/bold]")
            for r in non_comp[:5]:
                sev = r.get("severity", "")
                sc  = {"CRITICAL": "red", "HIGH": "orange1", "MEDIUM": "yellow"}.get(sev, "white")
                console.print(
                    f"  [{sc}][{sev}][/{sc}] "
                    f"[bold]{r.get('check_area', '')}[/bold] — "
                    f"{r.get('reason', '')[:120]}"
                )
            if len(non_comp) > 5:
                console.print(
                    f"  [dim]... and {len(non_comp) - 5} more. "
                    f"Use --print-report to see all.[/dim]"
                )

    out_path = Path(f"output/validation_report_{doc_id}.json")
    console.print(
        f"\n✅ [bold green]Validation complete.[/bold green] "
        f"Score: [bold]{report['score']['weighted_score']:.1f}%[/bold] "
        f"([bold]{report['score']['verdict']}[/bold])   "
        f"Report: [cyan]{out_path}[/cyan]"
    )


if __name__ == "__main__":
    main()