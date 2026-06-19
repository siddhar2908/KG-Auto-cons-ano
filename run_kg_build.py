#!/usr/bin/env python3
"""
run_kg_build.py
---------------
STEP 1C: Build Knowledge Graph from rulebook (primary) or DPR (optional).

Architecture:
  The KG is built from the RULEBOOK — not the DPR.
  Rulebook triples represent WHAT IS REQUIRED (structured rule knowledge).
  DPR Fact nodes (from run_push.py) represent WHAT THE DPR CLAIMS.
  Validation = match DPR Fact nodes against the rule KG semantically.

  This means:
    - Rule text → triples like (design speed) → [shall be] → (≥160 km/h per RDSO §X)
    - DPR fact: {attribute: "design speed", value: "130", unit: "km/h"}
    - Semantic matching finds that DPR fact maps to the "design speed" rule triple
    - Value comparison (130 vs ≥160) → Non-Compliant

  FAISS index is built over RULEBOOK triples (not DPR triples).
  The validation engine queries this rule FAISS index for every DPR fact.

Two modes:
  --source rulebook   Build KG from rulebook text (DEFAULT — this is the new flow)
  --source dpr        Build KG from DPR pages (legacy / supplementary analysis)
  --source both       Build both (rulebook first, then DPR)

Usage:
  python run_kg_build.py                          # rulebook mode (default)
  python run_kg_build.py --source rulebook        # explicit rulebook mode
  python run_kg_build.py --source dpr             # legacy DPR mode
  python run_kg_build.py --source both            # both
  python run_kg_build.py --workers 6 --embed-workers 4
  python run_kg_build.py --skip-concepts --skip-faiss   # fastest
  python run_kg_build.py --force-rebuild                # full reset
"""

import sys
import argparse
import json
import threading
import time
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    MofNCompleteColumn, TimeElapsedColumn, TaskProgressColumn,
)
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import PROCESSED_DIR, NodeLabel, RULEBOOKS_INPUT_DIR, DPR_INPUT_DIR
from utils.neo4j_client import init_schema, run_read, run_write
from extractors.document_loader import load_document
from extractors.kg_builder import (
    build_kg_from_page, get_kg_stats,
    clear_page_cache, _load_page_cache,
)
from extractors.kg_embeddings import build_kg_index

console = Console()
_progress_lock = threading.Lock()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DPR Validation — Step 1C: KG Build (rulebook-first architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Source modes:
  --source rulebook   Build KG from rulebook (structured rule knowledge) [DEFAULT]
  --source dpr        Build KG from DPR pages (legacy supplementary mode)
  --source both       Build both

Speed flags:
  --skip-concepts     No concept induction (faster, no schema layer)
  --skip-faiss        Skip FAISS index build (save for later)
  --workers N         Parallel page workers (default 4)
  --embed-workers N   Parallel embedding batches (default 4)
  --force-rebuild     Clear cache + Neo4j KG data and rebuild
        """
    )
    parser.add_argument("--doc-id",        type=str, help="DPR document ID")
    parser.add_argument("--rulebook-id",   type=str, help="Rulebook doc ID (overrides state file)")
    parser.add_argument("--from-state",    action="store_true")
    parser.add_argument("--source",        type=str, default="rulebook",
                        choices=["rulebook", "dpr", "both"],
                        help="Which document to build KG from (default: rulebook)")
    parser.add_argument("--workers",       type=int, default=4)
    parser.add_argument("--embed-workers", type=int, default=4)
    parser.add_argument("--skip-concepts", action="store_true")
    parser.add_argument("--skip-faiss",    action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-cache",      action="store_true")
    return parser.parse_args()


# ─── State loader ─────────────────────────────────────────────────────────────

def load_state() -> dict:
    state_file = Path("output/.extraction_state.json")
    if state_file.exists():
        return json.loads(state_file.read_text(encoding="utf-8"))
    return {}


def save_state(updates: dict):
    state_file = Path("output/.extraction_state.json")
    state = load_state()
    state.update(updates)
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ─── Metadata loader ──────────────────────────────────────────────────────────

def load_metadata(doc_id: str) -> dict:
    meta_path = PROCESSED_DIR / doc_id / "metadata.json"
    if not meta_path.exists():
        console.print(f"[red]No metadata for {doc_id}. Run run_extraction.py first.[/red]")
        sys.exit(1)
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ─── Document path resolver ───────────────────────────────────────────────────

def find_doc_file(filename: str, is_rulebook: bool) -> Path:
    """Locate the source file on disk for a given doc ID."""
    search_dirs = [
        RULEBOOKS_INPUT_DIR if is_rulebook else DPR_INPUT_DIR,
        Path("data/uploads"),
        Path("data"),
        Path("."),
    ]
    for d in search_dirs:
        p = d / filename
        if p.exists():
            return p
    return None


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _split_cached_pages(pages: list, doc_id: str, use_cache: bool) -> tuple[list, list]:
    if not use_cache:
        return [], pages
    cached, uncached = [], []
    for page in pages:
        if _load_page_cache(doc_id, page.page_num) is not None:
            cached.append(page)
        else:
            uncached.append(page)
    return cached, uncached


# ─── Per-page KG build worker ─────────────────────────────────────────────────

def _kg_build_page(page, doc_id: str, sector: str, induce_schema: bool, use_cache: bool) -> dict:
    try:
        result = build_kg_from_page(
            text=page.text,
            doc_id=doc_id,
            sector=sector,
            page_num=page.page_num,
            write_to_db=True,
            induce_schema=induce_schema,
            use_cache=use_cache,
        )
        triples_data = [
            {
                "head":              t.head,
                "relation":          t.relation,
                "tail":              t.tail,
                "triple_type":       t.triple_type,
                "head_concepts":     t.head_concepts,
                "relation_concepts": t.relation_concepts,
                "tail_concepts":     t.tail_concepts,
                "source_page":       page.page_num + 1,
                "doc_id":            doc_id,
                "sector":            sector,
            }
            for t in result.triples
        ]
        return {
            "page_num":     page.page_num,
            "triples":      len(result.triples),
            "entities":     len(result.all_entities),
            "concepts":     len(result.concept_map),
            "error":        None,
            "triples_data": triples_data,
        }
    except Exception as e:
        logger.warning(f"KG build failed on page {page.page_num + 1}: {e}")
        return {
            "page_num": page.page_num, "triples": 0, "entities": 0,
            "concepts": 0, "error": str(e), "triples_data": [],
        }


# ─── Speed summary panel ──────────────────────────────────────────────────────

def _print_speed_summary(label: str, t_elapsed: float, total_pages: int,
                         cached_count: int, total_triples: int, total_entities: int, errors: int):
    pages_per_min = round(total_pages / (t_elapsed / 60), 1) if t_elapsed > 0 else 0
    from rich.table import Table
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("", style="dim")
    table.add_column("", style="bold")
    table.add_row("Source",             label)
    table.add_row("Pages processed",    str(total_pages))
    table.add_row("Pages from cache",   f"{cached_count} (no Ollama call)")
    table.add_row("Pages via Ollama",   str(total_pages - cached_count))
    table.add_row("Triples extracted",  str(total_triples))
    table.add_row("Entities extracted", str(total_entities))
    table.add_row("Errors",             str(errors))
    table.add_row("Elapsed",            f"{t_elapsed:.1f}s")
    table.add_row("Throughput",         f"{pages_per_min} pages/min")
    console.print(Panel(table, title=f"[bold green]KG Build Complete — {label}[/bold green]", border_style="green"))


# ─── Core build function ──────────────────────────────────────────────────────

def build_kg_for_doc(
    doc_id: str,
    filename: str,
    sector: str,
    is_rulebook: bool,
    args,
) -> dict:
    """
    Build KG triples + concepts for one document (rulebook or DPR).
    Returns stats dict.
    """
    label = "Rulebook" if is_rulebook else "DPR"
    use_cache = not args.no_cache

    # Force rebuild: clear cache + KG data for this doc_id
    if args.force_rebuild:
        console.print(f"[yellow]⚠ Force rebuild: clearing KG data for {label} ({doc_id})...[/yellow]")
        clear_page_cache(doc_id)
        run_write("MATCH (e:Entity {doc_id: $id}) DETACH DELETE e", {"id": doc_id})
        run_write("MATCH (o:OntologyClass {doc_id: $id}) DETACH DELETE o", {"id": doc_id})

    # Check if already built
    if not args.force_rebuild:
        stats = get_kg_stats(doc_id)
        if stats["triples"] > 0:
            console.print(
                f"[yellow]{label} KG already exists ({stats['triples']} triples, "
                f"{stats['entities']} entities). Use --force-rebuild to rebuild.[/yellow]"
            )
            return stats

    # Find source file
    doc_path = find_doc_file(filename, is_rulebook)
    if doc_path is None:
        console.print(f"[red]Cannot find {label} file: {filename}[/red]")
        console.print(f"  Expected in: {'data/rulebooks/' if is_rulebook else 'data/dpr/'}")
        return {"triples": 0, "entities": 0, "concepts": 0}

    console.print(f"📄 Loading {label}: [cyan]{doc_path.name}[/cyan]")
    doc = load_document(doc_path, doc_id)

    # Filter pages: use processed_pages from metadata
    metadata = load_metadata(doc_id)
    processed_pages_set = set(metadata.get("processed_pages", []))

    pages_to_build = [
        p for p in doc.pages
        if (not processed_pages_set or (p.page_num + 1) in processed_pages_set)
        and p.text and len(p.text.strip()) > 50
    ]

    if not pages_to_build:
        console.print(f"[yellow]No processable pages found for {label}.[/yellow]")
        return {"triples": 0, "entities": 0, "concepts": 0}

    # Split cached vs uncached
    cached_pages, uncached_pages = _split_cached_pages(pages_to_build, doc_id, use_cache)

    console.print(
        f"\n   Total pages:      [bold]{len(pages_to_build)}[/bold]\n"
        f"   Cached (skip):    [green]{len(cached_pages)}[/green]\n"
        f"   Need Ollama:      [cyan]{len(uncached_pages)}[/cyan]\n"
        f"   Workers:          [cyan]{args.workers}[/cyan]\n"
        f"   Concept induction:{'[dim]disabled[/dim]' if args.skip_concepts else '[green]batched[/green]'}"
    )

    # Parallel KG build
    total_triples, total_entities, total_concepts, errors = 0, 0, 0, 0
    all_triples_data = []
    t_start = time.time()

    console.rule(f"[bold]Triple Extraction — {label}[/bold]")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Building {label} KG...", total=len(pages_to_build))

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _kg_build_page, page, doc_id, sector,
                    not args.skip_concepts, use_cache,
                ): page.page_num
                for page in pages_to_build
            }
            for future in as_completed(futures):
                result = future.result()
                with _progress_lock:
                    progress.advance(task)
                    progress.update(task, description=f"p{result['page_num']+1} {result['triples']}t")
                total_triples  += result["triples"]
                total_entities += result["entities"]
                total_concepts += result["concepts"]
                all_triples_data.extend(result.get("triples_data", []))
                if result["error"]:
                    errors += 1

    t_elapsed = time.time() - t_start

    # Save triples JSON
    triples_path = PROCESSED_DIR / doc_id / "triples_raw.json"
    triples_path.write_text(
        json.dumps(all_triples_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    stats = get_kg_stats(doc_id)
    _print_speed_summary(label, t_elapsed, len(pages_to_build), len(cached_pages),
                         stats["triples"], stats["entities"], errors)
    return stats


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    state = load_state()
    doc_id      = args.doc_id
    rulebook_id = args.rulebook_id

    # Auto-load from state
    if not doc_id or not rulebook_id:
        dpr_state  = state.get("dpr", {})
        rb_entries = state.get("rulebooks", [])
        if not doc_id:
            doc_id = dpr_state.get("doc_id")
        if not rulebook_id and rb_entries:
            rulebook_id = rb_entries[0].get("doc_id")
        if doc_id or rulebook_id:
            console.print(
                f"📂 Auto-loaded: dpr=[cyan]{doc_id}[/cyan]  "
                f"rulebook=[cyan]{rulebook_id}[/cyan]"
            )

    if not doc_id and not rulebook_id:
        console.print("[red]No document found. Run run_extraction.py first.[/red]")
        sys.exit(1)

    console.print("🔌 Connecting to Neo4j...")
    init_schema()

    source = args.source
    kg_stats = {}

    # ── RULEBOOK KG build (default / primary) ─────────────────────────────────
    if source in ("rulebook", "both"):
        if not rulebook_id:
            console.print("[red]No rulebook doc_id found. Run run_extraction.py --rulebook first.[/red]")
            if source == "rulebook":
                sys.exit(1)
        else:
            rb_meta  = load_metadata(rulebook_id)
            rb_file  = rb_meta.get("filename", "")
            rb_sector = rb_meta.get("sector", "Rail Infrastructure")

            console.rule("[bold cyan]Building KG from Rulebook[/bold cyan]")
            console.print(
                "This builds structured triples from the rulebook text.\n"
                "These triples represent WHAT IS REQUIRED — the rule knowledge graph.\n"
                "DPR facts will be validated against this graph.\n"
            )

            rb_stats = build_kg_for_doc(
                doc_id=rulebook_id,
                filename=rb_file,
                sector=rb_sector,
                is_rulebook=True,
                args=args,
            )
            kg_stats["rulebook"] = rb_stats

            # Build FAISS index over rulebook KG (this is what validation searches)
            if not args.skip_faiss:
                console.rule("[bold]FAISS Index — Rulebook Rule Graph[/bold]")
                console.print(
                    f"Building semantic index over {rb_stats.get('entities', 0)} rule entities "
                    f"and {rb_stats.get('triples', 0)} rule triples...\n"
                    "The validation engine will query this index to match DPR facts to rules."
                )
                idx_stats = build_kg_index(
                    rulebook_id,
                    force_rebuild=True,
                    embed_workers=args.embed_workers,
                )
                console.print(
                    f"   Rule node index: [green]{idx_stats['nodes']}[/green] vectors\n"
                    f"   Rule edge index: [green]{idx_stats['edges']}[/green] vectors\n"
                    f"   Saved to: [cyan]{PROCESSED_DIR / rulebook_id / 'faiss'}[/cyan]"
                )
                kg_stats["rulebook"]["faiss"] = idx_stats

    # ── DPR KG build (optional supplementary) ─────────────────────────────────
    if source in ("dpr", "both"):
        if not doc_id:
            console.print("[red]No DPR doc_id found. Pass --doc-id explicitly.[/red]")
            if source == "dpr":
                sys.exit(1)
        else:
            dpr_meta   = load_metadata(doc_id)
            dpr_file   = dpr_meta.get("filename", "")
            dpr_sector = dpr_meta.get("sector", "Rail Infrastructure")

            console.rule("[bold]Building KG from DPR (supplementary)[/bold]")
            console.print(
                "This builds triples from the DPR text.\n"
                "These supplement fact-level validation with structural/contextual relationships.\n"
            )

            dpr_stats = build_kg_for_doc(
                doc_id=doc_id,
                filename=dpr_file,
                sector=dpr_sector,
                is_rulebook=False,
                args=args,
            )
            kg_stats["dpr"] = dpr_stats

            # DPR FAISS index (optional — used for semantic completeness check)
            if not args.skip_faiss:
                console.rule("[bold]FAISS Index — DPR[/bold]")
                idx_stats = build_kg_index(
                    doc_id,
                    force_rebuild=True,
                    embed_workers=args.embed_workers,
                )
                console.print(
                    f"   DPR node index: [green]{idx_stats['nodes']}[/green] vectors\n"
                    f"   DPR edge index: [green]{idx_stats['edges']}[/green] vectors"
                )
                kg_stats["dpr"]["faiss"] = idx_stats

    # ── Update state file ──────────────────────────────────────────────────────
    save_state({
        "kg_build": {
            "source":      source,
            "doc_id":      doc_id,
            "rulebook_id": rulebook_id,
            "stats":       kg_stats,
            "faiss_built": not args.skip_faiss,
        }
    })

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    if source in ("rulebook", "both") and "rulebook" in kg_stats:
        rbs = kg_stats["rulebook"]
        console.print(
            f"[bold]Rule KG:[/bold] {rbs.get('triples',0)} triples, "
            f"{rbs.get('entities',0)} entities  "
            f"[dim](doc_id: {rulebook_id})[/dim]"
        )
    if source in ("dpr", "both") and "dpr" in kg_stats:
        ds = kg_stats["dpr"]
        console.print(
            f"[bold]DPR  KG:[/bold] {ds.get('triples',0)} triples, "
            f"{ds.get('entities',0)} entities  "
            f"[dim](doc_id: {doc_id})[/dim]"
        )

    console.print(f"\nNext step: [bold]python run_engines.py --doc-id {doc_id}[/bold]")


if __name__ == "__main__":
    main()