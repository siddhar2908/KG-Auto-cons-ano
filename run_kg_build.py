#!/usr/bin/env python3
"""
run_kg_build.py
---------------
STEP 1C: Build Knowledge Graph from extracted text.
Runs AFTER run_push.py, BEFORE run_engines.py.

What this does:
  1. Reloads page text from the original PDF (already parsed — no re-OCR)
  2. Runs combined triple extraction per page in parallel (entity-relation +
     event-entity in ONE LLM call, not two)
  3. Runs batched concept induction (all entities+relations per page in 1-2
     LLM calls, not ~60)
  4. Writes Entity nodes + TRIPLE edges + OntologyClass nodes to Neo4j using
     UNWIND bulk queries (2 queries per page, not 3N)
  5. Builds FAISS indexes with parallel embedding batches

Performance flags:
  --workers N        Parallel pages processed concurrently (default 4).
                     Each worker now does far less Ollama work per page.
  --embed-workers N  Parallel embedding batches for FAISS (default 4).
  --resume           Skip pages already in the disk cache (default on).
  --no-cache         Ignore disk cache; re-extract all pages.
  --skip-concepts    Skip concept induction entirely (fastest mode).
  --skip-faiss       Skip FAISS index build.
  --force-rebuild    Clear cache + Neo4j data and rebuild from scratch.

Usage:
    python run_kg_build.py --from-state
    python run_kg_build.py --doc-id <id>
    python run_kg_build.py --doc-id <id> --workers 6 --embed-workers 4
    python run_kg_build.py --doc-id <id> --skip-concepts --skip-faiss   # fastest
    python run_kg_build.py --doc-id <id> --force-rebuild                # full reset
"""

import sys
import argparse
import json
import threading
import time
from pathlib import Path
from collections import defaultdict

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    MofNCompleteColumn, TimeElapsedColumn, TaskProgressColumn,
)
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import PROCESSED_DIR, NodeLabel
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
        description="DPR Validation — Step 1C: KG Build (triples + concepts + FAISS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Speed guide:
  Default (--workers 4):        balanced — parallel pages, all optimisations on
  --skip-concepts:              ~30% faster — no concept/schema layer
  --skip-faiss:                 skips embedding entirely (save for later)
  --workers 8:                  more parallelism (needs more RAM + Ollama queuing)
  --embed-workers 2:            lower if Ollama runs out of memory during FAISS
  --force-rebuild:              full reset — clears cache and Neo4j data
        """
    )
    parser.add_argument("--doc-id",        type=str,  help="Document ID")
    parser.add_argument("--from-state",    action="store_true")
    parser.add_argument("--workers",       type=int, default=4,
                        help="Parallel page workers (default 4)")
    parser.add_argument("--embed-workers", type=int, default=4,
                        help="Parallel FAISS embedding workers (default 4)")
    parser.add_argument("--skip-concepts", action="store_true",
                        help="Skip concept induction (faster, no schema layer)")
    parser.add_argument("--skip-faiss",    action="store_true",
                        help="Skip FAISS index build")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Clear page cache + delete Neo4j KG data and rebuild")
    parser.add_argument("--no-cache",      action="store_true",
                        help="Ignore disk page cache (re-extract all pages via Ollama)")
    return parser.parse_args()


# ─── Load metadata ────────────────────────────────────────────────────────────

def load_metadata(doc_id: str) -> dict:
    meta_path = PROCESSED_DIR / doc_id / "metadata.json"
    if not meta_path.exists():
        console.print(f"[red]No metadata for {doc_id}. Run run_extraction.py first.[/red]")
        sys.exit(1)
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ─── Per-page KG build worker ─────────────────────────────────────────────────

def _kg_build_page(
    page,
    doc_id: str,
    sector: str,
    induce_schema: bool,
    use_cache: bool,
) -> dict:
    """Worker: build KG for one page. Returns stats dict."""
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
                "relation":         t.relation,
                "tail":             t.tail,
                "triple_type":      t.triple_type,
                "head_concepts":    t.head_concepts,
                "relation_concepts": t.relation_concepts,
                "tail_concepts":    t.tail_concepts,
                "source_page":      page.page_num + 1,
                "doc_id":           doc_id,
                "sector":           sector,
            }
            for t in result.triples
        ]
        return {
            "page_num":    page.page_num,
            "triples":     len(result.triples),
            "entities":    len(result.all_entities),
            "concepts":    len(result.concept_map),
            "error":       None,
            "cached":      False,
            "triples_data": triples_data,
        }
    except Exception as e:
        logger.warning(f"KG build failed on page {page.page_num + 1}: {e}")
        return {
            "page_num":    page.page_num,
            "triples":     0,
            "entities":    0,
            "concepts":    0,
            "error":       str(e),
            "cached":      False,
            "triples_data": [],
        }


# ─── Pre-flight: identify cached vs uncached pages ────────────────────────────

def _split_cached_pages(pages: list, doc_id: str, use_cache: bool) -> tuple[list, list]:
    """
    Split pages into (cached_pages, uncached_pages).
    Cached pages already have disk cache entries — they skip Ollama entirely.
    """
    if not use_cache:
        return [], pages

    cached, uncached = [], []
    for page in pages:
        if _load_page_cache(doc_id, page.page_num) is not None:
            cached.append(page)
        else:
            uncached.append(page)
    return cached, uncached


# ─── Speed summary panel ──────────────────────────────────────────────────────

def _print_speed_summary(
    t_elapsed: float,
    total_pages: int,
    cached_count: int,
    total_triples: int,
    total_entities: int,
    errors: int,
):
    pages_per_min = round(total_pages / (t_elapsed / 60), 1) if t_elapsed > 0 else 0
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("", style="dim")
    table.add_column("", style="bold")
    table.add_row("Pages processed",    str(total_pages))
    table.add_row("Pages from cache",   f"{cached_count} (no Ollama call)")
    table.add_row("Pages via Ollama",   str(total_pages - cached_count))
    table.add_row("Triples extracted",  str(total_triples))
    table.add_row("Entities extracted", str(total_entities))
    table.add_row("Errors",             str(errors))
    table.add_row("Elapsed",            f"{t_elapsed:.1f}s")
    table.add_row("Throughput",         f"{pages_per_min} pages/min")
    console.print(Panel(table, title="[bold green]KG Build Complete[/bold green]", border_style="green"))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    doc_id = args.doc_id
    state_file = Path("output/.extraction_state.json")

    if doc_id is None and state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        doc_id = state.get("dpr", {}).get("doc_id")
        if doc_id:
            console.print(f"📂 Auto-loaded: doc_id=[cyan]{doc_id}[/cyan]")

    if not doc_id:
        console.print("[red]No document found. Run run_extraction.py first or pass --doc-id.[/red]")
        sys.exit(1)

    console.print("🔌 Connecting to Neo4j...")
    init_schema()

    metadata = load_metadata(doc_id)
    sector   = metadata.get("sector", "")
    filename = metadata.get("filename", "")

    console.print(
        f"📋 Document: [cyan]{filename}[/cyan] | "
        f"Sector: [green]{sector}[/green]"
    )

    # ── Force rebuild: clear cache + KG data ──────────────────────────────────
    if args.force_rebuild:
        console.print("[yellow]⚠ Force rebuild: clearing page cache and Neo4j KG data...[/yellow]")
        clear_page_cache(doc_id)
        run_write(
            "MATCH (e:Entity {doc_id: $id}) DETACH DELETE e",
            {"id": doc_id}
        )
        run_write(
            "MATCH (o:OntologyClass {doc_id: $id}) DETACH DELETE o",
            {"id": doc_id}
        )

    # ── Check if KG already fully built (no force-rebuild) ────────────────────
    if not args.force_rebuild:
        stats = get_kg_stats(doc_id)
        if stats["triples"] > 0:
            console.print(
                f"[yellow]KG already exists: "
                f"{stats['triples']} triples, {stats['entities']} entities. "
                f"Use --force-rebuild to rebuild from scratch.[/yellow]"
            )
            if not args.skip_faiss:
                console.print("🔢 Building FAISS indexes (skipping KG extraction)...")
                idx_stats = build_kg_index(
                    doc_id,
                    force_rebuild=False,
                    embed_workers=args.embed_workers,
                )
                console.print(f"   FAISS: {idx_stats['nodes']} nodes, {idx_stats['edges']} edges")
            sys.exit(0)

    # ── Load document pages ────────────────────────────────────────────────────
    from config.settings import DPR_INPUT_DIR
    possible_paths = [
        DPR_INPUT_DIR / filename,
        Path("data/uploads") / filename,
        Path("data") / filename,
        Path(filename),
    ]
    doc_path = next((p for p in possible_paths if p.exists()), None)

    if doc_path is None:
        console.print(f"[red]Cannot find source document: {filename}[/red]")
        for p in possible_paths:
            console.print(f"  [dim]{p}[/dim]")
        sys.exit(1)

    console.print("📄 Loading document text (no re-OCR)...")
    doc = load_document(doc_path, doc_id)

    processed_pages_1indexed = set(metadata.get("processed_pages", []))
    pages_to_build = [
        p for p in doc.pages
        if (p.page_num + 1) in processed_pages_1indexed
        and p.text
        and len(p.text.strip()) > 50
    ]

    # ── Split cached vs uncached ───────────────────────────────────────────────
    use_cache = not args.no_cache
    cached_pages, uncached_pages = _split_cached_pages(pages_to_build, doc_id, use_cache)

    console.print(
        f"\n   Total pages to build:  [bold]{len(pages_to_build)}[/bold]\n"
        f"   Already cached (fast): [green]{len(cached_pages)}[/green]\n"
        f"   Need Ollama call:      [cyan]{len(uncached_pages)}[/cyan]\n"
        f"   Workers:               [cyan]{args.workers}[/cyan]\n"
        f"   Concept induction:     {'[dim]disabled[/dim]' if args.skip_concepts else '[green]enabled (batched)[/green]'}\n"
        f"   Page cache:            {'[dim]disabled[/dim]' if not use_cache else '[green]enabled[/green]'}"
    )

    # ── Parallel KG build ─────────────────────────────────────────────────────
    total_triples    = 0
    total_entities   = 0
    total_concepts   = 0
    errors           = 0
    all_triples_data = []
    t_start          = time.time()

    console.rule("[bold]Triple Extraction + Concept Induction[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Building KG...", total=len(pages_to_build))

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _kg_build_page,
                    page, doc_id, sector,
                    not args.skip_concepts,
                    use_cache,
                ): page.page_num
                for page in pages_to_build
            }

            for future in as_completed(futures):
                result = future.result()
                with _progress_lock:
                    suffix = "[dim](cache)[/dim]" if result.get("cached") else ""
                    progress.advance(task)
                    progress.update(
                        task,
                        description=(
                            f"p{result['page_num']+1} "
                            f"{result['triples']}t {suffix}"
                        ),
                    )
                total_triples  += result["triples"]
                total_entities += result["entities"]
                total_concepts += result["concepts"]
                all_triples_data.extend(result.get("triples_data", []))
                if result["error"]:
                    errors += 1

    t_elapsed = time.time() - t_start

    # ── Save combined triples JSON ─────────────────────────────────────────────
    triples_path = PROCESSED_DIR / doc_id / "triples_raw.json"
    triples_path.write_text(
        json.dumps(all_triples_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"\n💾 Triples saved: [cyan]{triples_path}[/cyan] ({len(all_triples_data)} triples)")

    # ── Final Neo4j stats ──────────────────────────────────────────────────────
    stats = get_kg_stats(doc_id)
    _print_speed_summary(
        t_elapsed,
        total_pages=len(pages_to_build),
        cached_count=len(cached_pages),
        total_triples=stats["triples"],
        total_entities=stats["entities"],
        errors=errors,
    )

    # ── FAISS index build ──────────────────────────────────────────────────────
    if not args.skip_faiss:
        console.rule("[bold]FAISS Index Build[/bold]")
        console.print(
            f"   Embedding {stats['entities']} nodes + {stats['triples']} edges "
            f"with mxbai-embed-large "
            f"(embed_workers={args.embed_workers})..."
        )
        idx_stats = build_kg_index(
            doc_id,
            force_rebuild=True,
            embed_workers=args.embed_workers,
        )
        console.print(
            f"   Node index: [green]{idx_stats['nodes']}[/green] vectors\n"
            f"   Edge index: [green]{idx_stats['edges']}[/green] vectors\n"
            f"   Saved to:   [cyan]{PROCESSED_DIR / doc_id / 'faiss'}[/cyan]"
        )
    else:
        console.print("[dim]FAISS index: skipped[/dim]")

    # ── Update state file ──────────────────────────────────────────────────────
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    else:
        state = {}
    state["kg_build"] = {
        "doc_id":      doc_id,
        "triples":     stats["triples"],
        "entities":    stats["entities"],
        "concepts":    stats["concepts"],
        "faiss_built": not args.skip_faiss,
        "elapsed_s":   round(t_elapsed, 1),
    }
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    console.print(f"\nNext step: [bold]python run_engines.py --doc-id {doc_id}[/bold]")


if __name__ == "__main__":
    main()