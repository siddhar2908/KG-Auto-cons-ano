"""
utils/neo4j_client.py
---------------------
Neo4j connection wrapper with schema initialisation.
Creates all constraints and indexes on first run.
"""

import sys
import logging
from contextlib import contextmanager
from loguru import logger
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

# Suppress Neo4j GQL property-not-found notifications (01N52)
# These fire when querying properties that don't exist yet (e.g. Rule.condition
# before any rules are loaded). Harmless but very noisy in console output.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

from config.settings import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
    NodeLabel, RelType
)

# ─── Singleton driver ─────────────────────────────────────────────────────────

_driver: Driver | None = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        try:
            _driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                max_connection_lifetime=3600,
                connection_acquisition_timeout=30,
            )
            _driver.verify_connectivity()
            logger.success(f"Connected to Neo4j at {NEO4J_URI}")
        except ServiceUnavailable:
            logger.error(f"Neo4j is not running at {NEO4J_URI}. Start it first.")
            sys.exit(1)
        except AuthError:
            logger.error("Neo4j authentication failed. Check NEO4J_USER/PASSWORD in settings.py")
            sys.exit(1)
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


@contextmanager
def get_session(database: str = NEO4J_DATABASE):
    driver = get_driver()
    with driver.session(database=database) as session:
        yield session


# ─── Schema initialisation ────────────────────────────────────────────────────

SCHEMA_QUERIES = [
    # Uniqueness constraints
    f"CREATE CONSTRAINT IF NOT EXISTS FOR (d:{NodeLabel.DOCUMENT}) REQUIRE d.doc_id IS UNIQUE",
    f"CREATE CONSTRAINT IF NOT EXISTS FOR (f:{NodeLabel.FACT}) REQUIRE f.fact_id IS UNIQUE",
    f"CREATE CONSTRAINT IF NOT EXISTS FOR (r:{NodeLabel.RULE}) REQUIRE r.rule_id IS UNIQUE",
    f"CREATE CONSTRAINT IF NOT EXISTS FOR (e:{NodeLabel.ENTITY}) REQUIRE e.entity_id IS UNIQUE",
    f"CREATE CONSTRAINT IF NOT EXISTS FOR (s:{NodeLabel.SECTOR}) REQUIRE s.name IS UNIQUE",
    f"CREATE CONSTRAINT IF NOT EXISTS FOR (v:{NodeLabel.VIOLATION}) REQUIRE v.violation_id IS UNIQUE",

    # Indexes for fast lookup
    f"CREATE INDEX IF NOT EXISTS FOR (f:{NodeLabel.FACT}) ON (f.sector)",
    f"CREATE INDEX IF NOT EXISTS FOR (f:{NodeLabel.FACT}) ON (f.fact_type)",
    f"CREATE INDEX IF NOT EXISTS FOR (f:{NodeLabel.FACT}) ON (f.doc_id)",
    f"CREATE INDEX IF NOT EXISTS FOR (r:{NodeLabel.RULE}) ON (r.sector)",
    f"CREATE INDEX IF NOT EXISTS FOR (r:{NodeLabel.RULE}) ON (r.standard_name)",
    f"CREATE INDEX IF NOT EXISTS FOR (v:{NodeLabel.VIOLATION}) ON (v.severity)",
]


def init_schema():
    """Run all schema setup queries. Safe to call multiple times (IF NOT EXISTS)."""
    logger.info("Initialising Neo4j schema...")
    with get_session() as session:
        for q in SCHEMA_QUERIES:
            try:
                session.run(q)
            except Exception as e:
                logger.warning(f"Schema query skipped ({e}): {q[:60]}...")

    # Ensure sector nodes exist
    _seed_sectors()
    logger.success("Neo4j schema ready.")


def _seed_sectors():
    """Create a Sector node for each of the 8 RITES sectors."""
    from config.settings import SECTORS, SECTOR_KEYS, SECTOR_STANDARDS
    with get_session() as session:
        for sector_name in SECTORS:
            key = SECTOR_KEYS[sector_name]
            standards = SECTOR_STANDARDS.get(key, [])
            session.run(
                f"""
                MERGE (s:{NodeLabel.SECTOR} {{name: $name}})
                SET s.key = $key,
                    s.standards = $standards
                """,
                name=sector_name, key=key, standards=standards
            )


# ─── Convenience write helper ─────────────────────────────────────────────────

def run_write(query: str, params: dict = None) -> list:
    with get_session() as session:
        result = session.run(query, params or {})
        return result.data()


def run_read(query: str, params: dict = None) -> list:
    with get_session() as session:
        result = session.run(query, params or {})
        return result.data()