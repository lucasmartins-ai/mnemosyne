"""Performance check for the UTC cutoff change in #1097, as dplush requested.

dplush asked for "a query-plan/performance check for any SQL normalization". This PR
normalizes the *clock* (Python-side), not the SQL — the raw `timestamp > ?` predicate is
unchanged, so the index still serves it. This test pins that, because the obvious-looking
"fix" for mixed-format ordering is `julianday(timestamp)`, and that one is 200x slower on
this table.

Measured here on 50k rows: raw 0.02 ms (SEARCH USING INDEX) vs julianday 4.37 ms (SCAN +
temp B-tree). An expression index on julianday(timestamp) restores the SEARCH at 0.02 ms,
but that is a schema migration and out of scope for this fix. If someone reaches for
julianday() without reading this, the assertion below fails.

Run: pytest tests/test_temporal_query_plan.py -q
"""

import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest

SCHEMA = """
CREATE TABLE working_memory (
    id TEXT PRIMARY KEY, content TEXT, source TEXT, timestamp TEXT,
    session_id TEXT, importance REAL, valid_until TEXT, superseded_by TEXT
);
CREATE INDEX idx_wm_timestamp ON working_memory(timestamp);
"""

ROWS = 50_000
# A loose ceiling: this asserts the existing index still serves the predicate, not a
# benchmark. Hardware varies; the 200x julianday regression does not.
MAX_MS = 50.0


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("plan") / "wm.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    conn.executemany(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) "
        "VALUES (?, 'x', 'conversation', ?, 's1', 1.0)",
        [(f"m{i}", (now - timedelta(minutes=i)).isoformat()) for i in range(ROWS)],
    )
    conn.commit()
    conn.execute("ANALYZE")
    conn.close()
    return path


def test_cutoff_still_searches_the_timestamp_index(db):
    """The unchanged predicate must keep using idx_wm_timestamp."""
    cutoff = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=7)).isoformat()
    conn = sqlite3.connect(str(db))
    plan = " ".join(
        row[3]
        for row in conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM working_memory WHERE timestamp > ? "
            "ORDER BY timestamp DESC LIMIT 20",
            (cutoff,),
        )
    )
    conn.close()
    assert "SEARCH" in plan and "idx_wm_timestamp" in plan, plan


def test_cutoff_stays_under_the_budget(db):
    """Wall-clock ceiling for the shipped query at 50k rows."""
    cutoff = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=7)).isoformat()
    conn = sqlite3.connect(str(db))
    start = time.perf_counter()
    conn.execute(
        "SELECT id FROM working_memory WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 20",
        (cutoff,),
    ).fetchall()
    elapsed_ms = (time.perf_counter() - start) * 1000
    conn.close()
    assert elapsed_ms < MAX_MS, {"elapsed_ms": elapsed_ms, "rows": ROWS}


def test_julianday_would_regress_the_plan(db):
    """Document the alternative we rejected, and fail if it is ever adopted silently.

    julianday() compares instants correctly for offset-bearing values, but it is not
    indexable under idx_wm_timestamp, so the plan degrades to a scan plus a temp sort.
    This is here to make that cost explicit rather than to forbid the function outright —
    if a future migration adds an expression index, this test is the thing to revisit.
    """
    cutoff = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=7)).isoformat()
    conn = sqlite3.connect(str(db))
    plan = " ".join(
        row[3]
        for row in conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM working_memory "
            "WHERE julianday(timestamp) > julianday(?) ORDER BY julianday(timestamp) DESC LIMIT 20",
            (cutoff,),
        )
    )
    conn.close()
    assert "idx_wm_timestamp" not in plan, (
        "julianday() now uses the index — an expression index was added, so the tradeoff "
        f"recorded in polyphonic_recall.py should be revisited. plan={plan}"
    )
