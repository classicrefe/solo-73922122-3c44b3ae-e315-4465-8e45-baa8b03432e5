"""SQLite schema and low-level helpers for the coating-lab re-banding tool.

Domain model
------------
* ``records`` – test coupons (试片) in registration order. Each has a unique
  coupon number, integer before/after bake thickness readings (µm), and a
  review status: ``active`` (现行有效) or ``needs_review`` (待复核).
* ``batches`` – one upload/preview/confirm cycle. Both replacement tables
  (before and after bake) live in the same batch, so they are enabled in one
  transaction.
* ``bands`` – one closed integer µm interval plus a piece count, per batch and
  side. Within one side the bands must not overlap.
* ``record_assignments`` – immutable audit of how every record was classified
  when a batch was confirmed, including which band it occupied on each side.
* ``app_state`` – the current batch pointer and a monotonically increasing
  generation used to invalidate previews once another swap takes effect.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

MAX_RECORDS = 60

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    coupon_no     TEXT    NOT NULL UNIQUE,
    before_um     INTEGER NOT NULL,
    after_um      INTEGER NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active', 'needs_review')),
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_records_status_id ON records(status, id);

-- The lab caps the batch at 60 coupons. Enforce it in the database as well as
-- in the importer so a code-path bug cannot quietly exceed the limit.
CREATE TRIGGER IF NOT EXISTS trg_records_max
BEFORE INSERT ON records
WHEN (SELECT COUNT(*) FROM records) >= 60
BEGIN
    SELECT RAISE(ABORT, 'records limit reached (max 60)');
END;

CREATE TABLE IF NOT EXISTS batches (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    status               TEXT    NOT NULL DEFAULT 'draft'
                         CHECK (status IN ('draft', 'confirmed', 'superseded')),
    note                 TEXT,
    created_at           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    preview_at           TEXT,
    preview_token        TEXT,
    preview_generation   INTEGER,
    records_fingerprint  TEXT,
    preview_json         TEXT,
    confirmed_at         TEXT,
    superseded_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status);

CREATE TABLE IF NOT EXISTS bands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id    INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    side        TEXT    NOT NULL CHECK (side IN ('before', 'after')),
    position    INTEGER NOT NULL,
    low_um      INTEGER NOT NULL,
    high_um     INTEGER NOT NULL,
    pieces      INTEGER NOT NULL CHECK (pieces >= 0),
    UNIQUE (batch_id, side, position),
    CHECK (low_um <= high_um)
);

CREATE INDEX IF NOT EXISTS idx_bands_batch_side ON bands(batch_id, side, position);

CREATE TABLE IF NOT EXISTS record_assignments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id      INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    record_id     INTEGER NOT NULL REFERENCES records(id),
    valid         INTEGER NOT NULL CHECK (valid IN (0, 1)),
    reason        TEXT,
    before_band   INTEGER REFERENCES bands(id),
    after_band    INTEGER REFERENCES bands(id),
    UNIQUE (batch_id, record_id)
);

CREATE INDEX IF NOT EXISTS idx_assignments_batch ON record_assignments(batch_id);

CREATE TABLE IF NOT EXISTS app_state (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    generation        INTEGER NOT NULL DEFAULT 0,
    current_batch_id  INTEGER REFERENCES batches(id)
);

INSERT OR IGNORE INTO app_state (id, generation, current_batch_id)
VALUES (1, 0, NULL);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a connection with the pragmas this app relies on."""
    conn = sqlite3.connect(str(path), timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 15000")
    return conn


def init_db(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Serialised write transaction.

    ``BEGIN IMMEDIATE`` takes the write lock up front, so two confirmations
    cannot interleave their read-check-write sequences.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def add_records(conn: sqlite3.Connection, rows: Iterable[Sequence[object]]) -> int:
    """Append records in the given order (caller owns the transaction)."""
    inserted = 0
    for row in rows:
        coupon_no, before_um, after_um = row
        conn.execute(
            "INSERT INTO records (coupon_no, before_um, after_um, status)"
            " VALUES (?, ?, ?, 'active')",
            (str(coupon_no).strip(), int(before_um), int(after_um)),
        )
        inserted += 1
    return inserted


def count_records(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])


def fetch_records(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM records ORDER BY id"))


def create_batch(
    conn: sqlite3.Connection,
    before_bands: Sequence[tuple[int, int, int]],
    after_bands: Sequence[tuple[int, int, int]],
    note: str | None = None,
) -> int:
    """Insert a draft batch with both tables. Returns the batch id."""
    cur = conn.execute("INSERT INTO batches (status, note) VALUES ('draft', ?)", (note,))
    batch_id = int(cur.lastrowid)
    for side, bands in (("before", before_bands), ("after", after_bands)):
        for position, (low, high, pieces) in enumerate(bands):
            conn.execute(
                "INSERT INTO bands (batch_id, side, position, low_um, high_um, pieces)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (batch_id, side, position, int(low), int(high), int(pieces)),
            )
    return batch_id


def fetch_batch(conn: sqlite3.Connection, batch_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()


def fetch_bands(conn: sqlite3.Connection, batch_id: int, side: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM bands WHERE batch_id = ? AND side = ? ORDER BY position",
            (batch_id, side),
        )
    )


def fetch_state(conn: sqlite3.Connection) -> sqlite3.Row:
    return conn.execute("SELECT * FROM app_state WHERE id = 1").fetchone()


def recent_batches(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM batches ORDER BY id DESC LIMIT ?", (limit,)))
