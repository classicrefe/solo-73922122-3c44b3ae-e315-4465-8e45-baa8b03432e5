"""Application services: preview, confirm, and demo seeding.

All mutating operations run inside a ``BEGIN IMMEDIATE`` transaction (see
``db.transaction``).  A confirmation either fully commits the new tables, the
record statuses and the state pointer, or leaves both the old tables and every
record status untouched: SQLite DML and DDL are transactional and every write
below happens inside one transaction, so an exception anywhere rolls back the
whole change.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from dataclasses import dataclass
from typing import Any, Sequence

from . import db
from .selection import Band, Preview, Record, Retained, select_records
from .validation import (
    ParsedTables,
    ValidationError,
    parse_and_validate_tables,
    parse_bands,
    to_records,
)

DISPLACED_REASON_LABELS = {
    "before_out_of_range": "烘烤前读数不在新表任何档内",
    "after_out_of_range": "烘烤后读数不在新表任何档内",
    "both_out_of_range": "烘烤前后读数均不在新表任何档内",
    "capacity": "所在档件数已满，按登记顺序落选",
    "previously_needs_review": "此前已待复核，保持待复核",
}


class ConflictError(RuntimeError):
    """Preview token is stale or another swap has already become effective."""

    def __init__(self, message: str, details: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = list(details or [])


@dataclass(frozen=True)
class ConfirmResult:
    batch_id: int
    retained_count: int
    newly_needs_review: int
    already_under_review: int
    superseded_batch_id: int | None


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_bands(conn: sqlite3.Connection, batch_id: int, side: str) -> list[Band]:
    return [
        Band(
            low=int(row["low_um"]),
            high=int(row["high_um"]),
            pieces=int(row["pieces"]),
            band_id=int(row["id"]),
            position=int(row["position"]),
        )
        for row in db.fetch_bands(conn, batch_id, side)
    ]


def _load_records(conn: sqlite3.Connection) -> list[Record]:
    return to_records(
        [
            (
                int(row["id"]),
                str(row["coupon_no"]),
                int(row["before_um"]),
                int(row["after_um"]),
                str(row["status"]),
            )
            for row in db.fetch_records(conn)
        ]
    )


def records_fingerprint(conn: sqlite3.Connection) -> str:
    """Stable digest of everything a preview's decision depends on."""
    digest = hashlib.sha256()
    for row in conn.execute(
        "SELECT id, coupon_no, before_um, after_um, status FROM records ORDER BY id"
    ):
        digest.update(
            (
                f"{row['id']}|{row['coupon_no']}|{row['before_um']}|"
                f"{row['after_um']}|{row['status']}\n"
            ).encode()
        )
    return digest.hexdigest()


def preview_batch(conn: sqlite3.Connection, batch_id: int) -> Preview:
    return select_records(
        _load_bands(conn, batch_id, "before"),
        _load_bands(conn, batch_id, "after"),
        _load_records(conn),
    )


@dataclass(frozen=True)
class PreviewSnapshot:
    batch_id: int
    token: str
    preview: Preview
    before_bands: list[Band]
    after_bands: list[Band]
    payload: dict[str, Any]
    generation: int
    stale: bool = False
    stale_reason: str | None = None

    @property
    def retained_count(self) -> int:
        return self.preview.retained_count


def _snapshot_payload(
    preview: Preview,
    before_bands: Sequence[Band],
    after_bands: Sequence[Band],
) -> dict[str, Any]:
    def band_dict(band: Band, used: int) -> dict[str, Any]:
        return {
            "position": band.position,
            "low": band.low,
            "high": band.high,
            "pieces": band.pieces,
            "used": used,
            "free": band.pieces - used,
        }

    def record_dict(record: Record, **extra: Any) -> dict[str, Any]:
        return {
            "record_id": record.record_id,
            "coupon_no": record.coupon_no,
            "before_um": record.before_um,
            "after_um": record.after_um,
            **extra,
        }

    return {
        "retained": [
            record_dict(
                item.record,
                before_position=before_bands[item.before_index].position,
                after_position=after_bands[item.after_index].position,
                before_band=[
                    before_bands[item.before_index].low,
                    before_bands[item.before_index].high,
                ],
                after_band=[
                    after_bands[item.after_index].low,
                    after_bands[item.after_index].high,
                ],
            )
            for item in preview.retained
        ],
        "displaced": [
            record_dict(
                item.record,
                reason=item.reason,
                reason_label=DISPLACED_REASON_LABELS.get(item.reason, item.reason),
            )
            for item in preview.displaced
        ],
        "already_under_review": [record_dict(r) for r in preview.already_under_review],
        "before_usage": [band_dict(item.band, item.used) for item in preview.before_usage],
        "after_usage": [band_dict(item.band, item.used) for item in preview.after_usage],
        "max_retainable": preview.max_retainable,
        "active_count": preview.active_count,
        "retained_count": preview.retained_count,
        "displaced_count": preview.displaced_count,
        "total_count": preview.all_records_count,
    }


def create_preview(
    conn: sqlite3.Connection,
    before_text: str,
    after_text: str,
    note: str | None = None,
) -> PreviewSnapshot:
    """Validate both tables, persist a draft batch, and store a preview."""
    tables: ParsedTables = parse_and_validate_tables(before_text, after_text)
    token = secrets.token_urlsafe(24)

    with db.transaction(conn):
        batch_id = db.create_batch(
            conn, tables.before, tables.after, note=(note or "").strip() or None
        )
        before_bands = _load_bands(conn, batch_id, "before")
        after_bands = _load_bands(conn, batch_id, "after")
        preview = select_records(before_bands, after_bands, _load_records(conn))
        payload = _snapshot_payload(preview, before_bands, after_bands)
        generation = int(db.fetch_state(conn)["generation"])
        conn.execute(
            "UPDATE batches SET preview_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'),"
            " preview_token = ?, preview_generation = ?, records_fingerprint = ?,"
            " preview_json = ? WHERE id = ?",
            (
                token,
                generation,
                records_fingerprint(conn),
                json.dumps(payload, ensure_ascii=False),
                batch_id,
            ),
        )

    return PreviewSnapshot(
        batch_id=batch_id,
        token=token,
        preview=preview,
        before_bands=before_bands,
        after_bands=after_bands,
        payload=payload,
        generation=generation,
    )


def load_snapshot(conn: sqlite3.Connection, batch_id: int) -> PreviewSnapshot | None:
    row = db.fetch_batch(conn, batch_id)
    if row is None or not row["preview_json"]:
        return None
    state = db.fetch_state(conn)
    stale_reason = None
    if row["status"] != "draft":
        stale_reason = "该批次已确认或已被取代。"
    elif int(row["preview_generation"] or 0) != int(state["generation"]):
        stale_reason = "预览之后已有另一次换表生效，须重新预览。"
    elif row["records_fingerprint"] != records_fingerprint(conn):
        stale_reason = "预览之后试片记录发生变化，须重新预览。"
    return PreviewSnapshot(
        batch_id=batch_id,
        token=str(row["preview_token"] or ""),
        preview=preview_batch(conn, batch_id),
        before_bands=_load_bands(conn, batch_id, "before"),
        after_bands=_load_bands(conn, batch_id, "after"),
        payload=json.loads(row["preview_json"]),
        generation=int(row["preview_generation"] or 0),
        stale=stale_reason is not None,
        stale_reason=stale_reason,
    )


def current_tables(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """The effective band tables plus per-band usage from the audit rows."""
    state = db.fetch_state(conn)
    if state["current_batch_id"] is None:
        return None
    batch_id = int(state["current_batch_id"])
    batch = db.fetch_batch(conn, batch_id)
    if batch is None:
        return None
    before_bands = _load_bands(conn, batch_id, "before")
    after_bands = _load_bands(conn, batch_id, "after")

    used_before: dict[int, int] = {}
    used_after: dict[int, int] = {}
    for row in conn.execute(
        "SELECT before_band, after_band FROM record_assignments"
        " WHERE batch_id = ? AND valid = 1",
        (batch_id,),
    ):
        if row["before_band"] is not None:
            used_before[int(row["before_band"])] = (
                used_before.get(int(row["before_band"]), 0) + 1
            )
        if row["after_band"] is not None:
            used_after[int(row["after_band"])] = (
                used_after.get(int(row["after_band"]), 0) + 1
            )

    def attach(bands: list[Band], used: dict[int, int]) -> list[dict[str, Any]]:
        return [
            {
                "position": band.position,
                "low": band.low,
                "high": band.high,
                "pieces": band.pieces,
                "used": used.get(int(band.band_id or 0), 0),
            }
            for band in bands
        ]

    counts = conn.execute(
        "SELECT COUNT(*) AS total,"
        " COALESCE(SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END), 0) AS valid,"
        " COALESCE(SUM(CASE WHEN status = 'needs_review' THEN 1 ELSE 0 END), 0) AS invalid"
        " FROM records"
    ).fetchone()
    return {
        "batch_id": batch_id,
        "note": batch["note"],
        "confirmed_at": batch["confirmed_at"],
        "before": attach(before_bands, used_before),
        "after": attach(after_bands, used_after),
        "active_count": int(counts["valid"]),
        "review_count": int(counts["invalid"]),
        "record_total": int(counts["total"]),
    }


def confirm_batch(
    conn: sqlite3.Connection, batch_id: int, token: str
) -> ConfirmResult:
    """Atomically enable both new tables and update record statuses.

    Raises ``ValidationError`` for bad input, ``ConflictError`` when the
    preview is no longer valid, and lets unexpected database errors propagate
    after the transaction has rolled back.
    """
    with db.transaction(conn):
        state = db.fetch_state(conn)
        batch = db.fetch_batch(conn, batch_id)
        if batch is None:
            raise ValidationError("换表批次不存在。")
        if batch["status"] != "draft":
            raise ConflictError(
                "该批次已确认或已被取代，不能再次确认。",
                ["请重新上传两张更正表并预览。"],
            )
        if not token or not secrets.compare_digest(
            str(batch["preview_token"] or ""), token
        ):
            raise ConflictError(
                "预览凭据无效，请重新预览。",
                ["确认必须来自同一次预览，旧凭据不能沿用。"],
            )
        if int(batch["preview_generation"] or 0) != int(state["generation"]):
            raise ConflictError(
                "预览之后已有另一次换表生效，须重新预览。",
                [
                    "为避免用过期结果覆盖现行表，本次确认未做任何改动。",
                    "请重新上传两张更正表并再次预览。",
                ],
            )
        if batch["records_fingerprint"] != records_fingerprint(conn):
            raise ConflictError(
                "预览之后试片记录发生变化，须重新预览。",
                ["记录内容（含状态）已与预览时不同，本次确认未做任何改动。"],
            )

        before_bands = _load_bands(conn, batch_id, "before")
        after_bands = _load_bands(conn, batch_id, "after")
        preview = select_records(before_bands, after_bands, _load_records(conn))

        snapshot_ids = [
            int(item["record_id"])
            for item in json.loads(batch["preview_json"])["retained"]
        ]
        if snapshot_ids != [item.record.record_id for item in preview.retained]:
            raise ConflictError(
                "预览结果与当前数据不一致，须重新预览。",
                ["系统未能复现预览时的保留清单，本次确认未做任何改动。"],
            )

        retained_by_id = {item.record.record_id: item for item in preview.retained}
        previous_id = state["current_batch_id"]
        superseded: int | None = None
        if previous_id is not None:
            superseded = int(previous_id)
            conn.execute(
                "UPDATE batches SET status = 'superseded',"
                " superseded_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')"
                " WHERE id = ? AND status = 'confirmed'",
                (superseded,),
            )

        conn.execute("DELETE FROM record_assignments WHERE batch_id = ?", (batch_id,))
        for record in _load_records(conn):
            item: Retained | None = retained_by_id.get(record.record_id)
            if item is not None:
                conn.execute(
                    "INSERT INTO record_assignments"
                    " (batch_id, record_id, valid, reason, before_band, after_band)"
                    " VALUES (?, ?, 1, NULL, ?, ?)",
                    (
                        batch_id,
                        record.record_id,
                        int(before_bands[item.before_index].band_id or 0),
                        int(after_bands[item.after_index].band_id or 0),
                    ),
                )
            else:
                if record.status == "active":
                    reason = "capacity"
                    for displaced in preview.displaced:
                        if displaced.record.record_id == record.record_id:
                            reason = displaced.reason
                            break
                else:
                    reason = "previously_needs_review"
                conn.execute(
                    "INSERT INTO record_assignments"
                    " (batch_id, record_id, valid, reason, before_band, after_band)"
                    " VALUES (?, ?, 0, ?, NULL, NULL)",
                    (batch_id, record.record_id, reason),
                )

        newly_review = [
            item.record.record_id
            for item in preview.displaced
            if item.record.status == "active"
        ]
        if newly_review:
            conn.executemany(
                "UPDATE records SET status = 'needs_review'"
                " WHERE id = ? AND status = 'active'",
                [(record_id,) for record_id in newly_review],
            )

        conn.execute(
            "UPDATE batches SET status = 'confirmed',"
            " confirmed_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?",
            (batch_id,),
        )
        conn.execute(
            "UPDATE app_state SET generation = generation + 1, current_batch_id = ?"
            " WHERE id = 1",
            (batch_id,),
        )

    return ConfirmResult(
        batch_id=batch_id,
        retained_count=preview.retained_count,
        newly_needs_review=len(newly_review),
        already_under_review=len(preview.already_under_review),
        superseded_batch_id=superseded,
    )


# ---------------------------------------------------------------------------
# Demo seed
# ---------------------------------------------------------------------------

DEMO_RECORDS: tuple[tuple[str, int, int], ...] = (
    ("P-001", 8, 11), ("P-002", 12, 15), ("P-003", 18, 21),
    ("P-004", 22, 26), ("P-005", 28, 31), ("P-006", 33, 36),
    ("P-007", 38, 43), ("P-008", 45, 48), ("P-009", 52, 55),
    ("P-010", 57, 61), ("P-011", 63, 66), ("P-012", 68, 72),
    ("P-013", 74, 77), ("P-014", 79, 83), ("P-015", 85, 88),
    ("P-016", 6, 9), ("P-017", 14, 18), ("P-018", 19, 24),
    ("P-019", 25, 29), ("P-020", 30, 34), ("P-021", 35, 39),
    ("P-022", 40, 44), ("P-023", 46, 50), ("P-024", 51, 54),
    ("P-025", 56, 60), ("P-026", 62, 65), ("P-027", 67, 70),
    ("P-028", 71, 75), ("P-029", 76, 80), ("P-030", 82, 86),
    ("P-031", 87, 90), ("P-032", 10, 13), ("P-033", 16, 20),
    ("P-034", 20, 25), ("P-035", 27, 30), ("P-036", 32, 35),
    ("P-037", 37, 41), ("P-038", 39, 45), ("P-039", 44, 47),
    ("P-040", 49, 53), ("P-041", 53, 57), ("P-042", 58, 62),
    ("P-043", 64, 68), ("P-044", 69, 73), ("P-045", 73, 76),
    ("P-046", 78, 82), ("P-047", 81, 84), ("P-048", 86, 89),
    ("P-049", 5, 8), ("P-050", 24, 27), ("P-051", 31, 33),
    ("P-052", 42, 46), ("P-053", 48, 52), ("P-054", 55, 59),
    ("P-055", 60, 63), ("P-056", 65, 69), ("P-057", 70, 74),
    ("P-058", 77, 81), ("P-059", 83, 87), ("P-060", 89, 92),
)

DEMO_BEFORE_TABLE = """\
# 烘烤前膜厚分档表（现行）
0-20 10
21-40 14
41-60 14
61-90 22
"""

DEMO_AFTER_TABLE = """\
# 烘烤后膜厚分档表（现行）
0-25 10
26-45 14
46-65 15
66-95 21
"""


def seed_demo(conn: sqlite3.Connection) -> bool:
    """Insert demo records and the first table pair once.

    Returns True when seeding happened.  The whole seed is one transaction, so
    a crash leaves an empty database rather than a half-seeded one.
    """
    with db.transaction(conn):
        if db.count_records(conn) > 0:
            return False
        if db.fetch_state(conn)["current_batch_id"] is not None:
            return False
        db.add_records(conn, DEMO_RECORDS)
        batch_id = db.create_batch(
            conn,
            parse_bands(DEMO_BEFORE_TABLE),
            parse_bands(DEMO_AFTER_TABLE),
            note="演示种子：初始分档表",
        )
        before_bands = _load_bands(conn, batch_id, "before")
        after_bands = _load_bands(conn, batch_id, "after")
        preview = select_records(before_bands, after_bands, _load_records(conn))
        retained = {item.record.record_id: item for item in preview.retained}
        for record in _load_records(conn):
            item = retained.get(record.record_id)
            conn.execute(
                "INSERT INTO record_assignments"
                " (batch_id, record_id, valid, reason, before_band, after_band)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    batch_id,
                    record.record_id,
                    1 if item else 0,
                    None if item else "previously_needs_review",
                    int(before_bands[item.before_index].band_id or 0) if item else None,
                    int(after_bands[item.after_index].band_id or 0) if item else None,
                ),
            )
        if preview.displaced:
            conn.executemany(
                "UPDATE records SET status = 'needs_review' WHERE id = ?",
                [(item.record.record_id,) for item in preview.displaced],
            )
        conn.execute(
            "UPDATE batches SET status = 'confirmed',"
            " confirmed_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?",
            (batch_id,),
        )
        conn.execute(
            "UPDATE app_state SET generation = generation + 1, current_batch_id = ?"
            " WHERE id = 1",
            (batch_id,),
        )
    return True
