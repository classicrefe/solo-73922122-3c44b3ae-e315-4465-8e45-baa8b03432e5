import sqlite3

import pytest

from app import db
from app.selection import Band, Record, select_records
from app.service import (
    ConflictError,
    confirm_batch,
    create_preview,
    current_tables,
    load_snapshot,
)
from app.validation import (
    ValidationError,
    parse_and_validate_tables,
    parse_bands,
    parse_records,
    validate_bands,
)


@pytest.fixture
def conn(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    connection = db.connect(path)
    yield connection
    connection.close()


def bands(*specs):
    return [Band(low=lo, high=hi, pieces=p, position=i) for i, (lo, hi, p) in enumerate(specs)]


def records(*specs):
    return [
        Record(record_id=i + 1, coupon_no=f"C-{i + 1:03d}", before_um=b, after_um=a)
        for i, (b, a) in enumerate(specs)
    ]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_parse_bands_accepts_supported_formats():
    text = """
    # comment
    0-20 15
    21..40 12
    41 60 8
    61,80,3
    """
    assert parse_bands(text) == [(0, 20, 15), (21, 40, 12), (41, 60, 8), (61, 80, 3)]


def test_parse_bands_rejects_partial_input():
    with pytest.raises(ValidationError) as excinfo:
        parse_bands("0-20 5\nnonsense\n")
    assert "第 2 行" in " ".join(excinfo.value.all_messages())
    assert len(excinfo.value.details) == 1


def test_validate_bands_rejects_overlap_even_when_touching():
    with pytest.raises(ValidationError):
        validate_bands([(0, 20, 5), (20, 30, 5)], "表")
    validate_bands([(0, 20, 5), (21, 30, 5)], "表")


def test_parse_and_validate_requires_equal_totals():
    with pytest.raises(ValidationError) as excinfo:
        parse_and_validate_tables("0-20 5\n", "0-20 6\n")
    assert "总件数" in excinfo.value.message


def test_parse_and_validate_checks_each_side_for_overlap():
    with pytest.raises(ValidationError):
        parse_and_validate_tables("0-20 5\n15-30 5\n", "0-20 10\n")


def test_parse_records_preserves_order_and_rejects_duplicates():
    rows = parse_records("A-1 3 4\nB-2,5,6\n")
    assert rows == [("A-1", 3, 4), ("B-2", 5, 6)]
    with pytest.raises(ValidationError):
        parse_records("A-1 3 4\nA-1 5 6\n")


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_lexicographic_tiebreak_prefers_earliest_record():
    preview = select_records(
        bands((0, 10, 1), (11, 20, 1)),
        bands((0, 10, 1)),
        records((1, 1), (2, 2)),
    )
    assert preview.max_retainable == 1
    assert [r.record.record_id for r in preview.retained] == [1]


def test_lexicographic_choice_never_reduces_the_maximum():
    # Keeping record 1 would cap the result at 1; records 2+3 achieve 2, so the
    # earliest record must be dropped even though it is registered first.
    preview = select_records(
        bands((0, 10, 1), (11, 20, 1)),
        bands((0, 10, 1), (11, 20, 1)),
        records((1, 1), (1, 15), (15, 1)),
    )
    assert preview.max_retainable == 2
    assert [r.record.record_id for r in preview.retained] == [2, 3]
    assert [d.record.record_id for d in preview.displaced] == [1]
    assert preview.displaced[0].reason == "capacity"


def test_capacity_is_per_band_on_each_side():
    preview = select_records(
        bands((0, 10, 1)),
        bands((0, 10, 1)),
        records((5, 5), (6, 6), (7, 7)),
    )
    assert preview.retained_count == 1
    assert [d.record.record_id for d in preview.displaced] == [2, 3]


def test_band_intervals_are_closed_integers():
    preview = select_records(
        bands((0, 10, 2), (11, 20, 2)),
        bands((0, 10, 2), (11, 20, 2)),
        records((10, 11), (11, 20)),
    )
    assert preview.retained_count == 2


def test_out_of_range_records_are_reported_not_selected():
    preview = select_records(
        bands((0, 10, 2)),
        bands((0, 10, 2)),
        records((5, 5), (50, 5), (5, 50), (50, 50)),
    )
    reasons = {d.record.record_id: d.reason for d in preview.displaced}
    assert reasons == {
        2: "before_out_of_range",
        3: "after_out_of_range",
        4: "both_out_of_range",
    }


def test_records_already_under_review_are_not_candidates():
    recs = [
        Record(1, "A", 5, 5, "active"),
        Record(2, "B", 5, 5, "needs_review"),
    ]
    preview = select_records(bands((0, 10, 1)), bands((0, 10, 1)), recs)
    assert [r.record.record_id for r in preview.retained] == [1]
    assert [r.record_id for r in preview.already_under_review] == [2]


def test_matching_uses_full_bipartite_structure():
    # A per-band greedy count would accept 3; the true maximum is 2.
    preview = select_records(
        bands((0, 10, 1), (11, 20, 1)),
        bands((0, 10, 1), (11, 20, 1)),
        records((5, 15), (15, 5), (15, 15)),
    )
    assert preview.max_retainable == 2
    assert preview.retained_count == 2


# --------------------------------------------------------------------------
# Persistence and confirmation
# --------------------------------------------------------------------------


def _seed_pair(conn):
    """Import three records and put the first table pair into force."""
    with db.transaction(conn):
        db.add_records(conn, [("A-1", 5, 5), ("A-2", 15, 15), ("A-3", 15, 15)])
    snapshot = create_preview(conn, "0-10 1\n11-20 2\n", "0-10 1\n11-20 2\n")
    confirm_batch(conn, snapshot.batch_id, snapshot.token)
    return snapshot


def test_confirm_applies_both_tables_and_marks_displaced(conn):
    snapshot = _seed_pair(conn)
    assert snapshot.preview.retained_count == 3  # 1+2 pieces, all fit

    before = create_preview(conn, "0-10 1\n11-20 1\n", "0-10 1\n11-20 1\n")
    assert before.preview.retained_count == 2
    result = confirm_batch(conn, before.batch_id, before.token)
    assert result.newly_needs_review == 1
    assert result.superseded_batch_id == snapshot.batch_id
    statuses = {
        row["coupon_no"]: row["status"]
        for row in conn.execute("SELECT coupon_no, status FROM records")
    }
    assert statuses["A-1"] == "active"
    assert statuses["A-2"] == "active"
    assert statuses["A-3"] == "needs_review"

    current = current_tables(conn)
    assert current["batch_id"] == before.batch_id
    assert [b["used"] for b in current["before"]] == [1, 1]
    assert current["active_count"] == 2
    assert current["review_count"] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM batches WHERE status = 'confirmed'"
    ).fetchone()[0] == 1


def test_confirm_twice_is_rejected(conn):
    snapshot = _seed_pair(conn)
    # _seed_pair confirmed batch 1; a second confirm of it must fail.
    assert db.fetch_state(conn)["current_batch_id"] == snapshot.batch_id
    with pytest.raises(ConflictError):
        confirm_batch(conn, snapshot.batch_id, snapshot.token)


def test_bad_token_is_rejected_without_changes(conn):
    _seed_pair(conn)
    draft = create_preview(conn, "0-10 1\n11-20 1\n", "0-10 1\n11-20 1\n")
    with pytest.raises(ConflictError):
        confirm_batch(conn, draft.batch_id, "not-the-token")
    assert db.fetch_state(conn)["current_batch_id"] == 1
    assert load_snapshot(conn, draft.batch_id).stale is False


def test_stale_preview_after_another_swap_is_rejected(conn):
    _seed_pair(conn)
    first = create_preview(conn, "0-10 1\n11-20 1\n", "0-10 1\n11-20 1\n")
    second = create_preview(conn, "0-10 2\n11-20 1\n", "0-10 2\n11-20 1\n")
    confirm_batch(conn, second.batch_id, second.token)
    with pytest.raises(ConflictError) as excinfo:
        confirm_batch(conn, first.batch_id, first.token)
    assert "另一次换表生效" in excinfo.value.message
    assert load_snapshot(conn, first.batch_id).stale is True
    assert db.fetch_state(conn)["current_batch_id"] == second.batch_id


def test_record_change_between_preview_and_confirm_is_rejected(conn):
    snapshot = _seed_pair(conn)
    with db.transaction(conn):
        db.add_records(conn, [("A-9", 5, 5)])
    with pytest.raises(ConflictError):
        confirm_batch(conn, snapshot.batch_id, snapshot.token)
    assert db.fetch_state(conn)["current_batch_id"] == 1


def test_write_failure_rolls_back_tables_and_statuses(conn):
    snapshot = _seed_pair(conn)
    target = create_preview(conn, "0-10 1\n11-20 1\n", "0-10 1\n11-20 1\n")
    conn.execute(
        "CREATE TRIGGER fail_confirm BEFORE UPDATE ON records"
        " BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
    )

    with pytest.raises(sqlite3.DatabaseError):
        confirm_batch(conn, target.batch_id, target.token)

    state = db.fetch_state(conn)
    assert state["current_batch_id"] == snapshot.batch_id
    assert state["generation"] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM records WHERE status = 'needs_review'"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT status FROM batches WHERE id = ?", (target.batch_id,)
    ).fetchone()["status"] == "draft"
    assert conn.execute(
        "SELECT COUNT(*) FROM record_assignments WHERE batch_id = ?",
        (target.batch_id,),
    ).fetchone()[0] == 0


def test_import_order_drives_priority(conn):
    with db.transaction(conn):
        db.add_records(conn, [("Z-9", 5, 5), ("A-1", 5, 5)])
    ids = [row["coupon_no"] for row in db.fetch_records(conn)]
    assert ids == ["Z-9", "A-1"]
    snapshot = create_preview(conn, "0-10 1\n", "0-10 1\n")
    assert snapshot.preview.retained[0].record.coupon_no == "Z-9"


def test_database_enforces_sixty_record_cap(conn):
    with db.transaction(conn):
        db.add_records(conn, [(f"R-{i}", 1, 1) for i in range(60)])
    with pytest.raises(sqlite3.DatabaseError):
        with db.transaction(conn):
            db.add_records(conn, [("R-60", 1, 1)])
    assert db.count_records(conn) == 60
