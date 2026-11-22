"""Concurrency: two confirmations of different drafts must not both apply."""

import threading

from app import db
from app.service import ConflictError, confirm_batch, create_preview


def test_concurrent_confirmations_serialise(tmp_path):
    path = tmp_path / "concurrent.db"
    db.init_db(path)
    conn = db.connect(path)
    try:
        with db.transaction(conn):
            db.add_records(conn, [("A-1", 5, 5), ("A-2", 15, 15)])
        first = create_preview(conn, "0-10 1\n11-20 1\n", "0-10 1\n11-20 1\n")
        second = create_preview(conn, "0-10 2\n11-20 2\n", "0-10 2\n11-20 2\n")

        results: list[tuple] = []
        barrier = threading.Barrier(2)

        def worker(snapshot):
            worker_conn = db.connect(path)
            try:
                barrier.wait()
                confirm_batch(worker_conn, snapshot.batch_id, snapshot.token)
                results.append(("ok", snapshot.batch_id))
            except ConflictError as exc:
                results.append(("conflict", snapshot.batch_id, exc.message))
            finally:
                worker_conn.close()

        threads = [
            threading.Thread(target=worker, args=(snapshot,))
            for snapshot in (first, second)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        outcomes = sorted(result[0] for result in results)
        assert outcomes == ["conflict", "ok"], results
        state = db.fetch_state(conn)
        winner = results[[r[0] for r in results].index("ok")][1]
        assert state["current_batch_id"] == winner
        assert state["generation"] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM batches WHERE status = 'confirmed'"
        ).fetchone()[0] == 1
    finally:
        conn.close()
