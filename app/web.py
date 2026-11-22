"""Flask app factory and routes for the coating-lab re-banding page."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from flask import Flask, g, redirect, render_template, request, url_for

from . import db
from .service import (
    DEMO_RECORDS,
    DISPLACED_REASON_LABELS,
    ConflictError,
    confirm_batch,
    create_preview,
    current_tables,
    load_snapshot,
    seed_demo,
)
from .validation import MAX_RECORDS, ValidationError, parse_records

DEFAULT_DB = Path(
    os.environ.get(
        "COATING_DB",
        Path(__file__).resolve().parent.parent / "data" / "coating.db",
    )
)

# A second table pair that deliberately creates both kinds of movement: four
# readings fall outside the new ranges, and a few bands are oversubscribed, so
# the demo shows range-based and capacity-based displacements side by side.
SUGGESTED_BEFORE = """\
0-25 12
26-45 14
46-65 14
66-84 14
"""

SUGGESTED_AFTER = """\
0-30 12
31-50 14
51-70 14
71-95 14
"""

DEMO_RECORDS_TEXT = "\n".join(
    f"{coupon}\t{before}\t{after}" for coupon, before, after in DEMO_RECORDS
)


def _records_rows(conn: sqlite3.Connection):
    return conn.execute(
        "SELECT id, coupon_no, before_um, after_um, status FROM records ORDER BY id"
    ).fetchall()


def _read_text_field(field: str) -> tuple[str, str | None]:
    """Return the uploaded file's text when present, else the textarea value.

    The second element is an error message for an unreadable upload.
    """
    upload = request.files.get(f"{field}_file")
    if upload is not None and upload.filename:
        try:
            return upload.read().decode("utf-8-sig"), None
        except UnicodeDecodeError:
            return "", f"文件 {upload.filename} 不是 UTF-8 文本，无法读取。"
    return request.form.get(field, ""), None


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        DATABASE=str(DEFAULT_DB),
        SEED_DEMO=True,
        MAX_RECORDS=MAX_RECORDS,
        TESTING=False,
    )
    if test_config:
        app.config.update(test_config)

    db.init_db(app.config["DATABASE"])
    if app.config.get("SEED_DEMO"):
        conn = db.connect(app.config["DATABASE"])
        try:
            seed_demo(conn)
        finally:
            conn.close()

    def get_conn() -> sqlite3.Connection:
        if "db" not in g:
            g.db = db.connect(app.config["DATABASE"])
        return g.db

    @app.teardown_appcontext
    def close_conn(_exc):
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    def render_index(**overrides):
        conn = get_conn()
        context = {
            "current": current_tables(conn),
            "records": _records_rows(conn),
            "batches": db.recent_batches(conn),
            "record_count": db.count_records(conn),
            "max_records": app.config["MAX_RECORDS"],
            "suggested_before": SUGGESTED_BEFORE,
            "suggested_after": SUGGESTED_AFTER,
            "demo_records": DEMO_RECORDS_TEXT,
            "error": None,
            "notice": None,
            "form": {
                "before": SUGGESTED_BEFORE,
                "after": SUGGESTED_AFTER,
                "note": "",
            },
        }
        context.update(overrides)
        status = 400 if context["error"] else 200
        return render_template("index.html", **context), status

    @app.get("/")
    def index():
        notice = None
        if request.args.get("imported"):
            notice = f"已导入 {request.args['imported']} 条试片记录。"
        elif request.args.get("confirmed"):
            notice = (
                f"换表已生效（批次 #{request.args['confirmed']}）。"
                "两张新表已同时启用，落选记录已标记待复核。"
            )
        elif request.args.get("skipped"):
            notice = "有重复的试片号被跳过。"
        return render_index(notice=notice)

    @app.post("/records/import")
    def import_records():
        conn = get_conn()
        raw, read_error = _read_text_field("records")
        if read_error:
            return render_index(error={"title": read_error, "details": []})
        existing = db.count_records(conn)
        try:
            rows = parse_records(raw)
            known = {
                str(row["coupon_no"])
                for row in conn.execute("SELECT coupon_no FROM records")
            }
            fresh = [row for row in rows if row[0] not in known]
            if existing + len(fresh) > app.config["MAX_RECORDS"]:
                raise ValidationError(
                    f"现有 {existing} 条，再导入 {len(fresh)} 条新记录将超过"
                    f"上限 {app.config['MAX_RECORDS']} 条，未导入任何内容。"
                )
        except ValidationError as exc:
            return render_index(error={"title": exc.message, "details": exc.details})

        inserted = 0
        with db.transaction(conn):
            for coupon, before_um, after_um in fresh:
                db.add_records(conn, [(coupon, before_um, after_um)])
                inserted += 1
        skipped = len(rows) - inserted
        target = url_for("index", imported=inserted)
        if skipped:
            target += f"&skipped={skipped}"
        return redirect(target)

    @app.post("/preview")
    def preview():
        conn = get_conn()
        before_text, before_error = _read_text_field("before")
        after_text, after_error = _read_text_field("after")
        note = request.form.get("note", "")
        read_error = before_error or after_error
        if read_error:
            return render_index(
                error={"title": read_error, "details": []},
                form={"before": before_text, "after": after_text, "note": note},
            )
        try:
            snapshot = create_preview(conn, before_text, after_text, note)
        except ValidationError as exc:
            return render_index(
                error={"title": exc.message, "details": exc.details},
                form={"before": before_text, "after": after_text, "note": note},
            )
        return render_template(
            "preview.html",
            snapshot=snapshot,
            payload=snapshot.payload,
            reason_labels=DISPLACED_REASON_LABELS,
            max_records=app.config["MAX_RECORDS"],
        )

    @app.get("/preview/<int:batch_id>")
    def view_preview(batch_id: int):
        conn = get_conn()
        snapshot = load_snapshot(conn, batch_id)
        if snapshot is None:
            return redirect(url_for("index"))
        return render_template(
            "preview.html",
            snapshot=snapshot,
            payload=snapshot.payload,
            reason_labels=DISPLACED_REASON_LABELS,
            max_records=app.config["MAX_RECORDS"],
        )

    @app.post("/preview/<int:batch_id>/confirm")
    def confirm(batch_id: int):
        conn = get_conn()
        token = request.form.get("token", "")
        try:
            result = confirm_batch(conn, batch_id, token)
        except (ConflictError, ValidationError) as exc:
            return (
                render_template(
                    "conflict.html",
                    message=exc.message,
                    details=exc.details,
                    snapshot=load_snapshot(conn, batch_id),
                ),
                409,
            )
        return redirect(url_for("index", confirmed=result.batch_id))

    @app.post("/demo/reset")
    def reset_demo():
        if not app.config.get("SEED_DEMO"):
            return redirect(url_for("index"))
        conn = get_conn()
        with db.transaction(conn):
            conn.execute(
                "UPDATE app_state SET generation = 0, current_batch_id = NULL"
                " WHERE id = 1"
            )
            for table in ("record_assignments", "bands", "batches", "records"):
                conn.execute(f"DELETE FROM {table}")
        seed_demo(conn)
        return redirect(url_for("index"))

    @app.get("/healthz")
    def healthz():
        conn = get_conn()
        conn.execute("SELECT 1").fetchone()
        return {"status": "ok"}

    @app.get("/records")
    def records():
        conn = get_conn()
        return render_template(
            "records.html",
            records=_records_rows(conn),
            record_count=db.count_records(conn),
            max_records=app.config["MAX_RECORDS"],
        )

    return app
