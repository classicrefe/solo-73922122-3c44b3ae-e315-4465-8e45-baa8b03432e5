"""HTTP-level tests for the Flask page."""

import re

import pytest

from app.web import SUGGESTED_AFTER, SUGGESTED_BEFORE, create_app


@pytest.fixture
def app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "SEED_DEMO": False,
            "DATABASE": str(tmp_path / "web.db"),
        }
    )


@pytest.fixture
def client(app):
    return app.test_client()


def _token(html: str) -> str:
    return re.search(r'name="token" value="([^"]+)"', html).group(1)


def test_empty_state_and_import(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "尚未启用任何分档表" in response.data.decode()

    response = client.post(
        "/records/import",
        data={"records": "A-1 5 5\nA-2 15 15\nA-3 15 15\n"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "已导入 3 条试片记录" in response.data.decode()


def test_import_is_atomic_on_bad_line(client):
    client.post("/records/import", data={"records": "A-1 5 5\n"})
    response = client.post(
        "/records/import", data={"records": "A-2 6 6\nbroken\n"}
    )
    assert response.status_code == 400
    assert "未导入任何内容" in response.data.decode()
    records = client.get("/records").data.decode()
    assert "A-1" in records and "A-2" not in records


def test_import_skips_duplicate_coupons(client):
    client.post("/records/import", data={"records": "A-1 5 5\n"})
    client.post(
        "/records/import", data={"records": "A-1 9 9\nA-2 15 15\n"}
    )
    page = client.get("/records").data.decode()
    assert page.count("A-1") == 1
    assert "A-2" in page


def test_import_respects_sixty_record_cap(client):
    rows = "\n".join(f"R-{i:03d} 5 5" for i in range(60))
    response = client.post("/records/import", data={"records": rows})
    assert response.status_code == 302

    response = client.post("/records/import", data={"records": "R-999 5 5\n"})
    assert response.status_code == 400
    assert "超过上限 60 条" in response.data.decode()
    assert "R-999" not in client.get("/records").data.decode()


def test_uploaded_files_are_accepted(client):
    import io

    response = client.post(
        "/records/import",
        data={"records_file": (io.BytesIO(b"F-1 5 5\nF-2 15 15\n"), "records.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 302

    response = client.post(
        "/preview",
        data={
            "before_file": (io.BytesIO(b"0-10 1\n11-20 1\n"), "before.txt"),
            "after_file": (io.BytesIO(b"0-10 1\n11-20 1\n"), "after.txt"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert "最多可保留 <strong>2</strong>" in response.data.decode()


def test_non_utf8_upload_is_reported(client):
    import io

    response = client.post(
        "/preview",
        data={
            "before_file": (io.BytesIO(b"\xff\xfe\x00bad"), "before.txt"),
            "after": "0-10 1\n",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "不是 UTF-8" in response.data.decode()


def test_preview_rejects_unequal_totals(client):
    client.post("/records/import", data={"records": "A-1 5 5\nA-2 15 15\n"})
    response = client.post(
        "/preview", data={"before": "0-10 1\n", "after": "0-20 3\n"}
    )
    assert response.status_code == 400
    assert "总件数不一致" in response.data.decode()


def test_preview_rejects_overlap(client):
    client.post("/records/import", data={"records": "A-1 5 5\n"})
    response = client.post(
        "/preview", data={"before": "0-10 1\n10-20 1\n", "after": "0-20 2\n"}
    )
    assert response.status_code == 400
    assert "重叠" in response.data.decode()


def test_full_swap_flow(client):
    client.post(
        "/records/import", data={"records": "A-1 5 5\nA-2 15 15\nA-3 15 15\n"}
    )
    response = client.post(
        "/preview",
        data={"before": "0-10 1\n11-20 1\n", "after": "0-10 1\n11-20 1\n"},
    )
    assert response.status_code == 200
    html = response.data.decode()
    assert "最多可保留 <strong>2</strong>" in html
    batch_id = re.search(r"/preview/(\d+)/confirm", html).group(1)

    response = client.post(
        f"/preview/{batch_id}/confirm", data={"token": _token(html)}
    )
    assert response.status_code == 302
    assert "confirmed=" in response.headers["Location"]

    page = client.get("/?confirmed=1").data.decode()
    assert "换表已生效" in page
    assert page.count("tag active") == 2
    assert page.count("tag needs_review") == 1


def test_bad_token_keeps_old_state(client):
    client.post(
        "/records/import", data={"records": "A-1 5 5\nA-2 15 15\n"}
    )
    client.post(
        "/preview", data={"before": "0-10 1\n11-20 1\n", "after": "0-10 1\n11-20 1\n"}
    )
    html = client.post(
        "/preview", data={"before": "0-10 1\n", "after": "0-10 1\n"}
    ).data.decode()
    batch_id = re.search(r"/preview/(\d+)/confirm", html).group(1)

    response = client.post(f"/preview/{batch_id}/confirm", data={"token": "nope"})
    assert response.status_code == 409
    page = client.get("/").data.decode()
    assert "换表已生效" not in page


def test_demo_mode_seeds_and_resets(tmp_path):
    app = create_app(
        {"TESTING": True, "SEED_DEMO": True, "DATABASE": str(tmp_path / "demo.db")}
    )
    client = app.test_client()
    page = client.get("/").data.decode()
    assert "试片 60 / 60 条" in page
    assert "已占" in page

    response = client.post("/demo/reset")
    assert response.status_code == 302
    assert client.get("/").status_code == 200


def test_suggested_tables_preview_retains_53(tmp_path):
    demo_app = create_app(
        {"TESTING": True, "SEED_DEMO": True, "DATABASE": str(tmp_path / "demo.db")}
    )
    client = demo_app.test_client()
    response = client.post(
        "/preview",
        data={"before": SUGGESTED_BEFORE, "after": SUGGESTED_AFTER},
    )
    html = response.data.decode()
    assert "最多可保留 <strong>53</strong>" in html
    assert "落选转待复核</span><strong>7</strong>" in html
    assert "烘烤前读数不在新表任何档内" in html
    assert "所在档件数已满" in html
