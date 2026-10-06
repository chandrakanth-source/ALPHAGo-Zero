from fastapi.testclient import TestClient

from web import server


def _start(client):
    r = client.post("/api/new_game", json={"board_size": 12, "human_color": 1, "level": "easy"})
    assert r.status_code == 200, r.text


def test_export_sgf_contains_moves():
    client = TestClient(server.app)
    _start(client)
    assert client.post("/api/move", json={"row": 3, "col": 4}).status_code == 200
    r = client.get("/api/export_sgf")
    assert r.status_code == 200
    assert r.text.startswith("(;GM[1]") and "SZ[12]" in r.text and ";B[ed]" in r.text


def test_hint_returns_heatmap():
    client = TestClient(server.app)
    _start(client)
    data = client.post("/api/hint").json()
    assert data["top_moves"], "expected MCTS visit distribution"
    assert abs(sum(m["prob"] for m in data["top_moves"]) - 1.0) < 0.05
