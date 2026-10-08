import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
from fastapi.testclient import TestClient
from app.main import app

def test_api():
    with TestClient(app) as c:
        s = c.get("/api/summary").json()
        assert s["registered"] == 600 and s["conflicts"] > 0
        m = c.get("/api/drives/D01/match").json()
        assert m["rows"] and "explanation" in m["rows"][0]
        r = c.post("/api/scheduler/resolve").json()
        assert r["conflicts"] == []
        o = c.get("/api/offers").json()["rows"][0]
        assert c.post(f"/api/offers/{o['id']}", json={"status": "Accepted"}).json()["status"] == "Accepted"
        assert c.get("/api/students/S0001").json()["recommendations"]
        assert c.get("/api/metrics").json()["matching"]["lift_over_random"] > 2
