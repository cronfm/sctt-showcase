import pytest
from fastapi.testclient import TestClient

from sctt_showcase.api import app


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_end_to_end_example_and_figure(client):
    request = client.get("/api/example").json()
    response = client.post("/api/infer", json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["position"]) == 3
    assert result["model"]["status"] == "synthetic-trained"
    assert len(result["candidate_positions"]) == request["top_k"]
    chart = client.post("/api/figure", json={**request, "color_mode": "feature"})
    assert chart.status_code == 200, chart.text
    assert any(trace["type"] == "scatter3d" for trace in chart.json()["data"])


def test_invalid_requests(client):
    assert client.post("/api/infer", json={"top_k": 99}).status_code == 422
    assert client.post("/api/infer", json={"sample_id": "missing"}).status_code == 404
    assert client.post("/api/infer", json={"candidate_offset_m": [1, 2]}).status_code == 422
    assert client.post("/api/infer", json={"candidate_offset_m": ["NaN", 0, 0]}).status_code == 422
    assert client.post("/api/figure", json={"backend": "not-a-backend"}).status_code == 422


def test_large_body_rejected_with_and_without_length(client):
    body = b" " * 262_145
    assert client.post("/api/infer", content=body).status_code == 413
    assert client.post("/api/infer", content=iter([body[:200_000], body[200_000:]])).status_code == 413


@pytest.mark.parametrize("value", ["1e309", "NaN", "Infinity", "-Infinity"])
def test_raw_nonfinite_json_is_validation_error(client, value):
    response = client.post(
        "/api/infer",
        content='{"candidate_offset_m":[' + value + ",0,0]}",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "candidate_offset_m", 0]


def test_assets_and_documentation(client):
    assert client.get("/").status_code == 200
    assert "plotly" in client.get("/vendor/plotly.min.js").text.lower()
    assert client.get("/openapi.json").json()["info"]["title"] == "SCTT Showcase API"
    assert client.get("/api/health").json()["synthetic"] is True
    assert len(client.get("/api/samples").json()) >= 2
    assert client.get("/static/../../pyproject.toml").status_code == 404
