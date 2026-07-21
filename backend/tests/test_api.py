import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Multimodal TB Detection System API",
        "version": "3.1.0",
        "status": "operational"
    }

def test_get_dashboard_stats():
    response = client.get("/api/stats/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "chartData" in data
    assert "ageData" in data

def test_invalid_route():
    response = client.get("/invalid")
    assert response.status_code == 404
