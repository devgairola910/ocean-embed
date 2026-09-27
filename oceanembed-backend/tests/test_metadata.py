def test_metadata_endpoint(client):
    res = client.get("/api/v1/metadata")
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "mock"
    assert data["model_version"] == "0.1.0-mock"
    assert data["domain_bounds"] == {
        "lat_min": 0.0,
        "lat_max": 25.0,
        "lon_min": 45.0,
        "lon_max": 100.0,
        "resolution_deg": 0.25,
    }
    assert len(data["depth_levels_m"]) == 15
    assert data["valid_date_range"] == {"start": "2024-01-01", "end": "2026-06-30"}
