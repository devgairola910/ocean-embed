def test_grid_success(client):
    params = {"date": "2025-05-15", "depth": 0.0}
    res1 = client.get("/api/v1/grid", params=params)
    assert res1.status_code == 200
    data1 = res1.json()

    assert data1["mode"] == "mock"
    assert data1["depth_m"] == 0.0
    assert len(data1["lats"]) > 0
    assert len(data1["lons"]) > 0
    assert len(data1["temperature_c"]) == len(data1["lats"])
    assert len(data1["temperature_c"][0]) == len(data1["lons"])

    # Determinism check
    res2 = client.get("/api/v1/grid", params=params)
    assert res2.status_code == 200
    assert res1.json() == res2.json()


def test_grid_invalid_depth(client):
    res = client.get("/api/v1/grid", params={"date": "2025-05-15", "depth": 12.3})
    assert res.status_code == 400
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_DEPTH"
