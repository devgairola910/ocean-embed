def test_transect_success(client):
    params = {
        "lat1": 10.0,
        "lon1": 65.0,
        "lat2": 15.0,
        "lon2": 75.0,
        "date": "2025-05-15",
        "n_points": 10,
    }
    res1 = client.get("/api/v1/transect", params=params)
    assert res1.status_code == 200
    data1 = res1.json()

    assert data1["mode"] == "mock"
    assert data1["n_points"] == 10
    assert len(data1["points"]) == 10
    assert data1["points"][0]["distance_km"] == 0.0
    assert data1["points"][-1]["distance_km"] > 0.0

    # Determinism check
    res2 = client.get("/api/v1/transect", params=params)
    assert res2.status_code == 200
    assert res1.json() == res2.json()


def test_transect_out_of_domain(client):
    params = {
        "lat1": 22.0,
        "lon1": 78.0,
        "lat2": 23.0,
        "lon2": 79.0,
        "date": "2025-05-15",
        "n_points": 10,
    }
    res = client.get("/api/v1/transect", params=params)
    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "OUT_OF_DOMAIN"
