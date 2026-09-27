def test_predict_success(client):
    params = {"lat": 12.5, "lon": 65.0, "date": "2025-05-15"}
    res1 = client.get("/api/v1/predict", params=params)
    assert res1.status_code == 200
    data1 = res1.json()

    assert data1["mode"] == "mock"
    assert data1["date"] == "2025-05-15"
    assert data1["requested_location"] == {"lat": 12.5, "lon": 65.0}
    assert len(data1["profile"]) == 15
    assert "sst" in data1["surface_inputs_used"]

    # Determinism check
    res2 = client.get("/api/v1/predict", params=params)
    assert res2.status_code == 200
    assert res1.json() == res2.json()

    # Thermocline monotonic decrease check
    temps = [p["temperature_c"] for p in data1["profile"]]
    assert temps[0] > temps[-1]


def test_predict_out_of_domain_land(client):
    res = client.get("/api/v1/predict", params={"lat": 20.0, "lon": 78.0, "date": "2025-05-15"})
    assert res.status_code == 400
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "OUT_OF_DOMAIN"


def test_predict_out_of_domain_bounds(client):
    res = client.get("/api/v1/predict", params={"lat": 35.0, "lon": 80.0, "date": "2025-05-15"})
    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "OUT_OF_DOMAIN"
