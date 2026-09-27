def test_argo_validation(client):
    params = {"date_from": "2024-01-01", "date_to": "2026-06-30"}
    res1 = client.get("/api/v1/argo/validation", params=params)
    assert res1.status_code == 200
    data1 = res1.json()

    assert data1["mode"] == "mock"
    assert data1["n_matchups"] == 5
    assert "overall" in data1
    assert len(data1["by_depth"]) == 15

    # Determinism check
    res2 = client.get("/api/v1/argo/validation", params=params)
    assert res2.status_code == 200
    assert res1.json() == res2.json()


def test_argo_profiles_success(client):
    params = {"lat": 12.5, "lon": 65.0, "radius_km": 100.0, "date": "2025-05-15"}
    res1 = client.get("/api/v1/argo/profiles", params=params)
    assert res1.status_code == 200
    data1 = res1.json()

    assert data1["mode"] == "mock"
    assert len(data1["profiles"]) >= 3
    float_ids = [p["float_id"] for p in data1["profiles"]]
    assert "ARGO_TEST_001" in float_ids
    assert "ARGO_TEST_002" in float_ids
    assert "ARGO_TEST_003" in float_ids

    # Determinism check
    res2 = client.get("/api/v1/argo/profiles", params=params)
    assert res2.status_code == 200
    assert res1.json() == res2.json()


def test_argo_profiles_out_of_domain(client):
    res = client.get("/api/v1/argo/profiles", params={"lat": 20.0, "lon": 78.0, "date": "2025-05-15"})
    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "OUT_OF_DOMAIN"
