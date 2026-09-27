def test_ohc_product(client):
    params = {"lat": 12.5, "lon": 65.0, "date": "2025-05-15"}
    res1 = client.get("/api/v1/products/ohc", params=params)
    assert res1.status_code == 200
    data1 = res1.json()

    assert data1["mode"] == "mock"
    assert data1["unit"] == "GJ/m^2"
    assert data1["ohc_value"] > 0

    # Determinism check
    res2 = client.get("/api/v1/products/ohc", params=params)
    assert res2.status_code == 200
    assert res1.json() == res2.json()


def test_marine_heatwave_product(client):
    params = {"lat": 12.5, "lon": 65.0, "date": "2025-05-15"}
    res1 = client.get("/api/v1/products/marine-heatwave", params=params)
    assert res1.status_code == 200
    data1 = res1.json()

    assert data1["mode"] == "mock"
    assert "is_heatwave" in data1
    assert data1["climatological_threshold_c"] == 29.5

    # Determinism check
    res2 = client.get("/api/v1/products/marine-heatwave", params=params)
    assert res2.status_code == 200
    assert res1.json() == res2.json()


def test_cyclone_risk_product(client):
    params = {"lat": 12.5, "lon": 65.0, "date": "2025-05-15"}
    res1 = client.get("/api/v1/products/cyclone-risk", params=params)
    assert res1.status_code == 200
    data1 = res1.json()

    assert data1["mode"] == "mock"
    assert "tchp_score_kj_cm2" in data1
    assert data1["risk_category"] in ["Low", "Moderate", "High"]

    # Determinism check
    res2 = client.get("/api/v1/products/cyclone-risk", params=params)
    assert res2.status_code == 200
    assert res1.json() == res2.json()


def test_product_out_of_domain(client):
    res = client.get("/api/v1/products/ohc", params={"lat": 20.0, "lon": 78.0, "date": "2025-05-15"})
    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "OUT_OF_DOMAIN"
