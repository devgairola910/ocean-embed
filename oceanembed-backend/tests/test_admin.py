def test_admin_trigger_and_status(client):
    res_trigger = client.post("/api/v1/admin/ingest/trigger?date=2025-05-15")
    assert res_trigger.status_code == 200
    trigger_data = res_trigger.json()

    assert "job_id" in trigger_data
    job_id = trigger_data["job_id"]
    assert trigger_data["status"] == "running"

    res_status = client.get(f"/api/v1/admin/ingest/status/{job_id}")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] in ["running", "succeeded"]


def test_admin_status_not_found(client):
    res = client.get("/api/v1/admin/ingest/status/NON_EXISTENT_JOB")
    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "JOB_NOT_FOUND"
