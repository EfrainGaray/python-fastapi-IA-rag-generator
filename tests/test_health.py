def test_health_ok(client):
    # mock_es.ping() returns a MagicMock (truthy); mock embedding_model.encode("") works fine
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "checks" in data
    assert data["checks"]["elasticsearch"] == "ok"
    assert data["checks"]["embedding_model"] == "ok"
    assert data["version"] == "2.0.0"


def test_health_degraded_when_es_down(client):
    import app.main as main_module
    from unittest.mock import MagicMock

    bad_es = MagicMock()
    bad_es.ping.side_effect = Exception("Connection refused")
    original_es = main_module.app.state.es
    main_module.app.state.es = bad_es
    try:
        response = client.get("/health")
    finally:
        main_module.app.state.es = original_es

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["checks"]["elasticsearch"] == "error"
