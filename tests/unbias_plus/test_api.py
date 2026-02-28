"""Tests for API module."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from unbias_plus.api import app
from unbias_plus.schema import BiasedSegment, BiasResult


@pytest.fixture
def sample_bias_result() -> BiasResult:
    """Return a sample BiasResult for API tests.

    Returns
    -------
    BiasResult
        A sample bias result for mocking pipeline output.

    """
    return BiasResult(
        binary_label="biased",
        severity=3,
        bias_found=True,
        biased_segments=[
            BiasedSegment(
                original="desperate for clicks",
                replacement="seeking audience engagement",
                severity="medium",
                bias_type="loaded language",
                reasoning="Pejorative motive attributed without evidence.",
            )
        ],
        unbiased_text="Some established media outlets seek audience engagement.",
    )


@pytest.fixture
def client(sample_bias_result: BiasResult) -> TestClient:
    """Return a TestClient with mocked pipeline.

    Parameters
    ----------
    sample_bias_result : BiasResult
        Sample result to return from mocked pipeline.

    Returns
    -------
    TestClient
        FastAPI test client with mocked UnBiasPlus.

    """
    mock_pipe = MagicMock()
    mock_pipe.analyze.return_value = sample_bias_result
    mock_pipe._model.model_name_or_path = "mock-model"

    with patch("unbias_plus.api.UnBiasPlus", return_value=mock_pipe):
        yield TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """Test /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_endpoint_returns_200(client: TestClient) -> None:
    """Test /analyze returns 200 for valid request."""
    response = client.post(
        "/analyze",
        json={"text": "Women are too emotional to lead."},
    )
    assert response.status_code == 200


def test_analyze_endpoint_returns_bias_result(client: TestClient) -> None:
    """Test /analyze response contains expected BiasResult fields."""
    response = client.post(
        "/analyze",
        json={"text": "Women are too emotional to lead."},
    )
    data = response.json()
    assert "binary_label" in data
    assert "severity" in data
    assert "bias_found" in data
    assert "biased_segments" in data
    assert "unbiased_text" in data


def test_analyze_endpoint_missing_text(client: TestClient) -> None:
    """Test /analyze returns 422 when text field is missing."""
    response = client.post("/analyze", json={})
    assert response.status_code == 422


def test_analyze_endpoint_model_not_loaded() -> None:
    """Test /analyze returns 500 when model is not loaded."""
    with patch("unbias_plus.api.UnBiasPlus", return_value=MagicMock()):
        test_client = TestClient(app)
        app.state.pipe = None
        response = test_client.post(
            "/analyze",
            json={"text": "test"},
        )
    assert response.status_code == 500
