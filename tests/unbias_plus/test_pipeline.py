"""Tests for pipeline module."""

from unittest.mock import MagicMock, patch

import pytest

from unbias_plus.pipeline import UnBiasPlus
from unbias_plus.schema import BiasResult


@pytest.fixture
def mock_pipeline(sample_json: str) -> UnBiasPlus:
    """Return an UnBiasPlus instance with a mocked model.

    Parameters
    ----------
    sample_json : str
        Sample JSON string fixture.

    Returns
    -------
    UnBiasPlus
        Pipeline with mocked LLM — no GPU needed.

    """
    with patch("unbias_plus.pipeline.UnBiasModel") as mock_model_cls:
        mock_model = MagicMock()
        mock_model.generate.return_value = sample_json
        mock_model.model_name_or_path = "mock-model"
        mock_model_cls.return_value = mock_model
        pipe = UnBiasPlus("mock-model")
    return pipe


def test_analyze_returns_bias_result(
    mock_pipeline: UnBiasPlus,
    sample_text: str,
) -> None:
    """Test analyze returns a BiasResult object."""
    with patch.object(
        mock_pipeline._model, "generate", return_value=mock_pipeline._model.generate.return_value
    ):
        result = mock_pipeline.analyze(sample_text)
    assert isinstance(result, BiasResult)


def test_analyze_to_dict_returns_dict(
    mock_pipeline: UnBiasPlus,
    sample_text: str,
    sample_json: str,
) -> None:
    """Test analyze_to_dict returns a dictionary."""
    with patch.object(mock_pipeline._model, "generate", return_value=sample_json):
        result = mock_pipeline.analyze_to_dict(sample_text)
    assert isinstance(result, dict)
    assert "binary_label" in result


def test_analyze_to_json_returns_string(
    mock_pipeline: UnBiasPlus,
    sample_text: str,
    sample_json: str,
) -> None:
    """Test analyze_to_json returns a JSON string."""
    with patch.object(mock_pipeline._model, "generate", return_value=sample_json):
        result = mock_pipeline.analyze_to_json(sample_text)
    assert isinstance(result, str)


def test_analyze_to_cli_returns_string(
    mock_pipeline: UnBiasPlus,
    sample_text: str,
    sample_json: str,
) -> None:
    """Test analyze_to_cli returns a formatted string."""
    with patch.object(mock_pipeline._model, "generate", return_value=sample_json):
        result = mock_pipeline.analyze_to_cli(sample_text)
    assert isinstance(result, str)
    assert "BIASED" in result


def test_analyze_raises_on_bad_output(
    mock_pipeline: UnBiasPlus,
    sample_text: str,
) -> None:
    """Test analyze raises ValueError when LLM returns invalid output."""
    with patch.object(
        mock_pipeline._model, "generate", return_value="not valid json"
    ):
        with pytest.raises(ValueError):
            mock_pipeline.analyze(sample_text)
