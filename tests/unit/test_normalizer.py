"""
SAHA – Unit tests for NormalizationPipeline (§1.4).
"""
import pytest

from saha.eval.normalizer import NormalizationPipeline


@pytest.fixture
def pipeline() -> NormalizationPipeline:
    return NormalizationPipeline()


class TestNormalizationPipeline:
    def test_strips_provider_metadata(self, pipeline: NormalizationPipeline) -> None:
        raw = {
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "model": "claude-3-5-sonnet",
            "type": "message",
            "role": "assistant",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "text": "Hello, world!",
        }
        normalized, err = pipeline.normalize(raw, provider_id="claude")
        assert err is None
        assert "id" not in normalized
        assert "model" not in normalized
        assert "usage" not in normalized
        assert "text" in normalized

    def test_guarantees_text_key(self, pipeline: NormalizationPipeline) -> None:
        raw = {"content": "Some answer"}
        normalized, err = pipeline.normalize(raw)
        assert err is None
        assert normalized["text"] == "Some answer"

    def test_text_from_message_alias(self, pipeline: NormalizationPipeline) -> None:
        raw = {"message": "Response via alias"}
        normalized, err = pipeline.normalize(raw)
        assert normalized["text"] == "Response via alias"

    def test_empty_input_yields_empty_text(self, pipeline: NormalizationPipeline) -> None:
        normalized, err = pipeline.normalize({})
        assert err is None
        assert normalized["text"] == ""

    def test_none_values_become_empty_string(self, pipeline: NormalizationPipeline) -> None:
        raw = {"text": None, "other": None}
        normalized, err = pipeline.normalize(raw)
        assert normalized["text"] == ""
        assert normalized["other"] == ""

    def test_sorts_string_lists(self, pipeline: NormalizationPipeline) -> None:
        raw = {"tags": ["zebra", "apple", "mango"]}
        normalized, err = pipeline.normalize(raw)
        assert normalized["tags"] == ["apple", "mango", "zebra"]

    def test_nested_dict_canonicalized(self, pipeline: NormalizationPipeline) -> None:
        raw = {"meta": {"key": None, "val": "x"}}
        normalized, err = pipeline.normalize(raw)
        assert normalized["meta"]["key"] == ""
        assert normalized["meta"]["val"] == "x"

    def test_error_on_exception(self, pipeline: NormalizationPipeline) -> None:
        # Force an internal exception by passing non-dict (monkeypatching _strip_metadata)
        original = pipeline._strip_metadata
        def bad_strip(raw):
            raise ValueError("forced error")
        pipeline._strip_metadata = bad_strip

        _, err = pipeline.normalize({"text": "hi"}, provider_id="test")
        assert err is not None
        assert "NORMALIZATION_FAILED" in err.code
        pipeline._strip_metadata = original
