"""Tests for Langfuse client (without actual Langfuse server)."""

from unittest.mock import MagicMock

from rag.observability.langfuse_client import LangfuseTracer


class TestLangfuseTracerDisabled:
    """Test behavior when tracing is disabled."""

    def test_disabled_when_enable_tracing_false(self):
        """Tracer should be disabled when enable_tracing=False."""
        settings = MagicMock()
        settings.enable_tracing = False

        tracer = LangfuseTracer(settings)
        assert tracer.is_enabled is False

    def test_disabled_when_keys_missing(self):
        """Tracer should be disabled when API keys are not set."""
        settings = MagicMock()
        settings.enable_tracing = True
        settings.langfuse_public_key = None
        settings.langfuse_secret_key = None

        tracer = LangfuseTracer(settings)
        assert tracer.is_enabled is False

    def test_create_trace_returns_none_when_disabled(self):
        settings = MagicMock()
        settings.enable_tracing = False

        tracer = LangfuseTracer(settings)
        result = tracer.create_trace("test query")
        assert result is None

    def test_create_span_returns_none_when_disabled(self):
        settings = MagicMock()
        settings.enable_tracing = False

        tracer = LangfuseTracer(settings)
        result = tracer.create_span(None, "test_span")
        assert result is None

    def test_score_trace_does_nothing_when_disabled(self):
        settings = MagicMock()
        settings.enable_tracing = False

        tracer = LangfuseTracer(settings)
        # Should not raise any error
        tracer.score_trace(None, "test", 0.5)

    def test_flush_does_nothing_when_disabled(self):
        settings = MagicMock()
        settings.enable_tracing = False

        tracer = LangfuseTracer(settings)
        # Should not raise any error
        tracer.flush()
