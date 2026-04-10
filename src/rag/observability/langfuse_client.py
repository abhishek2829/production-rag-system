"""Langfuse client — manages connection to the Langfuse observability server.

WHAT THIS FILE DOES:
Connects to Langfuse and provides simple methods to record traces and spans.

LANGFUSE v4 API:
In v4, the main method is start_observation() which creates both traces
and spans. It uses a context manager pattern:

  with langfuse.start_observation(name="my-trace") as trace:
      # everything inside is part of this trace
      with langfuse.start_observation(name="step1") as span:
          # this is a span within the trace
          do_work()

For our use case, we use a simpler approach: create observations
and manually end them (so we can control timing precisely).
"""

from __future__ import annotations

import logging

from langfuse import Langfuse

from rag.config import Settings

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Manages the connection to Langfuse for recording traces and spans."""

    def __init__(self, settings: Settings) -> None:
        self._enabled = False
        self._client: Langfuse | None = None

        if not settings.enable_tracing:
            logger.info("Tracing disabled (RAG_ENABLE_TRACING=false)")
            return

        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            logger.warning("Tracing enabled but Langfuse keys not set. Skipping.")
            return

        try:
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            self._enabled = True
            logger.info("Langfuse tracer connected: host=%s", settings.langfuse_host)
        except Exception as e:
            logger.warning("Failed to connect to Langfuse: %s. Tracing disabled.", e)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def create_trace(self, query: str, name: str = "rag-query") -> object | None:
        """Start a new trace for a query.

        In Langfuse v4, a trace is created using start_observation()
        which returns an observation object we can attach child spans to.
        """
        if not self._enabled or not self._client:
            return None

        try:
            # In v4, we use start_observation with as_type="span" at the top level
            # This creates a trace automatically and returns a span we can nest under
            trace = self._client.start_observation(
                name=name,
                input={"query": query},
                metadata={"source": "traced_pipeline"},
            )
            logger.debug("Created trace for: '%s...'", query[:50])
            return trace
        except Exception as e:
            logger.warning("Failed to create trace: %s", e)
            return None

    def create_span(
        self,
        trace: object | None,
        name: str,
        input_data: dict | None = None,
        output_data: dict | None = None,
        duration_seconds: float = 0.0,
        metadata: dict | None = None,
    ) -> object | None:
        """Record one step (span) within a trace."""
        if not self._enabled or trace is None or not self._client:
            return None

        try:
            # Create a child span under the current trace context
            span = self._client.start_observation(
                name=name,
                input=input_data or {},
                metadata={
                    "duration_seconds": duration_seconds,
                    **(metadata or {}),
                },
            )
            # In v4: update() sets the output, end() marks it as finished
            span.update(output=output_data or {})
            span.end()
            logger.debug("Created span '%s' (%.3fs)", name, duration_seconds)
            return span
        except Exception as e:
            logger.warning("Failed to create span '%s': %s", name, e)
            return None

    def score_trace(
        self,
        trace: object | None,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        """Attach a quality score to a trace."""
        if not self._enabled or trace is None or not self._client:
            return

        try:
            # Get the trace_id from the observation
            trace_id = getattr(trace, "trace_id", None)
            if trace_id:
                self._client.create_score(
                    name=name,
                    value=value,
                    trace_id=trace_id,
                    comment=comment,
                    data_type="NUMERIC",
                )
                logger.debug("Scored trace: %s=%.3f", name, value)
        except Exception as e:
            logger.warning("Failed to score trace (%s): %s", name, e)

    def update_trace(
        self,
        trace: object | None,
        output: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Update a trace with final output."""
        if not self._enabled or trace is None:
            return

        try:
            trace.update(output=output or {}, metadata=metadata or {})
            trace.end()
        except Exception as e:
            logger.warning("Failed to update trace: %s", e)

    def flush(self) -> None:
        """Force send all pending traces to Langfuse."""
        if self._client:
            try:
                self._client.flush()
                logger.info("Langfuse traces flushed")
            except Exception as e:
                logger.warning("Failed to flush Langfuse: %s", e)
