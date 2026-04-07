"""Answer generation with citation enforcement using Anthropic Claude.

The generator takes retrieved chunks and produces a cited answer.
Prompts are loaded from version-controlled YAML files via the PromptManager.

Key design decisions:
- Uses Anthropic Claude — shows multi-provider capability
- Prompts are versioned — every response records which prompt version was used
- System prompt enforces citation rules and refusal when unsupported
- Each source chunk is numbered for traceable citations [Source N]
- Temperature=0 for deterministic, reproducible outputs
"""

from __future__ import annotations

import logging
from pathlib import Path

from anthropic import Anthropic

from rag.config import Settings
from rag.generation.prompt_manager import PromptConfig, load_prompt_config
from rag.models import RAGResponse, RetrievedChunk

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates cited answers from retrieved chunks using Anthropic Claude.

    Uses the Anthropic SDK directly for the generation step.
    Loads versioned prompts from YAML config files.
    """

    def __init__(self, settings: Settings, prompts_path: Path | None = None) -> None:
        self._settings = settings
        self._client = Anthropic(api_key=settings.anthropic_api_key)

        # Load versioned prompt config
        if prompts_path is None:
            prompts_path = Path("configs/prompts.yaml")

        self._prompt_config = load_prompt_config(prompts_path)

        logger.info(
            "Generator initialized: model=%s, prompt_version=%s, prompt_hash=%s",
            settings.llm_model,
            self._prompt_config.version,
            self._prompt_config.content_hash,
        )

    @property
    def prompt_config(self) -> PromptConfig:
        """Current prompt configuration (for logging/tracing)."""
        return self._prompt_config

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into numbered source blocks."""
        template = self._prompt_config.context_chunk_template
        sections = []
        for i, rc in enumerate(chunks, start=1):
            section = template.format(
                index=i,
                source=rc.chunk.source,
                content=rc.chunk.content,
            )
            sections.append(section.strip())
        return "\n\n".join(sections)

    def generate(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> RAGResponse:
        """Generate a cited answer for the query using retrieved chunks.

        Args:
            query: The user's question.
            retrieved_chunks: Chunks retrieved from the vector store.

        Returns:
            RAGResponse with the answer, citations, and metadata.
        """
        if not retrieved_chunks:
            return RAGResponse(
                answer=(
                    "I don't have any documents to search through. "
                    "Please ingest documents first."
                ),
                citations=[],
                query=query,
                retrieved_chunks=[],
            )

        # Format the context from retrieved chunks
        context = self._format_context(retrieved_chunks)

        # Build the user prompt from versioned template
        user_prompt = self._prompt_config.user_prompt_template.format(
            context=context,
            question=query,
        )

        # Call Anthropic Claude
        logger.info("Generating answer for: '%s...'", query[:50])
        response = self._client.messages.create(
            model=self._settings.llm_model,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
            system=self._prompt_config.system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = response.content[0].text

        # Parse which sources were actually cited in the answer
        citations = self._extract_citations(answer, retrieved_chunks)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        logger.info(
            "Generated answer: %d chars, %d citations, %d+%d tokens (in+out), "
            "prompt_version=%s",
            len(answer),
            len(citations),
            input_tokens,
            output_tokens,
            self._prompt_config.version,
        )

        return RAGResponse(
            answer=answer,
            citations=citations,
            query=query,
            retrieved_chunks=retrieved_chunks,
        )

    def _extract_citations(
        self, answer: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Extract which source chunks were actually cited in the answer.

        Looks for [Source N] patterns and maps them back to retrieved chunks.
        """
        cited: list[RetrievedChunk] = []
        for i, chunk in enumerate(chunks, start=1):
            if f"[Source {i}]" in answer:
                cited.append(chunk)
        return cited
