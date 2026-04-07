"""Prompt versioning and management.

WHY VERSION PROMPTS?
Imagine this scenario:
- Monday: Your RAG system has 90% faithfulness score
- Tuesday: You change the system prompt to "be more concise"
- Wednesday: Faithfulness drops to 70% — but you don't know why

With prompt versioning, every response records WHICH prompt version was used.
When quality drops, you can trace it back to the exact prompt change.

This is the same idea as version control for code (Git), but for prompts.
In the AI world, prompts ARE code — they change behavior just like code does.

HOW IT WORKS:
- Each prompt config YAML file has a "version" field
- Every RAG response records the prompt version that was used
- In Phase 3, CI will compare quality across prompt versions

INTERVIEW QUESTION: "How do you manage prompt changes in production?"
ANSWER: "We version-control prompts in YAML files alongside the code.
Each response logs which prompt version generated it. CI gates prevent
deploying prompt changes that degrade quality below our thresholds."
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptConfig:
    """A versioned prompt configuration.

    Attributes:
        version: Human-readable version string (e.g., "1.0", "1.1").
        system_prompt: The system prompt sent to the LLM.
        user_prompt_template: Template for the user message (has {context} and {question}).
        context_chunk_template: Template for formatting each source chunk.
        content_hash: Auto-generated hash of the prompt content.
                      Even if someone forgets to bump the version number,
                      we can detect changes by comparing hashes.
    """

    version: str
    system_prompt: str
    user_prompt_template: str
    context_chunk_template: str
    content_hash: str


def load_prompt_config(path: Path) -> PromptConfig:
    """Load a versioned prompt configuration from a YAML file.

    The YAML file should have these fields:
        version: "1.0"
        system_prompt: "..."
        user_prompt_template: "..."
        context_chunk_template: "..."

    Args:
        path: Path to the YAML prompt config file.

    Returns:
        PromptConfig with version and content hash.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    # Generate a content hash — this catches changes even if version isn't bumped
    content = (
        data.get("system_prompt", "")
        + data.get("user_prompt_template", "")
        + data.get("context_chunk_template", "")
    )
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]

    config = PromptConfig(
        version=data.get("version", "unknown"),
        system_prompt=data["system_prompt"],
        user_prompt_template=data["user_prompt_template"],
        context_chunk_template=data["context_chunk_template"],
        content_hash=content_hash,
    )

    logger.info(
        "Loaded prompt config: version=%s, hash=%s, path=%s",
        config.version,
        config.content_hash,
        path,
    )

    return config
