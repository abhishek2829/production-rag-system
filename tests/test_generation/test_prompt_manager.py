"""Tests for prompt versioning."""

from pathlib import Path

from rag.generation.prompt_manager import load_prompt_config


class TestPromptManager:
    def test_loads_prompt_config(self, tmp_path: Path):
        """Should load version, prompts, and generate content hash."""
        config_file = tmp_path / "prompts.yaml"
        config_file.write_text(
            'version: "2.0"\n'
            "system_prompt: You are a helpful assistant.\n"
            "user_prompt_template: Answer this\n"
            "context_chunk_template: Source content\n"
        )

        config = load_prompt_config(config_file)

        assert config.version == "2.0"
        assert "helpful assistant" in config.system_prompt
        assert "Answer this" in config.user_prompt_template
        assert config.content_hash  # should be a non-empty string

    def test_hash_changes_when_content_changes(self, tmp_path: Path):
        """Different prompt content should produce different hashes."""
        config1 = tmp_path / "v1.yaml"
        config1.write_text(
            'version: "1.0"\n'
            "system_prompt: Be helpful.\n"
            "user_prompt_template: '{question}'\n"
            "context_chunk_template: '{content}'\n"
        )

        config2 = tmp_path / "v2.yaml"
        config2.write_text(
            'version: "1.0"\n'  # Same version but different content!
            "system_prompt: Be very helpful and precise.\n"
            "user_prompt_template: '{question}'\n"
            "context_chunk_template: '{content}'\n"
        )

        c1 = load_prompt_config(config1)
        c2 = load_prompt_config(config2)

        # Even though both say version "1.0", the hash catches the change
        assert c1.content_hash != c2.content_hash

    def test_defaults_version_to_unknown(self, tmp_path: Path):
        """If version field is missing, should default to 'unknown'."""
        config_file = tmp_path / "prompts.yaml"
        config_file.write_text(
            "system_prompt: Be helpful.\n"
            "user_prompt_template: '{question}'\n"
            "context_chunk_template: '{content}'\n"
        )

        config = load_prompt_config(config_file)
        assert config.version == "unknown"
