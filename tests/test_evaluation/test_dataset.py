"""Tests for golden dataset loading."""

import json
from pathlib import Path

import pytest

from rag.evaluation.dataset import EvalCase, load_golden_dataset


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Create a sample golden dataset file."""
    data = [
        {
            "id": "test_q1",
            "question": "What is RAG?",
            "expected_behavior": "answer",
            "expected_sources": ["rag.md"],
            "must_contain": ["retrieval"],
            "notes": "Basic test",
        },
        {
            "id": "test_refuse",
            "question": "What is biryani?",
            "expected_behavior": "refuse",
            "expected_sources": [],
            "must_contain": [],
            "notes": "Should refuse",
        },
    ]
    path = tmp_path / "test_dataset.json"
    path.write_text(json.dumps(data))
    return path


class TestLoadGoldenDataset:
    def test_loads_all_cases(self, sample_dataset: Path):
        cases = load_golden_dataset(sample_dataset)
        assert len(cases) == 2

    def test_first_case_fields(self, sample_dataset: Path):
        cases = load_golden_dataset(sample_dataset)
        assert cases[0].id == "test_q1"
        assert cases[0].question == "What is RAG?"
        assert cases[0].expected_behavior == "answer"
        assert cases[0].expected_sources == ["rag.md"]
        assert cases[0].must_contain == ["retrieval"]

    def test_refuse_case(self, sample_dataset: Path):
        cases = load_golden_dataset(sample_dataset)
        assert cases[1].expected_behavior == "refuse"
        assert cases[1].expected_sources == []

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_golden_dataset(Path("/nonexistent/dataset.json"))

    def test_raises_on_invalid_json(self, tmp_path: Path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text('{"not": "an array"}')
        with pytest.raises(ValueError, match="JSON array"):
            load_golden_dataset(bad_file)


class TestLoadRealGoldenDataset:
    def test_real_dataset_loads(self):
        """Verify the actual golden dataset file is valid."""
        path = Path("eval/golden_dataset.json")
        if not path.exists():
            pytest.skip("Golden dataset not found")

        cases = load_golden_dataset(path)
        assert len(cases) >= 20  # we have 23 test cases

        # Check all cases have required fields
        for case in cases:
            assert case.id, f"Case missing id"
            assert case.question, f"Case {case.id} missing question"
            assert case.expected_behavior in (
                "answer", "refuse"
            ), f"Case {case.id} has invalid expected_behavior"

    def test_has_both_answer_and_refuse_cases(self):
        """Golden dataset should test both answering and refusing."""
        path = Path("eval/golden_dataset.json")
        if not path.exists():
            pytest.skip("Golden dataset not found")

        cases = load_golden_dataset(path)
        behaviors = {c.expected_behavior for c in cases}
        assert "answer" in behaviors, "Need 'answer' test cases"
        assert "refuse" in behaviors, "Need 'refuse' test cases"
