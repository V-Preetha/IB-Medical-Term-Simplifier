"""Focused tests for validation and crash-safe persistence."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

try:
    import httpx  # noqa: F401
except ImportError:
    httpx_stub = types.ModuleType("httpx")

    class HTTPError(Exception):
        """Test substitute for httpx.HTTPError."""

    class HTTPStatusError(HTTPError):
        """Test substitute for httpx.HTTPStatusError."""

    httpx_stub.HTTPError = HTTPError
    httpx_stub.HTTPStatusError = HTTPStatusError
    sys.modules["httpx"] = httpx_stub

try:
    import tqdm  # noqa: F401
except ImportError:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = object
    sys.modules["tqdm"] = tqdm_stub

from checkpoint import CheckpointError, CheckpointManager  # noqa: E402
from generator import (  # noqa: E402
    GENERATION_PROMPT,
    GenerationError,
    LLMGenerator,
)
import main as pipeline_main  # noqa: E402
from main import select_instruction_variants  # noqa: E402
from validator import (  # noqa: E402
    ValidationError,
    build_and_validate_sample,
    parse_and_validate_response,
)


def valid_content() -> dict[str, object]:
    """Return a minimal valid assistant response."""
    return {
        "summary": "The scan found no acute abnormality.",
        "simplification": {
            "clinical": "No acute abnormality was identified.",
            "general": "The scan did not show a new serious problem.",
            "child": "The picture did not show a new problem.",
        },
        "entities": [
            {
                "term": "acute",
                "type": "Other",
                "meaning": "new or sudden",
            }
        ],
    }


class ValidatorTests(unittest.TestCase):
    """Validate strict JSON and required output levels."""

    def test_builds_expected_sample(self) -> None:
        sample = build_and_validate_sample("CT: no acute finding.", valid_content())
        self.assertEqual(sample["messages"][2]["content"], valid_content())

    def test_rejects_markdown_wrapped_json(self) -> None:
        raw = f"```json\n{json.dumps(valid_content())}\n```"
        with self.assertRaises(ValidationError):
            parse_and_validate_response(raw)

    def test_rejects_empty_simplification(self) -> None:
        content = valid_content()
        content["simplification"]["child"] = "  "  # type: ignore[index]
        with self.assertRaises(ValidationError):
            parse_and_validate_response(json.dumps(content))

    def test_rejects_nonstandard_json_numbers(self) -> None:
        content = valid_content()
        content["unexpected"] = float("nan")
        with self.assertRaises(ValidationError):
            parse_and_validate_response(json.dumps(content))

    def test_rejects_invalid_entity_type(self) -> None:
        content = valid_content()
        content["entities"][0]["type"] = "Diagnosis"  # type: ignore[index]
        with self.assertRaises(ValidationError):
            parse_and_validate_response(json.dumps(content))

    def test_rejects_empty_entity_meaning(self) -> None:
        content = valid_content()
        content["entities"][0]["meaning"] = ""  # type: ignore[index]
        with self.assertRaises(ValidationError):
            parse_and_validate_response(json.dumps(content))

    def test_instruction_is_added_to_user_content(self) -> None:
        sample = build_and_validate_sample(
            "CT: no acute finding.",
            valid_content(),
            "Simplify this medical report.",
        )
        user_content = sample["messages"][1]["content"]
        self.assertTrue(
            user_content.startswith("Simplify this medical report.")
        )
        self.assertIn("Medical report:\nCT: no acute finding.", user_content)


class CheckpointTests(unittest.TestCase):
    """Exercise the append/commit crash recovery protocol."""

    def test_recovers_uncommitted_output_append(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "dataset.jsonl"
            failed = root / "failed.csv"
            manager = CheckpointManager(root / "checkpoint.json")

            manager.begin_report("r-1", 1, output, failed)
            output.write_text('{"uncommitted":true}\n', encoding="utf-8")

            resumed = CheckpointManager(root / "checkpoint.json")
            self.assertTrue(resumed.recover_incomplete(output, failed))
            self.assertEqual(output.read_bytes(), b"")
            self.assertIsNone(resumed.state.active_report_id)
            self.assertEqual(resumed.state.last_processed_row_index, 0)

    def test_committed_append_survives_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "dataset.jsonl"
            failed = root / "failed.csv"
            manager = CheckpointManager(root / "checkpoint.json")

            manager.begin_report("r-1", 1, output, failed)
            output.write_text('{"committed":true}\n', encoding="utf-8")
            manager.complete_report("r-1", 1, succeeded=True, retries=2)

            resumed = CheckpointManager(root / "checkpoint.json")
            self.assertFalse(resumed.recover_incomplete(output, failed))
            self.assertIn("committed", output.read_text(encoding="utf-8"))
            self.assertEqual(resumed.state.last_processed_report_id, "r-1")
            self.assertEqual(resumed.state.successful, 1)
            self.assertEqual(resumed.state.retries, 2)
            self.assertEqual(resumed.state.samples_written, 0)

    def test_loads_checkpoint_without_new_sample_counter(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.json"
            path.write_text(
                json.dumps(
                    {
                        "last_processed_report_id": "legacy",
                        "last_processed_row_index": 1,
                        "successful": 1,
                    }
                ),
                encoding="utf-8",
            )
            manager = CheckpointManager(path)
            self.assertEqual(manager.state.samples_written, 0)

    def test_rejects_changed_instruction_settings_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.json"
            manager = CheckpointManager(path)
            manager.bind_run_configuration(5, 42)

            resumed = CheckpointManager(path)
            with self.assertRaises(CheckpointError):
                resumed.bind_run_configuration(3, 42)


class GeneratorTests(unittest.TestCase):
    """Exercise prompt formatting and bounded validation retries."""

    @staticmethod
    def _generator(max_retries: int) -> LLMGenerator:
        instance = LLMGenerator.__new__(LLMGenerator)
        instance.settings = SimpleNamespace(
            max_retries=max_retries,
            initial_backoff_seconds=0,
            enable_self_check=False,
        )
        return instance

    def test_prompt_formats_reports_containing_braces(self) -> None:
        prompt = GENERATION_PROMPT.format(report="Finding: {uncertain}.")
        self.assertIn("Finding: {uncertain}.", prompt)
        self.assertIn('"simplification"', prompt)

    def test_retries_invalid_json_then_succeeds(self) -> None:
        generator = self._generator(max_retries=3)
        responses = iter(["not json", json.dumps(valid_content())])
        generator._request = (  # type: ignore[method-assign]
            lambda _report: next(responses)
        )
        retry_numbers: list[int] = []

        with patch("generator.time.sleep"):
            content, retries = generator.generate(
                "Medical report",
                on_retry=lambda number, _error, _delay: retry_numbers.append(
                    number
                ),
            )

        self.assertEqual(content, valid_content())
        self.assertEqual(retries, 1)
        self.assertEqual(retry_numbers, [1])

    def test_makes_initial_attempt_plus_three_retries(self) -> None:
        generator = self._generator(max_retries=3)
        calls = 0

        def invalid_response(_report: str) -> str:
            nonlocal calls
            calls += 1
            return "{}"

        generator._request = invalid_response  # type: ignore[method-assign]
        with (
            patch("generator.time.sleep"),
            self.assertRaises(GenerationError) as context,
        ):
            generator.generate("Medical report")

        self.assertEqual(calls, 4)
        self.assertEqual(context.exception.retries, 3)

    def test_self_check_regenerates_once_and_keeps_second_output(self) -> None:
        generator = self._generator(max_retries=3)
        generator.settings.enable_self_check = True
        first = valid_content()
        second = valid_content()
        second["summary"] = "Regenerated factual summary."
        generated = iter([json.dumps(first), json.dumps(second)])
        audits = iter(
            [
                json.dumps(
                    {
                        "medical_facts_changed": True,
                        "hallucinated": False,
                        "diagnoses_removed": False,
                        "medications_changed": False,
                    }
                ),
                json.dumps(
                    {
                        "medical_facts_changed": False,
                        "hallucinated": False,
                        "diagnoses_removed": False,
                        "medications_changed": False,
                    }
                ),
            ]
        )
        generator._request = (  # type: ignore[method-assign]
            lambda _report: next(generated)
        )
        generator._request_prompt = (  # type: ignore[method-assign]
            lambda _prompt: next(audits)
        )

        with patch("generator.time.sleep"):
            content, retries = generator.generate("Medical report")

        self.assertEqual(content["summary"], second["summary"])
        self.assertEqual(retries, 1)


class InstructionDiversityTests(unittest.TestCase):
    """Verify stable per-report instruction diversity."""

    def test_first_cycle_is_unique_and_reproducible(self) -> None:
        first = select_instruction_variants("r-1", 1, 10, 42)
        second = select_instruction_variants("r-1", 1, 10, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 10)

    def test_more_than_ten_samples_uses_shuffled_cycles(self) -> None:
        selected = select_instruction_variants("r-1", 1, 15, 42)
        self.assertEqual(len(selected), 15)
        self.assertEqual(len(set(selected[:10])), 10)


class PipelineIntegrationTests(unittest.TestCase):
    """Run one streamed report through the multi-sample writer."""

    def test_one_generation_writes_five_consistent_samples(self) -> None:
        class FakeGenerator:
            calls = 0

            def __init__(self, _settings: object) -> None:
                pass

            def __enter__(self) -> "FakeGenerator":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def generate(
                self,
                _report: str,
                on_retry: object = None,
            ) -> tuple[dict[str, object], int]:
                FakeGenerator.calls += 1
                return valid_content(), 0

        class FakeProgress:
            def __enter__(self) -> "FakeProgress":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def update(self, _amount: int) -> None:
                return None

            def set_postfix(self, *_args: object, **_kwargs: object) -> None:
                return None

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_csv = root / "reports.csv"
            with input_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "report_id",
                        "specialty",
                        "report_type",
                        "difficulty",
                        "report",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "report_id": "r-1",
                        "specialty": "Radiology",
                        "report_type": "CT",
                        "difficulty": "Medium",
                        "report": "CT: no acute finding.",
                    }
                )

            settings = SimpleNamespace(
                input_csv=input_csv,
                output_jsonl=root / "dataset.jsonl",
                checkpoint_file=root / "checkpoint.json",
                failed_csv=root / "failed.csv",
                log_file=root / "pipeline.log",
                provider=SimpleNamespace(
                    provider="openai",
                    model="test-model",
                ),
                samples_per_report=5,
                enable_self_check=False,
                random_seed=42,
            )
            try:
                with (
                    patch("main.LLMGenerator", FakeGenerator),
                    patch("main.tqdm", return_value=FakeProgress()),
                ):
                    pipeline_main.run(settings)
            finally:
                for handler in pipeline_main.LOGGER.handlers:
                    handler.close()
                pipeline_main.LOGGER.handlers.clear()

            lines = settings.output_jsonl.read_text(
                encoding="utf-8"
            ).splitlines()
            samples = [json.loads(line) for line in lines]
            self.assertEqual(len(samples), 5)
            instructions = {
                sample["messages"][1]["content"].splitlines()[0]
                for sample in samples
            }
            explanations = {
                json.dumps(
                    sample["messages"][2]["content"],
                    sort_keys=True,
                )
                for sample in samples
            }
            self.assertEqual(len(instructions), 5)
            self.assertEqual(len(explanations), 1)
            self.assertEqual(FakeGenerator.calls, 1)

            checkpoint = CheckpointManager(settings.checkpoint_file)
            self.assertEqual(checkpoint.state.successful, 1)
            self.assertEqual(checkpoint.state.samples_written, 5)


if __name__ == "__main__":
    unittest.main()
