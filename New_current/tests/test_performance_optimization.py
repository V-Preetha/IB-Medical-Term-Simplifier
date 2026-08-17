from types import SimpleNamespace

import torch

from app.clinical.gliner_medical import GlinerClinicalNerProvider
from app.clinical.qwen_simplifier import QwenMedicalReportSimplifier
from app.performance import PipelineProgressStore


def test_long_reports_use_one_batched_gliner_call_and_merge_duplicates() -> None:
    class BatchModel:
        def __init__(self) -> None:
            self.calls = 0
            self.chunk_count = 0

        def inference(self, texts, labels, **kwargs):
            self.calls += 1
            self.chunk_count = len(texts)
            assert "disease" in labels
            assert kwargs["batch_size"] <= 4
            return [
                [
                    {
                        "score": 0.9,
                        "label": "disease",
                        "text": "hypertension",
                        "start": text.index("hypertension"),
                        "end": text.index("hypertension") + len("hypertension"),
                    }
                ]
                for text in texts
            ]

    model = BatchModel()
    provider = GlinerClinicalNerProvider(
        model,
        model_id="test",
        chunk_characters=1500,
        batch_size=4,
    )
    paragraph = (
        "The patient has hypertension and remains clinically stable after review. "
        "The documented treatment plan remains unchanged. "
    )
    text = "\n\n".join(paragraph * 8 for _ in range(8))

    entities = provider(text)

    assert model.calls == 1
    assert model.chunk_count > 1
    assert len(entities) == 1
    assert entities[0].text == "hypertension"


def test_gliner_chunks_preserve_offsets_and_sentence_boundaries() -> None:
    provider = GlinerClinicalNerProvider(
        object(),
        model_id="test",
        chunk_characters=1500,
    )
    sentence = "A complete clinical sentence contains a documented finding. "
    text = sentence * 90

    chunks = provider._chunks(text)

    assert len(chunks) > 1
    assert "".join(chunk for _, chunk in chunks) == text
    assert all(text[offset : offset + len(chunk)] == chunk for offset, chunk in chunks)
    assert all(chunk.rstrip().endswith(".") for _, chunk in chunks[:-1])


def test_qwen_context_is_compact_and_deduplicates_generic_entities() -> None:
    simplifier = object.__new__(QwenMedicalReportSimplifier)
    simplifier.max_input_characters = 48_000
    report = (
        "On 2025-01-10 the infrarenal abdominal aortic aneurysm measured 4.9 cm. "
        "Follow-up imaging was recommended. "
        + ("Administrative boilerplate with no clinical facts. " * 500)
    )
    entities = {
        "diseases": (
            "abdominal aortic aneurysm",
            "infrarenal abdominal aortic aneurysm",
            "Aortic aneurysm",
        ),
        "measurements": ("4.9 cm",),
    }

    deduplicated = simplifier._deduplicate_entities(entities)
    context = simplifier._build_compact_context(report, deduplicated, ())

    assert deduplicated["diseases"] == ("infrarenal abdominal aortic aneurysm",)
    assert len("\n".join(context["supporting_text"])) < len(report) / 10
    assert "measured 4.9 cm" in context["supporting_text"][0]
    assert context["chronological_findings"]
    assert context["important_measurements"] == ("4.9 cm",)


def test_progress_store_exposes_latest_internal_stage() -> None:
    store = PipelineProgressStore(max_entries=2)
    store.update("request-1", "Clinical NER", 50)
    store.update("request-1", "Finalizing", 100, complete=True)

    snapshot = store.get("request-1")

    assert snapshot is not None
    assert snapshot.stage == "Finalizing"
    assert snapshot.percent == 100
    assert snapshot.complete is True


def test_qwen_caches_template_and_uses_only_generation_config() -> None:
    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Tokenizer:
        eos_token_id = 1

        def __init__(self) -> None:
            self.template_calls = 0

        def apply_chat_template(self, messages, **kwargs):
            self.template_calls += 1
            assert kwargs["enable_thinking"] is False
            return f"prefix:{messages[1]['content']}:suffix"

        def __call__(self, prompt, **kwargs):
            assert "STRUCTURED FACTS" in prompt
            return Inputs(input_ids=torch.tensor([[1, 2, 3]]))

        def decode(self, generated_ids, **kwargs):
            return json_payload

    class Model:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.generation_kwargs = None

        def generate(self, **kwargs):
            self.generation_kwargs = kwargs
            return torch.tensor([[1, 2, 3, 4]])

    json_payload = (
        '{"executive_summary":"Summary","key_findings":[],"timeline":[],'
        '"medical_terms_explained":[],"simple_explanation":"Explanation",'
        '"recommended_follow_up":[]}'
    )
    tokenizer = Tokenizer()
    model = Model()
    generation_config = SimpleNamespace(cache_implementation="dynamic")
    simplifier = QwenMedicalReportSimplifier(
        model,
        tokenizer,
        model_id="test",
        device="cpu",
        max_input_characters=1000,
        max_new_tokens=10,
        torch_module=torch,
        generation_config=generation_config,
        attention_backend="sdpa",
        compute_dtype=torch.float32,
    )

    result = simplifier.simplify("Hypertension.", {"diseases": ("Hypertension",)}, ())

    assert result.executive_summary == "Summary"
    assert tokenizer.template_calls == 1
    assert model.generation_kwargs["generation_config"] is generation_config
    assert "do_sample" not in model.generation_kwargs
    assert "temperature" not in model.generation_kwargs
    assert "top_p" not in model.generation_kwargs
    assert "top_k" not in model.generation_kwargs
