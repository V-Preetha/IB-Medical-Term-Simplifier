from app.clinical.entities import MedicalEntityExtractor
from app.clinical.gliner_medical import GlinerClinicalNerProvider
from app.clinical.labs import LabResultExtractor
from app.clinical.models import NerEntity
from app.clinical.processor import MedicalDocumentProcessor


def test_extracts_explicit_laboratory_values_without_changing_them() -> None:
    text = "Hemoglobin: 13.2 g/dL\nHbA1c 6.5%\nALT = 54 U/L\nBlood Pressure 120/80 mmHg"

    results = LabResultExtractor().extract(text)

    assert [result.to_dict() for result in results] == [
        {"name": "Hemoglobin", "value": "13.2", "unit": "g/dL"},
        {"name": "HbA1c", "value": "6.5", "unit": "%"},
        {"name": "ALT", "value": "54", "unit": "U/L"},
        {"name": "Blood Pressure", "value": "120/80", "unit": "mmHg"},
    ]


def test_does_not_create_lab_result_without_explicit_unit() -> None:
    results = LabResultExtractor().extract("HbA1c was reviewed. ALT 54.")

    assert results == ()


def test_maps_model_entities_and_rejects_text_not_present_in_report() -> None:
    text = "The patient takes Metformin for Diabetes Mellitus."
    extractor = MedicalEntityExtractor()

    entities = extractor.extract(
        text,
        (
            NerEntity("Metformin", "DRUG", 18, 27, 0.99),
            NerEntity("Diabetes Mellitus", "DISEASE", 32, 49, 0.98),
            NerEntity("Hypertension", "DISEASE", None, None, 0.95),
        ),
    )

    assert entities["medications"] == ("Metformin",)
    assert entities["diseases"] == ("Diabetes Mellitus",)


def test_structured_report_references_extracted_entities() -> None:
    text = (
        "Diagnosis: Diabetes Mellitus and Hypertension. "
        "Medication: Metformin. HbA1c 6.5%. Anatomy: Liver."
    )

    output = MedicalDocumentProcessor().process(
        text,
        (
            NerEntity("Diabetes Mellitus", "DISEASE"),
            NerEntity("Hypertension", "DISEASE"),
            NerEntity("Metformin", "CHEM"),
            NerEntity("Liver", "ANATOMY"),
            NerEntity("HbA1c", "LAB_TEST"),
            NerEntity("6.5%", "measurement"),
        ),
    )

    assert output.entities["diseases"] == (
        "Diabetes Mellitus",
        "Hypertension",
    )
    assert output.entities["medications"] == ("Metformin",)
    assert output.entities["anatomy"] == ("Liver",)
    assert output.lab_results[0].to_dict() == {
        "name": "HbA1c",
        "value": "6.5",
        "unit": "%",
    }
    assert output.entities["measurements"] == ("6.5%",)
    assert "Diabetes Mellitus" in output.simplification.simplified_explanation
    assert "Metformin" in output.simplification.simplified_explanation
    assert "Executive Summary" in output.human_readable_report
    assert "Important Findings" in output.human_readable_report
    assert "Recommended Follow-up" in output.human_readable_report


def test_structured_extraction_is_deterministic() -> None:
    processor = MedicalDocumentProcessor()
    text = "Warfarin 5 mg. Hemoglobin 13.2 g/dL."

    first = processor.process(text).to_dict()
    second = processor.process(text).to_dict()

    assert first == second


def test_extracts_measurement_ranges_and_vital_signs_from_explicit_text() -> None:
    text = "Aorta measures 4.9–5.0 cm. Blood Pressure 120/80 mmHg."
    output = MedicalDocumentProcessor().process(
        text,
        (
            NerEntity("4.9–5.0 cm", "measurement", 15, 25, 0.95),
            NerEntity("Blood Pressure", "vital sign", 27, 41, 0.96),
            NerEntity("120/80 mmHg", "measurement", 42, 54, 0.94),
        ),
    )

    assert output.entities["measurements"] == ("4.9–5.0 cm", "120/80 mmHg")
    assert output.entities["vital_signs"] == ("Blood Pressure",)


def test_gliner_provider_converts_real_model_shape_and_keeps_confidence() -> None:
    class FakeGlinerModel:
        def predict_entities(self, text: str, labels: list[str], **kwargs):
            assert "disease" in labels
            assert kwargs == {
                "threshold": 0.5,
                "flat_ner": False,
                "multi_label": True,
            }
            return [
                {
                    "score": 0.93,
                    "label": "medication",
                    "text": "metformin",
                    "start": 14,
                    "end": 23,
                },
            ]

    provider = GlinerClinicalNerProvider(
        FakeGlinerModel(),
        model_id="Ihor/test",
        confidence_threshold=0.5,
    )

    entities = provider("Patient takes metformin.")

    assert entities == (
        NerEntity(
            text="metformin",
            label="medication",
            start=14,
            end=23,
            confidence=0.93,
        ),
    )


def test_overlapping_entities_keep_highest_confidence_span() -> None:
    text = "HbA1c was measured."
    entities = MedicalEntityExtractor().extract(
        text,
        (
            NerEntity("HbA1c", "biomarker", 0, 5, 0.91),
            NerEntity("HbA1c", "laboratory test", 0, 5, 0.97),
        ),
    )

    assert entities["laboratory_tests"] == ("HbA1c",)
    assert entities["biomarkers"] == ()


def test_maps_all_gliner_clinical_labels_to_public_categories() -> None:
    values = (
        ("diabetes", "disease", "diseases"),
        ("pain", "symptom", "symptoms"),
        ("metformin", "medication", "medications"),
        ("heart", "anatomy", "anatomy"),
        ("angiography", "medical procedure", "procedures"),
        ("HbA1c", "laboratory test", "laboratory_tests"),
        ("troponin", "biomarker", "biomarkers"),
        ("6.5%", "measurement", "measurements"),
        ("blood pressure", "vital sign", "vital_signs"),
    )
    text = " | ".join(value for value, _, _ in values)
    model_entities = tuple(
        NerEntity(
            value,
            label,
            text.index(value),
            text.index(value) + len(value),
            0.9,
        )
        for value, label, _ in values
    )

    entities = MedicalEntityExtractor().extract(text, model_entities)

    for value, _, category in values:
        assert entities[category] == (value,)
