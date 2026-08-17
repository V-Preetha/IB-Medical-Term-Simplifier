"""Generate a deterministic 5,000-row medical simplification SFT corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

SYSTEM_PROMPT: Final[str] = "You are an expert medical communication assistant."
USER_PROMPT: Final[str] = "Simplify this medical report."
DEFAULT_OUTPUT: Final[Path] = Path("medical_simplifier_synthetic_5000.jsonl")
DEFAULT_MANIFEST: Final[Path] = Path(
    "medical_simplifier_synthetic_5000_manifest.json"
)
ALLOWED_ENTITY_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Disease",
        "Symptom",
        "Medication",
        "Procedure",
        "Anatomy",
        "Laboratory Test",
        "Imaging Finding",
        "Clinical Measurement",
        "Medical Device",
        "Other",
    }
)
REPORT_TYPES: Final[tuple[str, ...]] = (
    "Blood Test",
    "CBC",
    "Biochemistry",
    "MRI",
    "CT",
    "X-ray",
    "Ultrasound",
    "ECG",
    "Echocardiogram",
    "Discharge Summary",
    "Progress Note",
    "Prescription",
    "Operative Note",
    "Pathology Report",
    "Clinical Note",
)
DIFFICULTIES: Final[tuple[str, ...]] = ("Short", "Medium", "Long")


@dataclass(frozen=True)
class Scenario:
    """A coherent specialty-specific clinical scenario."""

    specialty: str
    condition: str
    condition_meaning: str
    symptom: str
    symptom_meaning: str
    medication: str
    medication_detail: str
    medication_meaning: str
    procedure: str
    procedure_meaning: str
    anatomy: str
    finding: str
    assessment: str
    lab_pattern: str
    rare: bool


@dataclass(frozen=True)
class Fact:
    """One fact expressed at all four required readability levels."""

    report: str
    clinical: str
    general: str
    child: str


def scenario(
    specialty: str,
    condition: str,
    condition_meaning: str,
    symptom: str,
    symptom_meaning: str,
    medication: str,
    medication_detail: str,
    medication_meaning: str,
    procedure: str,
    procedure_meaning: str,
    anatomy: str,
    finding: str,
    assessment: str,
    lab_pattern: str,
    rare: bool = False,
) -> Scenario:
    """Build a scenario while keeping the data declarations compact."""
    return Scenario(
        specialty=specialty,
        condition=condition,
        condition_meaning=condition_meaning,
        symptom=symptom,
        symptom_meaning=symptom_meaning,
        medication=medication,
        medication_detail=medication_detail,
        medication_meaning=medication_meaning,
        procedure=procedure,
        procedure_meaning=procedure_meaning,
        anatomy=anatomy,
        finding=finding,
        assessment=assessment,
        lab_pattern=lab_pattern,
        rare=rare,
    )


SCENARIOS: Final[tuple[Scenario, ...]] = (
    scenario(
        "Cardiology",
        "stable angina",
        "chest discomfort caused by temporarily reduced blood flow to the heart",
        "exertional chest pressure",
        "chest pressure brought on by physical activity",
        "metoprolol tartrate",
        "25 mg orally twice daily",
        "a beta blocker that lowers heart rate and workload",
        "exercise stress test",
        "a monitored heart test performed during exercise",
        "coronary arteries",
        "No resting ST-segment elevation; chest pressure occurred with exertion.",
        "The findings are consistent with stable angina; acute myocardial "
        "infarction is not identified.",
        "normal",
    ),
    scenario(
        "Cardiology",
        "hypertrophic cardiomyopathy",
        "an inherited condition in which heart muscle becomes abnormally thick",
        "exertional dyspnea",
        "shortness of breath during activity",
        "mavacamten",
        "5 mg orally once daily",
        "a medicine that reduces excessive heart-muscle contraction",
        "transthoracic echocardiogram",
        "an ultrasound examination of the heart",
        "interventricular septum",
        "Asymmetric septal hypertrophy measures 18 mm with a resting left "
        "ventricular outflow tract gradient of 45 mmHg.",
        "The findings support obstructive hypertrophic cardiomyopathy.",
        "normal",
        True,
    ),
    scenario(
        "Neurology",
        "acute ischemic stroke",
        "brain injury caused by a blocked artery",
        "sudden left arm weakness",
        "new loss of strength in the left arm",
        "aspirin",
        "81 mg orally once daily",
        "an antiplatelet medicine that reduces blood-clot formation",
        "MRI brain",
        "magnetic resonance imaging of the brain",
        "right middle cerebral artery territory",
        "A 9 mm focus of restricted diffusion is present in the right "
        "precentral gyrus without hemorrhage.",
        "The imaging confirms a small acute ischemic infarct; hemorrhage is "
        "not present.",
        "normal",
    ),
    scenario(
        "Neurology",
        "myasthenia gravis",
        "an autoimmune disorder causing muscles to tire and weaken",
        "fatigable bilateral ptosis",
        "drooping of both eyelids that worsens with use",
        "pyridostigmine",
        "60 mg orally every 6 hours",
        "a medicine that improves communication between nerves and muscles",
        "repetitive nerve stimulation",
        "a test of how nerves repeatedly activate muscles",
        "neuromuscular junction",
        "Repetitive stimulation shows a 14% decrement in compound muscle "
        "action potential amplitude.",
        "Generalized myasthenia gravis is suspected; acetylcholine receptor "
        "antibody results are pending.",
        "normal",
        True,
    ),
    scenario(
        "Oncology",
        "invasive ductal carcinoma",
        "a breast cancer that has grown beyond the milk duct",
        "painless left breast lump",
        "a lump in the left breast that does not hurt",
        "tamoxifen",
        "20 mg orally once daily",
        "a medicine that blocks estrogen effects in breast tissue",
        "ultrasound-guided core needle biopsy",
        "removal of small tissue samples using ultrasound guidance",
        "left breast",
        "An irregular 1.8 cm mass is present at the 2 o'clock position; biopsy "
        "shows estrogen-receptor-positive invasive ductal carcinoma.",
        "The pathology confirms invasive ductal carcinoma of the left breast.",
        "anemia",
    ),
    scenario(
        "Oncology",
        "multiple myeloma",
        "a cancer of antibody-producing plasma cells in bone marrow",
        "persistent thoracic back pain",
        "ongoing pain in the middle part of the back",
        "bortezomib",
        "1.3 mg/m2 subcutaneously on days 1, 8, and 15",
        "a medicine that disrupts protein handling in myeloma cells",
        "bone marrow biopsy",
        "collection of bone marrow tissue for microscopic examination",
        "bone marrow",
        "Marrow contains 28% clonal plasma cells with kappa light-chain "
        "restriction.",
        "The findings are diagnostic of plasma-cell myeloma.",
        "oncology",
        True,
    ),
    scenario(
        "Pulmonology",
        "chronic obstructive pulmonary disease",
        "long-term airflow obstruction, usually related to airway damage",
        "progressive exertional breathlessness",
        "breathing difficulty that worsens with activity",
        "tiotropium",
        "18 mcg inhaled once daily",
        "a long-acting inhaled medicine that opens the airways",
        "spirometry",
        "a breathing test that measures airflow",
        "bronchi",
        "Post-bronchodilator FEV1/FVC is 0.58 with FEV1 at 61% of predicted.",
        "The persistent airflow obstruction is consistent with moderate COPD.",
        "normal",
    ),
    scenario(
        "Pulmonology",
        "idiopathic pulmonary fibrosis",
        "progressive scarring of the lungs without an identified cause",
        "dry cough and exertional dyspnea",
        "a dry cough and shortness of breath during activity",
        "nintedanib",
        "150 mg orally twice daily",
        "an antifibrotic medicine that slows lung scarring",
        "high-resolution CT chest",
        "a detailed CT scan used to examine lung tissue",
        "subpleural lower lobes",
        "Basal subpleural reticulation, traction bronchiectasis, and honeycombing "
        "are present without focal consolidation.",
        "The pattern is typical of usual interstitial pneumonia and supports "
        "idiopathic pulmonary fibrosis.",
        "normal",
        True,
    ),
    scenario(
        "Gastroenterology",
        "gastroesophageal reflux disease",
        "repeated flow of stomach contents into the esophagus",
        "burning retrosternal discomfort",
        "a burning feeling behind the breastbone",
        "pantoprazole",
        "40 mg orally each morning",
        "a medicine that reduces stomach-acid production",
        "upper gastrointestinal endoscopy",
        "camera examination of the esophagus, stomach, and duodenum",
        "distal esophagus",
        "Los Angeles grade B erosive esophagitis is present without bleeding.",
        "The findings are consistent with reflux esophagitis; no ulcer is seen.",
        "normal",
    ),
    scenario(
        "Gastroenterology",
        "primary sclerosing cholangitis",
        "a chronic disease causing inflammation and scarring of bile ducts",
        "pruritus and fatigue",
        "itching and persistent tiredness",
        "ursodeoxycholic acid",
        "300 mg orally three times daily",
        "a bile acid medicine documented to support bile flow",
        "magnetic resonance cholangiopancreatography",
        "an MRI technique used to view the bile ducts",
        "intrahepatic bile ducts",
        "Multifocal short strictures and intervening dilatation create a beaded "
        "appearance of the intrahepatic ducts.",
        "Primary sclerosing cholangitis is likely; no dominant obstructing "
        "mass is identified.",
        "hepatic",
        True,
    ),
    scenario(
        "Nephrology",
        "chronic kidney disease stage 3b",
        "moderate-to-severe long-term reduction in kidney function",
        "bilateral ankle swelling",
        "swelling around both ankles",
        "losartan",
        "50 mg orally once daily",
        "a medicine that lowers blood pressure and reduces kidney protein loss",
        "renal ultrasound",
        "an ultrasound examination of the kidneys",
        "bilateral renal cortex",
        "Both kidneys show mild cortical thinning without hydronephrosis.",
        "Reduced eGFR over 6 months is consistent with chronic kidney disease "
        "stage 3b.",
        "renal",
    ),
    scenario(
        "Nephrology",
        "anti-glomerular basement membrane disease",
        "an autoimmune disease that rapidly damages kidney filters",
        "dark urine and reduced urine output",
        "dark-colored urine with less urine than usual",
        "cyclophosphamide",
        "100 mg orally once daily",
        "an immune-suppressing medicine used to limit antibody-mediated damage",
        "kidney biopsy",
        "removal of kidney tissue for microscopic examination",
        "glomerular basement membrane",
        "Biopsy shows crescentic glomerulonephritis with linear IgG staining.",
        "The biopsy supports anti-GBM disease; pulmonary involvement is not "
        "demonstrated in this report.",
        "renal",
        True,
    ),
    scenario(
        "Endocrinology",
        "type 2 diabetes mellitus",
        "a condition in which blood glucose remains too high",
        "polyuria and polydipsia",
        "frequent urination and increased thirst",
        "metformin",
        "500 mg orally twice daily with meals",
        "a medicine that reduces glucose production and improves insulin use",
        "glycated hemoglobin test",
        "a blood test estimating average glucose over about three months",
        "pancreatic insulin pathway",
        "HbA1c is 8.4%, above the documented target.",
        "The elevated HbA1c indicates type 2 diabetes is not at the documented "
        "glycemic target.",
        "diabetes",
    ),
    scenario(
        "Endocrinology",
        "primary adrenal insufficiency",
        "failure of the adrenal glands to produce enough cortisol",
        "fatigue, dizziness, and weight loss",
        "tiredness, light-headedness, and unintended loss of weight",
        "hydrocortisone",
        "10 mg orally on waking and 5 mg at 16:00",
        "a corticosteroid that replaces missing cortisol",
        "ACTH stimulation test",
        "a test of the adrenal glands' cortisol response",
        "adrenal cortex",
        "Cortisol is 3.1 mcg/dL at baseline and 4.2 mcg/dL at 60 minutes.",
        "The inadequate cortisol response supports primary adrenal "
        "insufficiency.",
        "adrenal",
        True,
    ),
    scenario(
        "Orthopedics",
        "knee osteoarthritis",
        "wear-related loss of cartilage in the knee joint",
        "activity-related right knee pain",
        "right knee pain that worsens with movement",
        "acetaminophen",
        "650 mg orally every 8 hours as needed",
        "a pain-relieving medicine",
        "weight-bearing knee radiographs",
        "standing X-rays of the knee",
        "medial tibiofemoral compartment",
        "Moderate medial joint-space narrowing and marginal osteophytes are "
        "present without acute fracture.",
        "The radiographic changes are consistent with moderate osteoarthritis.",
        "normal",
    ),
    scenario(
        "Orthopedics",
        "osteosarcoma",
        "a malignant bone tumor that produces immature bone",
        "progressive distal thigh pain",
        "worsening pain near the lower thigh",
        "doxorubicin",
        "75 mg/m2 intravenously on day 1 of the documented cycle",
        "a chemotherapy medicine that damages tumor-cell DNA",
        "CT-guided bone biopsy",
        "collection of bone tissue using CT guidance",
        "distal femoral metaphysis",
        "A mixed lytic and sclerotic lesion has aggressive periosteal reaction; "
        "biopsy shows malignant osteoid.",
        "The pathology confirms conventional high-grade osteosarcoma.",
        "anemia",
        True,
    ),
    scenario(
        "Pediatrics",
        "acute otitis media",
        "a middle-ear infection",
        "right ear pain and fever",
        "pain in the right ear with fever",
        "amoxicillin",
        "45 mg/kg orally twice daily",
        "an antibiotic used for susceptible bacterial infections",
        "pneumatic otoscopy",
        "examination of eardrum movement using a small air puff",
        "right tympanic membrane",
        "The right tympanic membrane is bulging and erythematous with reduced "
        "mobility.",
        "The examination is consistent with acute right otitis media.",
        "infection",
    ),
    scenario(
        "Pediatrics",
        "Kawasaki disease",
        "an inflammatory illness affecting blood vessels in young children",
        "five days of fever with red eyes and rash",
        "fever for five days together with red eyes and a skin rash",
        "intravenous immunoglobulin",
        "2 g/kg intravenously as a single infusion",
        "pooled antibodies used to reduce inflammation",
        "pediatric echocardiogram",
        "an ultrasound examination of a child's heart",
        "coronary arteries",
        "The proximal left anterior descending artery has a z-score of +2.7 "
        "without aneurysm.",
        "Kawasaki disease is likely; the coronary artery is mildly dilated but "
        "no aneurysm is present.",
        "inflammation",
        True,
    ),
    scenario(
        "Gynecology",
        "uterine leiomyomas",
        "noncancerous smooth-muscle growths in the uterus",
        "heavy menstrual bleeding",
        "menstrual bleeding that is heavier than usual",
        "tranexamic acid",
        "1 g orally three times daily for up to 5 days during menses",
        "a medicine that reduces bleeding by stabilizing blood clots",
        "transvaginal ultrasound",
        "an internal ultrasound used to examine pelvic organs",
        "uterine myometrium",
        "Three intramural fibroids are present; the largest measures 4.2 cm.",
        "The ultrasound confirms multiple intramural uterine leiomyomas.",
        "anemia",
    ),
    scenario(
        "Gynecology",
        "complete hydatidiform mole",
        "an abnormal pregnancy with overgrowth of placental tissue",
        "vaginal bleeding and severe nausea",
        "bleeding from the vagina with marked nausea",
        "ondansetron",
        "4 mg orally every 8 hours as needed",
        "an anti-nausea medicine",
        "suction dilation and curettage",
        "removal of abnormal tissue from the uterus using suction",
        "endometrial cavity",
        "Ultrasound shows heterogeneous echogenic material with numerous small "
        "cystic spaces and no embryo.",
        "A complete hydatidiform mole is suspected; histopathologic "
        "confirmation is pending.",
        "hcg",
        True,
    ),
    scenario(
        "Dermatology",
        "atopic dermatitis",
        "a chronic itchy inflammatory skin condition",
        "itchy flexural rash",
        "an itchy rash in skin folds",
        "hydrocortisone 1% cream",
        "a thin layer topically twice daily for 7 days",
        "a mild corticosteroid cream that reduces skin inflammation",
        "dermatologic examination",
        "inspection of the skin",
        "antecubital fossae",
        "Ill-defined erythematous scaly plaques with excoriations are present "
        "bilaterally without purulent drainage.",
        "The morphology and distribution are consistent with atopic dermatitis.",
        "normal",
    ),
    scenario(
        "Dermatology",
        "pemphigus vulgaris",
        "an autoimmune blistering disease of skin and mucous membranes",
        "painful oral erosions and fragile blisters",
        "painful mouth sores and easily broken skin blisters",
        "prednisone",
        "40 mg orally once daily",
        "a corticosteroid that suppresses inflammation and immune activity",
        "perilesional skin biopsy",
        "collection of skin next to a blister for microscopy",
        "suprabasal epidermis",
        "Direct immunofluorescence shows intercellular IgG and C3 in a net-like "
        "pattern.",
        "The biopsy findings support pemphigus vulgaris.",
        "inflammation",
        True,
    ),
    scenario(
        "Ophthalmology",
        "age-related cataract",
        "clouding of the eye's natural lens",
        "gradual painless blurred vision",
        "slowly worsening blurry vision without pain",
        "prednisolone acetate 1% eye drops",
        "one drop in the operated eye four times daily",
        "an anti-inflammatory eye drop documented for postoperative use",
        "slit-lamp examination",
        "microscope examination of the front of the eye",
        "crystalline lens",
        "A 3+ nuclear sclerotic cataract is present in the right eye.",
        "The lens opacity explains the reduced right-eye visual acuity.",
        "normal",
    ),
    scenario(
        "Ophthalmology",
        "giant cell arteritis",
        "inflammation of medium and large arteries that can threaten vision",
        "new temporal headache with transient visual loss",
        "a new temple-area headache with brief loss of vision",
        "prednisone",
        "60 mg orally once daily",
        "a corticosteroid that suppresses arterial inflammation",
        "temporal artery biopsy",
        "removal of a small artery segment for microscopic examination",
        "temporal artery",
        "The artery shows granulomatous inflammation with multinucleated giant "
        "cells and disruption of the internal elastic lamina.",
        "The biopsy confirms giant cell arteritis.",
        "inflammation",
        True,
    ),
    scenario(
        "Psychiatry",
        "major depressive disorder",
        "a mood disorder causing persistent low mood or loss of interest",
        "low mood, anhedonia, and insomnia",
        "low mood, reduced enjoyment, and difficulty sleeping",
        "sertraline",
        "50 mg orally once daily",
        "a selective serotonin reuptake inhibitor antidepressant",
        "PHQ-9 assessment",
        "a nine-question depression symptom scale",
        "mood and cognition",
        "PHQ-9 score is 17, indicating moderately severe depressive symptoms.",
        "The documented symptoms support major depressive disorder; no manic "
        "symptoms are reported.",
        "normal",
    ),
    scenario(
        "Psychiatry",
        "serotonin syndrome",
        "a potentially serious excess of serotonin activity",
        "agitation, tremor, and sweating",
        "restlessness, shaking, and heavy sweating",
        "sertraline",
        "50 mg orally daily before it was held",
        "a serotonin-active antidepressant that was stopped in this report",
        "neuromuscular examination",
        "assessment of reflexes, tone, and muscle movement",
        "lower-extremity reflexes",
        "Inducible ankle clonus and hyperreflexia are present with a "
        "temperature of 38.2 C.",
        "Serotonin syndrome is possible; sertraline was held and alternative "
        "causes have not been excluded.",
        "inflammation",
        True,
    ),
    scenario(
        "Emergency Medicine",
        "acute appendicitis",
        "inflammation of the appendix",
        "right lower-quadrant abdominal pain",
        "pain in the lower-right part of the abdomen",
        "ceftriaxone",
        "2 g intravenously once in the emergency department",
        "an antibiotic used to treat susceptible bacterial infections",
        "laparoscopic appendectomy",
        "keyhole surgery to remove the appendix",
        "appendix",
        "The appendix measures 11 mm with wall enhancement and surrounding fat "
        "stranding; no abscess is present.",
        "The CT findings are consistent with uncomplicated acute appendicitis.",
        "infection",
    ),
    scenario(
        "Emergency Medicine",
        "acute aortic dissection",
        "a tear within the wall of the body's main artery",
        "abrupt tearing chest pain radiating to the back",
        "sudden severe tearing pain from the chest to the back",
        "esmolol",
        "a 500 mcg/kg intravenous bolus followed by 50 mcg/kg/min",
        "a short-acting beta blocker used to reduce heart rate and pressure",
        "CT angiography",
        "contrast CT imaging of blood vessels",
        "ascending aorta",
        "An intimal flap begins in the ascending aorta and extends into the "
        "arch without pericardial effusion.",
        "The findings confirm Stanford type A aortic dissection.",
        "normal",
        True,
    ),
    scenario(
        "Radiology",
        "community-acquired pneumonia",
        "a lung infection acquired outside a hospital",
        "productive cough and fever",
        "a mucus-producing cough with fever",
        "azithromycin",
        "500 mg orally on day 1, then 250 mg daily on days 2 through 5",
        "an antibiotic used for susceptible bacterial infections",
        "chest radiograph",
        "an X-ray image of the chest",
        "right lower lobe",
        "Patchy right lower-lobe air-space opacity is present with air "
        "bronchograms and no pleural effusion.",
        "The appearance favors right lower-lobe pneumonia.",
        "infection",
    ),
    scenario(
        "Radiology",
        "meningioma",
        "a usually slow-growing tumor arising from the brain's coverings",
        "progressive morning headaches",
        "headaches that have gradually worsened in the morning",
        "dexamethasone",
        "2 mg orally twice daily",
        "a corticosteroid used here to reduce swelling",
        "contrast-enhanced MRI brain",
        "brain MRI performed after intravenous contrast",
        "left frontal convexity",
        "A 2.4 cm extra-axial enhancing mass has a dural tail and mild adjacent "
        "vasogenic edema without midline shift.",
        "The imaging most likely represents a meningioma.",
        "normal",
        True,
    ),
    scenario(
        "Pathology",
        "colonic adenocarcinoma",
        "a gland-forming cancer of the colon",
        "iron-deficiency anemia and altered bowel habits",
        "low iron-related blood count with a change in bowel habits",
        "capecitabine",
        "1,000 mg/m2 orally twice daily on days 1 through 14",
        "an oral chemotherapy medicine converted to fluorouracil in the body",
        "right hemicolectomy",
        "surgical removal of the right side of the colon",
        "ascending colon",
        "Resection shows a 3.1 cm moderately differentiated adenocarcinoma "
        "invading muscularis propria; 0 of 18 lymph nodes are involved.",
        "The specimen confirms pT2N0 colonic adenocarcinoma with negative "
        "resection margins.",
        "anemia",
    ),
    scenario(
        "Pathology",
        "AL amyloidosis",
        "a disorder in which abnormal light-chain protein deposits in organs",
        "foamy urine and leg swelling",
        "frothy urine together with swelling of the legs",
        "daratumumab",
        "1,800 mg subcutaneously once weekly in the documented cycle",
        "an antibody medicine targeting CD38 on plasma cells",
        "kidney biopsy",
        "removal of kidney tissue for microscopic examination",
        "renal glomeruli",
        "Congo red-positive deposits show apple-green birefringence and are "
        "typed as lambda light-chain amyloid.",
        "The biopsy confirms renal AL amyloidosis.",
        "renal",
        True,
    ),
)

SPECIALTIES: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(item.specialty for item in SCENARIOS)
)
REPORT_SPECIALTIES: Final[dict[str, frozenset[str]]] = {
    "ECG": frozenset(
        {
            "Cardiology",
            "Pulmonology",
            "Pediatrics",
            "Emergency Medicine",
            "Endocrinology",
            "Nephrology",
            "Psychiatry",
        }
    ),
    "Echocardiogram": frozenset(
        {
            "Cardiology",
            "Pulmonology",
            "Pediatrics",
            "Emergency Medicine",
            "Oncology",
            "Nephrology",
        }
    ),
    "Operative Note": frozenset(
        {
            "Oncology",
            "Orthopedics",
            "Gynecology",
            "Emergency Medicine",
            "Pathology",
            "Dermatology",
            "Nephrology",
            "Ophthalmology",
        }
    ),
    "Pathology Report": frozenset(
        {
            "Oncology",
            "Orthopedics",
            "Gynecology",
            "Dermatology",
            "Ophthalmology",
            "Pathology",
            "Nephrology",
        }
    ),
}

TEST_MEANINGS: Final[dict[str, str]] = {
    "Blood Test": "a laboratory evaluation of a blood sample",
    "CBC": "a complete blood count measuring blood cells",
    "Biochemistry": "blood tests measuring chemicals and organ function",
    "MRI": "magnetic resonance imaging using a strong magnet and radio waves",
    "CT": "computed tomography using X-rays to create cross-sectional images",
    "X-ray": "an imaging test using a small amount of ionizing radiation",
    "Ultrasound": "imaging that uses sound waves",
    "ECG": "a recording of the heart's electrical activity",
    "Echocardiogram": "an ultrasound examination of the heart",
    "Discharge Summary": "a summary of a completed hospital stay",
    "Progress Note": "a clinical update recorded during ongoing care",
    "Prescription": "a documented medication order",
    "Operative Note": "a record of a performed procedure",
    "Pathology Report": "microscopic examination of tissue or cells",
    "Clinical Note": "a clinician's record of an encounter",
}
DEVICE_DETAILS: Final[dict[str, tuple[str, str]]] = {
    "stable angina": (
        "ambulatory cardiac monitor",
        "a wearable device that records heart rhythm during daily activity",
    ),
    "hypertrophic cardiomyopathy": (
        "wearable cardiac monitor",
        "a device that records heart rate and rhythm outside the clinic",
    ),
    "acute ischemic stroke": (
        "rolling walker",
        "a wheeled walking aid used for mobility support",
    ),
    "invasive ductal carcinoma": (
        "implanted venous access port",
        "a device under the skin used for repeated intravenous access",
    ),
    "multiple myeloma": (
        "implanted venous access port",
        "a device under the skin used for repeated intravenous access",
    ),
    "chronic obstructive pulmonary disease": (
        "metered-dose inhaler",
        "a handheld device that delivers a measured dose of inhaled medicine",
    ),
    "idiopathic pulmonary fibrosis": (
        "portable oxygen concentrator",
        "a device that supplies concentrated oxygen",
    ),
    "chronic kidney disease stage 3b": (
        "home blood-pressure monitor",
        "a device used to measure blood pressure outside the clinic",
    ),
    "type 2 diabetes mellitus": (
        "continuous glucose monitor",
        "a wearable sensor that repeatedly measures tissue glucose",
    ),
    "knee osteoarthritis": (
        "hinged knee brace",
        "a support worn around the knee to improve stability",
    ),
    "acute otitis media": (
        "tympanostomy tube",
        "a small tube placed in the eardrum to ventilate the middle ear",
    ),
    "age-related cataract": (
        "intraocular lens",
        "an artificial lens placed inside the eye",
    ),
    "acute appendicitis": (
        "peripheral IV catheter",
        "a small tube placed in a vein for fluids or medicines",
    ),
    "acute aortic dissection": (
        "arterial line",
        "a catheter used for continuous blood-pressure measurement",
    ),
    "colonic adenocarcinoma": (
        "implanted venous access port",
        "a device under the skin used for repeated intravenous access",
    ),
}

ASSESSMENT_OPENINGS: Final[dict[str, tuple[str, ...]]] = {
    "definitive": (
        "Evaluation demonstrates",
        "The report confirms",
        "Overall assessment is",
        "Clinical impression is",
        "The results establish",
    ),
    "suggestive": (
        "Results suggest",
        "The report indicates",
        "Findings support",
        "Evidence points toward",
        "The appearance favours",
        "The evaluation is suggestive of",
        "The pattern appears consistent with",
        "A probable explanation is",
        "A possible explanation is",
        "This likely represents",
        "The differential diagnosis includes",
    ),
}

SUMMARY_OPENINGS: Final[tuple[str, ...]] = (
    "In plain language,",
    "The main message is that",
    "Overall, the report says",
    "The important finding is that",
    "The clinical picture suggests",
    "The doctor would explain that",
)

ENTITY_MEANING_TEMPLATES: Final[dict[str, tuple[str, ...]]] = {
    "Disease": (
        "{base}",
        "the medical name for {base}",
        "a condition described as {base}",
        "a diagnosis involving {base}",
        "a health problem involving {base}",
    ),
    "Symptom": (
        "{base}",
        "the reported experience of {base}",
        "a symptom described here as {base}",
        "what the patient noticed: {base}",
        "the complaint involving {base}",
    ),
    "Medication": (
        "{base}",
        "the medicine documented here; it is {core}",
        "a treatment that is {core}",
        "the prescribed drug, which is {core}",
        "a medicine used in this report as {core}",
    ),
    "Procedure": (
        "{base}",
        "a test or procedure that is {core}",
        "the examination used here; it is {core}",
        "a clinical procedure for {core}",
        "the documented test, which is {core}",
    ),
    "Laboratory Test": (
        "{base}",
        "a laboratory measurement of {core}",
        "a test that helps clinicians assess {core}",
        "a result that gives information about {core}",
        "a blood test used to evaluate {core}",
    ),
    "Imaging Finding": (
        "{base}",
        "the observation recorded on the images: {core}",
        "what the scan showed: {core}",
        "an imaging observation describing {core}",
        "the radiologist's recorded finding of {core}",
    ),
    "Clinical Measurement": (
        "{base}",
        "a clinical measurement of {core}",
        "a recorded body measurement showing {core}",
        "a value used to monitor {core}",
        "the measured value for {core}",
    ),
    "Medical Device": (
        "{base}",
        "the medical equipment documented here; it is {core}",
        "a device used as {core}",
        "the recorded medical device, which is {core}",
        "equipment that works as {core}",
    ),
    "Anatomy": (
        "{base}",
        "the body area described here: {core}",
        "the anatomical location involving {core}",
        "the part of the body referred to as {core}",
        "the report's named body site: {core}",
    ),
    "Other": (
        "{base}",
        "the documented clinical item described as {core}",
        "a report term meaning {core}",
        "the recorded item that refers to {core}",
        "the clinical phrase used for {core}",
    ),
}

CHILD_LAB_MEANINGS: Final[dict[str, tuple[str, ...]]] = {
    "WBC": (
        "counts the blood cells that fight germs",
        "shows how many infection-fighting cells are in the blood",
        "helps show whether the body may be fighting infection",
    ),
    "hemoglobin": (
        "is the part of blood that carries oxygen around the body",
        "shows how much oxygen-carrying protein is in the blood",
        "helps tell whether the blood can carry enough oxygen",
    ),
    "platelets": (
        "counts the blood pieces that help stop bleeding",
        "shows how many clot-making cells are present",
        "helps the blood form a plug after a cut",
    ),
    "MCV": (
        "shows the average size of red blood cells",
        "measures how big the red blood cells are",
        "helps describe the size of oxygen-carrying blood cells",
    ),
    "neutrophils": (
        "counts germ-fighting white blood cells",
        "shows the level of cells that often rise during infection",
        "measures one group of cells that fights germs",
    ),
    "sodium": (
        "helps balance water in the body",
        "is a blood salt that helps control body water",
        "helps nerves and muscles work while balancing water",
    ),
    "potassium": (
        "helps the heart beat and muscles work",
        "is a blood salt that helps the heartbeat stay steady",
        "helps nerves, muscles, and the heart work properly",
    ),
    "creatinine": (
        "is a waste level used to check how well the kidneys work",
        "helps show how well the kidneys clean the blood",
        "is a waste product measured to check the kidneys",
    ),
    "eGFR": (
        "estimates how well the kidneys clean the blood",
        "is a number showing the kidneys' cleaning ability",
        "helps show how much filtering work the kidneys can do",
    ),
    "glucose": (
        "measures sugar in the blood",
        "shows the amount of blood sugar",
        "checks how much sugar is moving through the blood",
    ),
    "HbA1c": (
        "shows the average blood sugar over about three months",
        "gives a three-month picture of blood sugar",
        "helps show how blood sugar has been running for several months",
    ),
    "bicarbonate": (
        "helps show whether the blood has the right acid balance",
        "checks part of the body's acid balance",
        "is a blood value that helps keep acid levels balanced",
    ),
    "CRP": (
        "is a blood sign of swelling or inflammation",
        "can rise when the body is inflamed",
        "helps show whether inflammation is present",
    ),
    "ESR": (
        "is a blood test that can rise with inflammation",
        "helps look for swelling or inflammation in the body",
        "measures a change in blood that may happen with inflammation",
    ),
    "lactate": (
        "can rise when body tissues are under stress",
        "helps show whether the body is having trouble getting enough oxygen",
        "is a blood value that may increase during serious body stress",
    ),
    "ferritin": (
        "shows how much iron the body has stored",
        "measures the body's saved iron",
        "helps tell whether iron stores are low",
    ),
    "serum iron": (
        "measures iron moving in the blood",
        "shows the amount of iron currently in the blood",
        "helps check whether enough iron is available",
    ),
    "beta-hCG": (
        "measures a hormone made during pregnancy",
        "shows the level of a pregnancy-related hormone",
        "checks a hormone linked to pregnancy tissue",
    ),
    "TSH": (
        "is a signal that tells the thyroid how hard to work",
        "helps check how the thyroid is being controlled",
        "measures the body's message to the thyroid gland",
    ),
}

CHILD_TERM_REPLACEMENTS: Final[tuple[tuple[str, str], ...]] = (
    ("acute myocardial infarction", "heart attack"),
    ("myocardial infarction", "heart attack"),
    ("hemorrhage", "bleeding"),
    ("dyspnea", "trouble breathing"),
    ("bilateral", "on both sides"),
    ("erythematous", "red"),
    ("edema", "swelling"),
    ("malignant", "cancerous"),
    ("benign", "not cancerous"),
    ("renal", "kidney"),
    ("hepatic", "liver"),
    ("cardiac", "heart"),
    ("pulmonary", "lung"),
    ("intravenously", "through a vein"),
    ("subcutaneously", "under the skin"),
    ("orally", "by mouth"),
    ("clinically indicated", "the care team thinks it is needed"),
    ("specialist", "doctor with special training"),
    (
        "acetylcholine receptor antibody",
        "blood marker linked to nerve-and-muscle signals",
    ),
    (
        "Congo red-positive deposits show apple-green birefringence",
        "a special stain found amyloid protein",
    ),
    (
        "granulomatous inflammation with multinucleated giant cells",
        "long-lasting swelling. Large joined immune cells were also seen",
    ),
    ("renal glomeruli", "tiny kidney filters"),
    ("glomerular basement membrane", "the thin kidney-filter layer"),
    ("granulomatous inflammation", "a type of long-lasting swelling"),
    ("multinucleated giant cells", "large joined immune cells"),
    ("internal elastic lamina", "artery's stretchy inner layer"),
    ("ST-segment elevation", "a heart-tracing sign"),
    ("left ventricular ejection fraction", "the heart's pumping percentage"),
    ("no evidence of", "the report did not find"),
    ("is not identified", "was not found"),
    ("cannot be excluded", "is still possible"),
    ("are consistent with", "fit with"),
    ("is consistent with", "fits with"),
    ("without", "with no"),
    ("consistent with", "fits with"),
    ("suggestive of", "looks like"),
)

LIFESTYLE_RECOMMENDATIONS: Final[dict[str, tuple[str, ...]]] = {
    "Cardiology": (
        "The plan documents review of activity tolerance and cardiovascular "
        "risk factors.",
        "The plan includes a heart-healthy lifestyle discussion at follow-up.",
    ),
    "Endocrinology": (
        "The plan includes nutrition review and continued glucose monitoring.",
        "Dietary counseling and review of the glucose record are documented.",
    ),
    "Nephrology": (
        "The plan includes dietitian review of sodium intake.",
        "Blood-pressure and kidney-function monitoring are documented.",
    ),
    "Gastroenterology": (
        "The plan includes review of food-related symptom triggers.",
        "Dietary review is documented as part of follow-up.",
    ),
    "Pulmonology": (
        "The plan includes review of inhaler technique and symptom monitoring.",
        "Breathing symptoms and treatment technique will be reviewed.",
    ),
}


def _number(rng: random.Random, low: float, high: float, digits: int = 1) -> str:
    value = round(rng.uniform(low, high), digits)
    if digits == 0:
        return str(int(value))
    return f"{value:.{digits}f}"


def _strip_article(text: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", text.strip(), flags=re.IGNORECASE)


def _vary_entity_meaning(
    entity_type: str,
    base: str,
    rng: random.Random,
) -> str:
    """Render context-correct entity meanings with deterministic diversity."""
    templates = ENTITY_MEANING_TEMPLATES[entity_type]
    template = rng.choice(templates)
    return template.format(base=base, core=_strip_article(base))


def _child_friendly_text(text: str) -> str:
    """Replace common clinical jargon while retaining names and values."""
    simplified = text
    for clinical, plain in CHILD_TERM_REPLACEMENTS:
        simplified = re.sub(
            re.escape(clinical),
            plain,
            simplified,
            flags=re.IGNORECASE,
        )
    return simplified


def _child_lab_meaning(
    name: str,
    fallback: str,
    rng: random.Random,
) -> str:
    variants = CHILD_LAB_MEANINGS.get(name)
    if variants is not None:
        return rng.choice(variants)
    fallback = _child_friendly_text(fallback)
    return rng.choice(
        (
            fallback,
            f"helps the care team check {fallback}",
            f"is a blood result that shows {fallback}",
        )
    )


def _assessment_core(assessment: str) -> tuple[str, bool]:
    """Remove repetitive lead-ins without losing clinical qualifiers."""
    definitive = bool(
        re.search(
            r"\b(confirms|confirmed|diagnostic|diagnosis is)\b",
            assessment,
            flags=re.IGNORECASE,
        )
    )
    patterns = (
        r"^The findings are consistent with\s+",
        r"^The imaging confirms\s+",
        r"^The pathology confirms\s+",
        r"^The biopsy confirms\s+",
        r"^The biopsy findings support\s+",
        r"^The findings support\s+",
        r"^The imaging most likely represents\s+",
        r"^The appearance favors\s+",
        r"^The pattern is typical of\s+",
        r"^The persistent airflow obstruction is consistent with\s+",
    )
    core = assessment.strip().rstrip(".")
    for pattern in patterns:
        updated = re.sub(pattern, "", core, flags=re.IGNORECASE)
        if updated != core:
            core = updated
            break
    return core, definitive


def _render_assessment(
    item: Scenario,
    rng: random.Random,
) -> tuple[str, str]:
    """Return varied clinician and child assessment wording."""
    core, definitive = _assessment_core(item.assessment)
    opening_group = "definitive" if definitive else "suggestive"
    opening = rng.choice(ASSESSMENT_OPENINGS[opening_group])
    clinical = f"{opening} {core[0].lower() + core[1:]}."

    child_core = _child_friendly_text(item.assessment)
    if re.search(r"\b(possible|suspected|likely|pending)\b", item.assessment,
                 flags=re.IGNORECASE):
        child = (
            f"Doctors are not fully certain yet. {child_core} "
            f"The name being considered is {item.condition}, which means "
            f"{_child_friendly_text(item.condition_meaning)}."
        )
    else:
        child = (
            f"Doctors call this {item.condition}. "
            f"It means {_child_friendly_text(item.condition_meaning)}. "
            f"{child_core}"
        )
    return clinical, child


def _recommendation_fact(
    item: Scenario,
    report_type: str,
    rng: random.Random,
) -> Fact:
    interval = rng.choice((1, 2, 4, 6, 8, 12))
    recommendations = [
        f"Continue {item.medication} at the recorded dose of "
        f"{item.medication_detail}.",
        f"Arrange {item.specialty} follow-up in {interval} weeks.",
        f"A referral to {item.specialty} is documented.",
        "Monitor the documented symptoms until follow-up.",
    ]
    if report_type in {"MRI", "CT", "X-ray", "Ultrasound"}:
        recommendations.append(
            f"Repeat {report_type} imaging in {interval} weeks if the "
            "specialist determines it is needed."
        )
    elif report_type in {"Blood Test", "CBC", "Biochemistry"}:
        recommendations.append(
            f"Repeat the documented blood tests in {interval} weeks."
        )
    elif report_type in {"ECG", "Echocardiogram"}:
        recommendations.append(
            f"Repeat the {report_type} in {interval} weeks if clinically "
            "indicated."
        )
    if item.specialty in LIFESTYLE_RECOMMENDATIONS:
        recommendations.append(
            rng.choice(LIFESTYLE_RECOMMENDATIONS[item.specialty])
        )
    if (
        item.specialty == "Emergency Medicine"
        or report_type == "Discharge Summary"
    ):
        recommendations.append(
            "The discharge instructions say to return if symptoms worsen."
        )

    rng.shuffle(recommendations)
    chosen = recommendations[: rng.choice((2, 2, 3))]
    report = "Recommendations: " + " ".join(chosen)
    child = "The plan says: " + " ".join(
        _child_friendly_text(value) for value in chosen
    )
    return Fact(
        report=report,
        clinical=report,
        general="The documented plan is: " + " ".join(chosen),
        child=child,
    )


def demographics(
    item: Scenario,
    rng: random.Random,
) -> tuple[int, str]:
    """Return an age and inclusive gender label suitable for the scenario."""
    if item.specialty == "Pediatrics":
        if item.condition == "Kawasaki disease":
            age = rng.randint(2, 6)
        else:
            age = rng.randint(1, 16)
        gender = rng.choice(("girl", "boy", "nonbinary child"))
        return age, gender
    if item.specialty == "Gynecology":
        return rng.randint(18, 54), rng.choice(
            ("woman", "woman", "nonbinary adult with a uterus")
        )
    if item.condition in {"age-related cataract", "giant cell arteritis"}:
        return rng.randint(55, 88), rng.choice(
            ("woman", "man", "nonbinary adult")
        )
    return rng.randint(18, 89), rng.choice(
        ("woman", "man", "nonbinary adult")
    )


def vital_values(
    item: Scenario,
    age: int,
    rng: random.Random,
) -> dict[str, str]:
    """Generate plausible vitals with scenario-specific adjustments."""
    systolic = rng.randint(108, 148)
    diastolic = rng.randint(66, 92)
    heart_rate = rng.randint(62, 96)
    respiratory_rate = rng.randint(14, 20)
    temperature = round(rng.uniform(36.4, 37.4), 1)
    oxygen = rng.randint(95, 99)

    if item.specialty == "Pediatrics":
        heart_rate = rng.randint(88, 128 if age < 8 else 108)
        systolic = rng.randint(88, 112)
        diastolic = rng.randint(52, 72)
    if item.lab_pattern == "infection":
        temperature = round(rng.uniform(38.0, 39.1), 1)
        heart_rate += rng.randint(8, 20)
    if item.specialty == "Pulmonology":
        oxygen = rng.randint(90, 94)
        respiratory_rate = rng.randint(20, 27)
    if item.condition == "primary adrenal insufficiency":
        systolic = rng.randint(88, 98)
        diastolic = rng.randint(54, 64)
    if item.condition == "acute aortic dissection":
        systolic = rng.randint(168, 198)
        diastolic = rng.randint(92, 112)
        heart_rate = rng.randint(92, 116)
    if item.condition == "serotonin syndrome":
        temperature = 38.2
        heart_rate = rng.randint(104, 124)

    return {
        "blood_pressure": f"{systolic}/{diastolic} mmHg",
        "heart_rate": f"{heart_rate} bpm",
        "respiratory_rate": f"{respiratory_rate}/min",
        "temperature": f"{temperature:.1f} C",
        "oxygen_saturation": f"{oxygen}%",
    }


def cbc_results(
    pattern: str,
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Create a coherent complete blood count pattern."""
    wbc = _number(rng, 4.8, 9.8)
    hemoglobin = _number(rng, 12.2, 15.4)
    platelets = _number(rng, 180, 360, 0)
    extra_name = "MCV"
    extra_value = f"{_number(rng, 82, 96, 0)} fL"
    extra_meaning = "mean corpuscular volume, the average red-cell size"

    if pattern == "infection":
        wbc = _number(rng, 12.4, 18.2)
        extra_name = "neutrophils"
        extra_value = f"{_number(rng, 78, 91, 0)}%"
        extra_meaning = "white blood cells that commonly rise with infection"
    elif pattern in {"anemia", "oncology"}:
        hemoglobin = _number(rng, 7.9, 10.6)
        extra_value = f"{_number(rng, 68, 79, 0)} fL"
    elif pattern == "renal":
        hemoglobin = _number(rng, 9.2, 11.4)
        extra_value = f"{_number(rng, 82, 94, 0)} fL"
    elif pattern == "inflammation":
        wbc = _number(rng, 10.6, 14.8)
        platelets = _number(rng, 330, 510, 0)

    return [
        ("WBC", f"{wbc} x10^9/L", "white blood cell count"),
        ("hemoglobin", f"{hemoglobin} g/dL", "the oxygen-carrying blood protein"),
        ("platelets", f"{platelets} x10^9/L", "cells that help blood clot"),
        (extra_name, extra_value, extra_meaning),
    ]


def chemistry_results(
    pattern: str,
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Create condition-aligned blood chemistry results."""
    if pattern == "renal":
        return [
            (
                "creatinine",
                f"{_number(rng, 1.8, 4.6)} mg/dL",
                "a waste marker used to assess kidney filtration",
            ),
            (
                "eGFR",
                f"{_number(rng, 14, 42, 0)} mL/min/1.73 m2",
                "an estimate of kidney filtering ability",
            ),
            (
                "potassium",
                f"{_number(rng, 4.8, 5.8)} mmol/L",
                "an electrolyte important for nerves and heart rhythm",
            ),
        ]
    if pattern == "diabetes":
        return [
            (
                "glucose",
                f"{_number(rng, 168, 286, 0)} mg/dL",
                "the amount of sugar in the blood",
            ),
            (
                "HbA1c",
                f"{_number(rng, 7.4, 10.8)}%",
                "estimated average blood glucose over about three months",
            ),
            (
                "bicarbonate",
                f"{_number(rng, 22, 28, 0)} mmol/L",
                "a marker of the blood's acid-base balance",
            ),
        ]
    if pattern == "hepatic":
        return [
            (
                "alkaline phosphatase",
                f"{_number(rng, 260, 610, 0)} U/L",
                "an enzyme that may rise with bile-duct obstruction",
            ),
            (
                "total bilirubin",
                f"{_number(rng, 1.8, 5.1)} mg/dL",
                "a pigment processed by the liver and bile ducts",
            ),
            (
                "ALT",
                f"{_number(rng, 62, 170, 0)} U/L",
                "an enzyme that can rise with liver-cell injury",
            ),
        ]
    if pattern == "adrenal":
        return [
            (
                "sodium",
                f"{_number(rng, 124, 131, 0)} mmol/L",
                "an electrolyte controlling fluid balance",
            ),
            (
                "potassium",
                f"{_number(rng, 5.2, 6.0)} mmol/L",
                "an electrolyte important for nerves and heart rhythm",
            ),
            (
                "glucose",
                f"{_number(rng, 54, 68, 0)} mg/dL",
                "the amount of sugar in the blood",
            ),
        ]
    if pattern == "hcg":
        return [
            (
                "beta-hCG",
                f"{_number(rng, 145000, 310000, 0)} mIU/mL",
                "a pregnancy-related hormone measured in blood",
            ),
            (
                "TSH",
                f"{_number(rng, 0.08, 0.35, 2)} mIU/L",
                "a hormone that regulates thyroid activity",
            ),
            (
                "creatinine",
                f"{_number(rng, 0.5, 0.9)} mg/dL",
                "a waste marker used to assess kidney filtration",
            ),
        ]
    if pattern == "inflammation":
        return [
            (
                "CRP",
                f"{_number(rng, 42, 126, 0)} mg/L",
                "C-reactive protein, a marker of inflammation",
            ),
            (
                "ESR",
                f"{_number(rng, 48, 105, 0)} mm/hr",
                "erythrocyte sedimentation rate, an inflammation marker",
            ),
            (
                "creatinine",
                f"{_number(rng, 0.6, 1.1)} mg/dL",
                "a waste marker used to assess kidney filtration",
            ),
        ]
    if pattern == "oncology":
        return [
            (
                "corrected calcium",
                f"{_number(rng, 10.8, 12.6)} mg/dL",
                "blood calcium adjusted for the albumin level",
            ),
            (
                "total protein",
                f"{_number(rng, 8.7, 10.6)} g/dL",
                "the combined amount of major proteins in blood",
            ),
            (
                "creatinine",
                f"{_number(rng, 1.2, 2.2)} mg/dL",
                "a waste marker used to assess kidney filtration",
            ),
        ]
    if pattern == "anemia":
        return [
            (
                "ferritin",
                f"{_number(rng, 5, 14, 0)} ng/mL",
                "a measure of stored iron",
            ),
            (
                "serum iron",
                f"{_number(rng, 18, 38, 0)} mcg/dL",
                "the amount of circulating iron",
            ),
            (
                "creatinine",
                f"{_number(rng, 0.6, 1.1)} mg/dL",
                "a waste marker used to assess kidney filtration",
            ),
        ]
    if pattern == "infection":
        return [
            (
                "CRP",
                f"{_number(rng, 48, 138, 0)} mg/L",
                "C-reactive protein, a marker of inflammation",
            ),
            (
                "lactate",
                f"{_number(rng, 1.0, 2.1)} mmol/L",
                "a marker that may rise when tissues are under stress",
            ),
            (
                "creatinine",
                f"{_number(rng, 0.6, 1.2)} mg/dL",
                "a waste marker used to assess kidney filtration",
            ),
        ]
    return [
        (
            "sodium",
            f"{_number(rng, 136, 143, 0)} mmol/L",
            "an electrolyte controlling fluid balance",
        ),
        (
            "potassium",
            f"{_number(rng, 3.7, 4.8)} mmol/L",
            "an electrolyte important for nerves and heart rhythm",
        ),
        (
            "creatinine",
            f"{_number(rng, 0.6, 1.1)} mg/dL",
            "a waste marker used to assess kidney filtration",
        ),
    ]


def _laboratory_fact(
    label: str,
    results: list[tuple[str, str, str]],
    rng: random.Random,
) -> Fact:
    result_text = "; ".join(
        f"{name} {value}" for name, value, _meaning in results
    )
    explanation = "; ".join(
        f"{name} {value} ({meaning})"
        for name, value, meaning in results
    )
    child_results: list[str] = []
    preserved_capitalization = {
        "ALT",
        "CBC",
        "CRP",
        "ESR",
        "HbA1c",
        "MCV",
        "TSH",
        "WBC",
        "beta-hCG",
        "eGFR",
    }
    for name, value, meaning in results:
        display_name = (
            name if name in preserved_capitalization
            else name.capitalize()
        )
        child_results.append(
            f"The {display_name} result was {value}. "
            f"This {_child_lab_meaning(name, meaning, rng)}."
        )
    child_explanation = " ".join(child_results)
    general_opening = rng.choice(
        (
            f"The {label.lower()} recorded",
            f"Laboratory review shows",
            f"The reported {label.lower()} values are",
            f"Results from the {label.lower()} include",
        )
    )
    child_opening = rng.choice(
        (
            "The blood check found",
            "These are the blood-test numbers",
            "The lab measured",
            "The blood results show",
        )
    )
    return Fact(
        report=f"{label}: {result_text}.",
        clinical=f"{label}: {result_text}.",
        general=f"{general_opening} {explanation}.",
        child=f"{child_opening}: {child_explanation}",
    )


def _patient_fact(
    age: int,
    gender: str,
    item: Scenario,
    duration: str,
    encounter_date: str,
) -> Fact:
    report = (
        f"Encounter date: {encounter_date}. Patient: {age}-year-old "
        f"{gender}. Presenting concern: "
        f"{item.symptom} for {duration}."
    )
    return Fact(
        report=report,
        clinical=report,
        general=(
            f"On {encounter_date}, the patient was a {age}-year-old "
            f"{gender} with {item.symptom_meaning} for {duration}."
        ),
        child=(
            f"On {encounter_date}, this {age}-year-old {gender} had "
            f"{item.symptom_meaning} for {duration}."
        ),
    )


def _vitals_fact(values: dict[str, str]) -> Fact:
    compact = (
        f"BP {values['blood_pressure']}, HR {values['heart_rate']}, "
        f"RR {values['respiratory_rate']}, temperature "
        f"{values['temperature']}, SpO2 {values['oxygen_saturation']}"
    )
    expanded = (
        f"blood pressure {values['blood_pressure']}, heart rate "
        f"{values['heart_rate']}, breathing rate "
        f"{values['respiratory_rate']}, temperature "
        f"{values['temperature']}, and oxygen saturation "
        f"{values['oxygen_saturation']}"
    )
    return Fact(
        report=f"Vital signs: {compact}.",
        clinical=f"Vital signs: {compact}.",
        general=f"The recorded vital signs were {expanded}.",
        child=f"The body checks showed {expanded}.",
    )


def _assessment_fact(item: Scenario, rng: random.Random) -> Fact:
    clinical_assessment, child_assessment = _render_assessment(item, rng)
    plain = f"{item.condition} means {item.condition_meaning}."
    assessment = (
        f"Clinical consideration: {item.condition}. "
        f"Assessment: {clinical_assessment}"
    )
    general_opening = rng.choice(
        (
            "In everyday terms,",
            "Put simply,",
            "For the patient, this means",
            "The practical meaning is that",
        )
    )
    return Fact(
        report=assessment,
        clinical=assessment,
        general=f"{assessment} {general_opening} {plain}",
        child=child_assessment,
    )


def _medication_fact(item: Scenario, rng: random.Random) -> Fact:
    exact = f"{item.medication} {item.medication_detail}"
    general_templates = (
        "The medication list records {exact}; {meaning}.",
        "Recorded treatment is {exact}; {meaning}.",
        "The report lists {exact}. This is {meaning}.",
        "Medication reconciliation confirms {exact}; {meaning}.",
    )
    child_templates = (
        "The medicine is {exact}. It is part of the recorded treatment.",
        "The report keeps the medicine and dose as {exact}.",
        "The recorded medicine is {exact}. That dose was not changed.",
        "The care note lists {exact} as the medicine plan.",
    )
    return Fact(
        report=f"Medication record: {exact}.",
        clinical=f"Medication record: {exact}.",
        general=rng.choice(general_templates).format(
            exact=exact,
            meaning=item.medication_meaning,
        ),
        child=rng.choice(child_templates).format(exact=exact),
    )


def _imaging_fact(
    report_type: str,
    item: Scenario,
    rng: random.Random,
) -> Fact:
    general_template = rng.choice(
        (
            "The {test} examined the {site}. It recorded: {finding}",
            "{test} review of the {site} shows: {finding}",
            "The images of the {site} demonstrate: {finding}",
            "According to the {test}, the {site} has this finding: {finding}",
        )
    )
    child_finding = _child_friendly_text(item.finding)
    return Fact(
        report=(
            f"{report_type} examination of the {item.anatomy}: {item.finding}"
        ),
        clinical=(
            f"{report_type} examination of the {item.anatomy}: {item.finding}"
        ),
        general=general_template.format(
            test=report_type,
            site=item.anatomy,
            finding=item.finding,
        ),
        child=(
            f"The {report_type} made pictures of the {item.anatomy}. "
            f"In simpler words, it showed: {child_finding}"
        ),
    )


def _ecg_fact(
    item: Scenario,
    values: dict[str, str],
    rng: random.Random,
) -> Fact:
    rate = values["heart_rate"]
    pr = rng.randint(138, 196)
    qrs = rng.randint(78, 104)
    qtc = rng.randint(398, 454)
    qualifier = (
        "Left ventricular hypertrophy voltage criteria are present."
        if item.condition == "hypertrophic cardiomyopathy"
        else "No acute ST-segment elevation is present."
    )
    finding = (
        f"Sinus rhythm at {rate}; PR {pr} ms, QRS {qrs} ms, "
        f"QTc {qtc} ms. {qualifier}"
    )
    return Fact(
        report=f"12-lead ECG: {finding}",
        clinical=f"12-lead ECG: {finding}",
        general=(
            f"The 12-lead ECG showed a regular sinus rhythm at {rate}. "
            f"PR was {pr} ms, QRS was {qrs} ms, and corrected QT was "
            f"{qtc} ms. {qualifier}"
        ),
        child=(
            "The 12-lead heart tracing recorded: "
            f"{_child_friendly_text(finding)}"
        ),
    )


def _echo_fact(item: Scenario, rng: random.Random) -> Fact:
    if item.condition == "hypertrophic cardiomyopathy":
        finding = item.finding
    elif item.condition == "Kawasaki disease":
        finding = item.finding
    else:
        ef = rng.randint(55, 68)
        finding = (
            f"Left ventricular ejection fraction is {ef}%; chamber size is "
            "normal and no pericardial effusion is present."
        )
    return Fact(
        report=f"Echocardiogram: {finding}",
        clinical=f"Echocardiogram: {finding}",
        general=rng.choice(
            (
                f"The heart ultrasound recorded: {finding}",
                f"Echocardiogram review shows: {finding}",
                f"The heart images demonstrate: {finding}",
            )
        ),
        child=(
            "The heart picture showed: "
            f"{_child_friendly_text(finding)}"
        ),
    )


def _procedure_fact(
    report_type: str,
    item: Scenario,
    rng: random.Random,
) -> Fact:
    if report_type == "Operative Note":
        blood_loss = rng.choice((5, 10, 20, 35, 50, 75))
        statement = (
            f"{item.procedure} was completed for {item.condition}. "
            f"Estimated blood loss was {blood_loss} mL. No immediate "
            "complication was documented."
        )
        return Fact(
            report=f"Procedure: {statement}",
            clinical=f"Procedure: {statement}",
            general=f"The procedure record states: {statement}",
            child=(
                "The procedure note says: "
                f"{_child_friendly_text(statement)}"
            ),
        )
    if report_type == "Pathology Report":
        statement = (
            f"Specimen from the {item.anatomy}: {item.finding}"
        )
        return Fact(
            report=f"Pathology: {statement}",
            clinical=f"Pathology: {statement}",
            general=f"The tissue examination recorded: {statement}",
            child=(
                "A small piece of tissue was checked closely. "
                f"{_child_friendly_text(statement)}"
            ),
        )
    statement = (
        f"Relevant procedure: {item.procedure}. Finding: {item.finding}"
    )
    return Fact(
        report=statement,
        clinical=statement,
        general=(
            f"The documented test was {item.procedure}, "
            f"{item.procedure_meaning}. It recorded: {item.finding}"
        ),
        child=(
            f"The report used {item.procedure}, "
            f"{_child_friendly_text(item.procedure_meaning)}. "
            f"It recorded: {_child_friendly_text(item.finding)}"
        ),
    )


def _report_type_fact(
    report_type: str,
    item: Scenario,
    values: dict[str, str],
    cbc: list[tuple[str, str, str]],
    chemistry: list[tuple[str, str, str]],
    rng: random.Random,
) -> Fact:
    if report_type == "Blood Test":
        return _laboratory_fact(
            "Blood tests",
            cbc[:3] + chemistry,
            rng,
        )
    if report_type == "CBC":
        return _laboratory_fact("CBC", cbc, rng)
    if report_type == "Biochemistry":
        return _laboratory_fact("Biochemistry", chemistry, rng)
    if report_type in {"MRI", "CT", "X-ray", "Ultrasound"}:
        return _imaging_fact(report_type, item, rng)
    if report_type == "ECG":
        return _ecg_fact(item, values, rng)
    if report_type == "Echocardiogram":
        return _echo_fact(item, rng)
    if report_type in {"Operative Note", "Pathology Report"}:
        return _procedure_fact(report_type, item, rng)
    if report_type == "Prescription":
        exact = f"{item.medication} {item.medication_detail}"
        return Fact(
            report=(
                f"Prescription: {exact}. Documented indication: "
                f"{item.condition}."
            ),
            clinical=(
                f"Prescription: {exact}. Documented indication: "
                f"{item.condition}."
            ),
            general=(
                f"The prescription is {exact} for the documented "
                f"{item.condition}; {item.medication_meaning}."
            ),
            child=(
                f"The medicine order is {exact} for the documented "
                f"{item.condition}. The medicine and dose are unchanged."
            ),
        )
    return _procedure_fact(report_type, item, rng)


def _context_facts(
    report_type: str,
    rng: random.Random,
) -> list[Fact]:
    allergy = rng.choice(
        (
            "No known drug allergies are documented.",
            "A latex allergy is documented; no drug allergy is listed.",
        )
    )
    status = rng.choice(("stable", "improving", "unchanged"))
    disposition = (
        "The patient was discharged in stable condition."
        if report_type == "Discharge Summary"
        else f"Clinical status is documented as {status}."
    )
    return [
        Fact(allergy, allergy, allergy, allergy),
        Fact(
            "No prior comparison study is available.",
            "No prior comparison study is available.",
            "There is no earlier study available for comparison.",
            "There is no older test to compare with this one.",
        ),
        Fact(
            disposition,
            disposition,
            disposition,
            disposition,
        ),
    ]


def _device_fact(item: Scenario) -> Fact | None:
    details = DEVICE_DETAILS.get(item.condition)
    if details is None:
        return None
    term, meaning = details
    statement = f"Device record: {term} is documented; {meaning}."
    return Fact(
        report=statement,
        clinical=statement,
        general=statement,
        child=statement,
    )


def _render_report(
    report_type: str,
    specialty: str,
    difficulty: str,
    facts: list[Fact],
    rng: random.Random,
) -> tuple[str, str]:
    """Render the same facts in varied real-world documentation styles."""
    imaging_types = {
        "MRI",
        "CT",
        "X-ray",
        "Ultrasound",
        "Echocardiogram",
    }
    if report_type in imaging_types:
        styles = ("radiology", "dictated", "sectioned")
    elif report_type == "Discharge Summary":
        styles = ("discharge", "narrative", "sectioned")
    elif report_type == "Progress Note":
        styles = ("soap", "narrative", "sectioned")
    elif report_type == "Pathology Report":
        styles = ("pathology", "dictated", "sectioned")
    else:
        styles = ("sectioned", "narrative", "dictated", "soap")
    style = rng.choice(styles)
    heading = (
        f"{report_type.upper()}\nSpecialty: {specialty}\n"
        f"Complexity: {difficulty}"
    )

    patient = facts[0]
    assessment = next(
        (
            fact for fact in facts
            if fact.report.startswith("Clinical consideration:")
        ),
        facts[-1],
    )
    recommendation = next(
        (
            fact for fact in facts
            if fact.report.startswith("Recommendations:")
        ),
        None,
    )
    remaining = [
        fact for fact in facts
        if fact is not patient
        and fact is not assessment
        and fact is not recommendation
    ]
    if rng.random() < 0.5 and len(remaining) > 2:
        split = rng.randint(1, len(remaining) - 1)
        remaining = remaining[split:] + remaining[:split]

    if style == "radiology":
        sections = [
            heading,
            f"CLINICAL HISTORY:\n{patient.report}",
            "FINDINGS:\n" + "\n".join(fact.report for fact in remaining),
            f"IMPRESSION:\n{assessment.report}",
        ]
        if recommendation is not None:
            sections.append(f"RECOMMENDATION:\n{recommendation.report}")
        return "\n\n".join(sections), style
    if style == "discharge":
        sections = [
            heading,
            f"REASON FOR ADMISSION:\n{patient.report}",
            "HOSPITAL COURSE:\n"
            + "\n".join(fact.report for fact in remaining),
            f"DISCHARGE ASSESSMENT:\n{assessment.report}",
        ]
        if recommendation is not None:
            sections.append(f"DISCHARGE PLAN:\n{recommendation.report}")
        return "\n\n".join(sections), style
    if style == "pathology":
        sections = [
            heading,
            f"CLINICAL INFORMATION:\n{patient.report}",
            "MICROSCOPIC DESCRIPTION:\n"
            + "\n".join(fact.report for fact in remaining),
            f"FINAL INTERPRETATION:\n{assessment.report}",
        ]
        if recommendation is not None:
            sections.append(f"COMMENT:\n{recommendation.report}")
        return "\n\n".join(sections), style
    if style == "soap":
        sections = [
            heading,
            f"SUBJECTIVE:\n{patient.report}",
            "OBJECTIVE:\n" + "\n".join(fact.report for fact in remaining),
            f"ASSESSMENT:\n{assessment.report}",
        ]
        if recommendation is not None:
            sections.append(f"PLAN:\n{recommendation.report}")
        return "\n\n".join(sections), style
    if style == "dictated":
        body = [patient, *remaining, assessment]
        if recommendation is not None:
            body.append(recommendation)
        return (
            heading
            + "\n\nDictated report:\n"
            + " ".join(fact.report for fact in body),
            style,
        )
    if style == "narrative":
        body = [patient, *remaining, assessment]
        if recommendation is not None:
            body.append(recommendation)
        return heading + "\n\n" + " ".join(
            fact.report for fact in body
        ), style
    return heading + "\n" + "\n".join(
        fact.report for fact in facts
    ), style


def select_scenario(index: int, report_type: str) -> Scenario:
    """Select balanced common/rare scenarios compatible with report type."""
    preferred = SPECIALTIES[(index // len(REPORT_TYPES)) % len(SPECIALTIES)]
    allowed = REPORT_SPECIALTIES.get(report_type)
    if allowed is not None and preferred not in allowed:
        candidates = [
            item for item in SCENARIOS if item.specialty in allowed
        ]
        return candidates[(index + index // len(REPORT_TYPES)) %
                          len(candidates)]
    candidates = [
        item for item in SCENARIOS if item.specialty == preferred
    ]
    scenario_cycle = index // (len(REPORT_TYPES) * len(SPECIALTIES))
    return candidates[scenario_cycle % len(candidates)]


def _deduplicate_entities(
    candidates: list[dict[str, str]],
    report: str,
) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    entities: list[dict[str, str]] = []
    report_lower = report.lower()
    for entity in candidates:
        key = (entity["term"].lower(), entity["type"])
        if key in seen or entity["term"].lower() not in report_lower:
            continue
        seen.add(key)
        entities.append(entity)
    return entities


def _build_entities(
    item: Scenario,
    report_type: str,
    report: str,
    cbc: list[tuple[str, str, str]],
    chemistry: list[tuple[str, str, str]],
    rng: random.Random,
) -> list[dict[str, str]]:
    candidates = [
        {
            "term": item.condition,
            "type": "Disease",
            "meaning": item.condition_meaning,
        },
        {
            "term": item.symptom,
            "type": "Symptom",
            "meaning": item.symptom_meaning,
        },
        {
            "term": item.medication,
            "type": "Medication",
            "meaning": item.medication_meaning,
        },
        {
            "term": item.procedure,
            "type": "Procedure",
            "meaning": item.procedure_meaning,
        },
        {
            "term": item.anatomy,
            "type": "Anatomy",
            "meaning": f"the anatomical site described in this {report_type}",
        },
        {
            "term": report_type,
            "type": (
                "Laboratory Test"
                if report_type in {"Blood Test", "CBC", "Biochemistry"}
                else "Procedure"
                if report_type in {
                    "MRI",
                    "CT",
                    "X-ray",
                    "Ultrasound",
                    "ECG",
                    "Echocardiogram",
                }
                else "Other"
            ),
            "meaning": TEST_MEANINGS[report_type],
        },
        {
            "term": "BP",
            "type": "Clinical Measurement",
            "meaning": "blood pressure",
        },
        {
            "term": "SpO2",
            "type": "Clinical Measurement",
            "meaning": "oxygen saturation measured by pulse oximetry",
        },
    ]
    for name, _value, meaning in cbc + chemistry:
        candidates.append(
            {
                "term": name,
                "type": "Laboratory Test",
                "meaning": meaning,
            }
        )
    if report_type in {
        "MRI",
        "CT",
        "X-ray",
        "Ultrasound",
        "Echocardiogram",
    }:
        candidates.append(
            {
                "term": item.finding,
                "type": "Imaging Finding",
                "meaning": (
                    f"the recorded imaging observation related to "
                    f"{item.condition}"
                ),
            }
        )
    device = DEVICE_DETAILS.get(item.condition)
    if device is not None:
        candidates.append(
            {
                "term": device[0],
                "type": "Medical Device",
                "meaning": device[1],
            }
        )
    for entity in candidates:
        entity["meaning"] = _vary_entity_meaning(
            entity["type"],
            entity["meaning"],
            rng,
        )
    return _deduplicate_entities(candidates, report)


def _render_simplification(
    facts: list[Fact],
    level: str,
    rng: random.Random,
) -> str:
    parts = [getattr(fact, level) for fact in facts]
    if level == "clinical":
        separator = rng.choice((" ", "\n", "\n\n"))
        return separator.join(parts)
    if level == "general" and len(parts) > 5 and rng.random() < 0.5:
        midpoint = len(parts) // 2
        return " ".join(parts[:midpoint]) + "\n\n" + " ".join(
            parts[midpoint:]
        )
    return " ".join(parts)


def _patient_summary(
    age: int,
    gender: str,
    item: Scenario,
    rng: random.Random,
) -> str:
    plain_assessment = _child_friendly_text(item.assessment)
    meaning = _child_friendly_text(item.condition_meaning)
    opening = rng.choice(SUMMARY_OPENINGS)
    assessment_lower = (
        plain_assessment[0].lower() + plain_assessment[1:]
    )
    templates = (
        (
            "The report evaluates {symptom}. {assessment} "
            "{condition} means {meaning}."
        ),
        (
            "This {age}-year-old {gender} was seen for {symptom}. "
            "The main message is: {assessment} The term {condition} "
            "describes {meaning}."
        ),
        (
            "The report explains the clinical situation behind {symptom}. "
            "{assessment} In everyday words, {condition} is {meaning}."
        ),
        (
            "For this {age}-year-old {gender}, the main concern was "
            "{symptom}. {assessment} In plain language, {condition} "
            "means {meaning}."
        ),
        (
            "{opening} {assessment_lower} The evaluation connects "
            "{symptom} with {condition}, which means {meaning}."
        ),
    )
    return rng.choice(templates).format(
        opening=opening,
        age=age,
        gender=gender,
        symptom=item.symptom_meaning,
        assessment=plain_assessment,
        assessment_lower=assessment_lower,
        condition=item.condition,
        meaning=meaning,
    )


def build_sample(index: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one fact-aligned training sample and its compact metadata."""
    rng = random.Random(f"{seed}:{index}")
    report_type = REPORT_TYPES[index % len(REPORT_TYPES)]
    item = select_scenario(index, report_type)
    difficulty = DIFFICULTIES[(index // 7 + index) % len(DIFFICULTIES)]
    age, gender = demographics(item, rng)
    duration = rng.choice(
        (
            "2 days",
            "5 days",
            "3 weeks",
            "2 months",
            "6 months",
            "1 year",
        )
    )
    encounter_date = (
        date(2012, 11, 1) + timedelta(days=index)
    ).isoformat()
    values = vital_values(item, age, rng)
    cbc = cbc_results(item.lab_pattern, rng)
    chemistry = chemistry_results(item.lab_pattern, rng)

    patient = _patient_fact(
        age,
        gender,
        item,
        duration,
        encounter_date,
    )
    primary = _report_type_fact(
        report_type,
        item,
        values,
        cbc,
        chemistry,
        rng,
    )
    assessment = _assessment_fact(item, rng)
    medication = _medication_fact(item, rng)
    vitals = _vitals_fact(values)
    supplemental_lab = (
        _laboratory_fact("Biochemistry", chemistry, rng)
        if report_type not in {"Blood Test", "Biochemistry"}
        else _laboratory_fact("CBC", cbc, rng)
    )
    context = _context_facts(report_type, rng)
    device = _device_fact(item)
    recommendation = _recommendation_fact(item, report_type, rng)

    medication_facts = [] if report_type == "Prescription" else [medication]
    facts = [patient, primary, assessment, *medication_facts]
    if difficulty in {"Medium", "Long"}:
        facts = [
            patient,
            vitals,
            primary,
            supplemental_lab,
            assessment,
            *medication_facts,
            recommendation,
        ]
    if difficulty == "Long":
        facts = [
            patient,
            context[0],
            vitals,
            primary,
            supplemental_lab,
            assessment,
            *medication_facts,
            context[1],
            context[2],
            recommendation,
        ]
        if device is not None:
            facts.insert(-1, device)

    report, report_style = _render_report(
        report_type,
        item.specialty,
        difficulty,
        facts,
        rng,
    )
    clinical = _render_simplification(facts, "clinical", rng)
    general = _render_simplification(facts, "general", rng)
    child = _render_simplification(facts, "child", rng)
    summary = _patient_summary(age, gender, item, rng)
    entities = _build_entities(
        item,
        report_type,
        report,
        cbc,
        chemistry,
        rng,
    )

    sample = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {
                "role": "assistant",
                "content": {
                    "report": report,
                    "summary": summary,
                    "simplification": {
                        "clinical": clinical,
                        "general": general,
                        "child": child,
                    },
                    "entities": entities,
                },
            },
        ]
    }
    metadata: dict[str, Any] = {
        "specialty": item.specialty,
        "condition": item.condition,
        "report_type": report_type,
        "difficulty": difficulty,
        "rarity": "rare_or_uncommon" if item.rare else "common",
        "gender": gender,
        "age": age,
        "report_style": report_style,
    }
    validate_sample(sample)
    return sample, metadata


def _nonempty_string(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")


def _estimated_syllables(word: str) -> int:
    normalized = re.sub(r"[^a-z]", "", word.lower())
    if not normalized:
        return 1
    groups = re.findall(r"[aeiouy]+", normalized)
    count = len(groups)
    if normalized.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def child_readability_metrics(text: str) -> dict[str, float]:
    """Estimate whether child output remains approachable despite terms."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return {
            "grade": 0.0,
            "average_sentence_words": 0.0,
            "longest_sentence_words": 0.0,
        }
    sentence_lengths = [
        len(re.findall(r"[A-Za-z]+", sentence))
        for sentence in sentences
    ] or [len(words)]
    syllables = sum(_estimated_syllables(word) for word in words)
    grade = (
        0.39 * (len(words) / max(1, len(sentences)))
        + 11.8 * (syllables / len(words))
        - 15.59
    )
    return {
        "grade": round(grade, 2),
        "average_sentence_words": round(
            sum(sentence_lengths) / len(sentence_lengths),
            2,
        ),
        "longest_sentence_words": float(max(sentence_lengths)),
    }


def validate_sample(sample: Any) -> None:
    """Strictly validate one completed training example."""
    if not isinstance(sample, dict) or set(sample) != {"messages"}:
        raise ValueError("Sample must contain only the messages field.")
    messages = sample["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("messages must contain exactly three items.")
    expected_roles = ("system", "user", "assistant")
    for position, role in enumerate(expected_roles):
        message = messages[position]
        if not isinstance(message, dict):
            raise ValueError(f"Message {position} must be an object.")
        if set(message) != {"role", "content"}:
            raise ValueError(f"Message {position} has an invalid schema.")
        if message["role"] != role:
            raise ValueError(f"Message {position} must have role {role}.")

    if messages[0]["content"] != SYSTEM_PROMPT:
        raise ValueError("System prompt does not match the dataset contract.")
    if messages[1]["content"] != USER_PROMPT:
        raise ValueError("User prompt does not match the dataset contract.")

    content = messages[2]["content"]
    if not isinstance(content, dict):
        raise ValueError("Assistant content must be an object.")
    expected_content = {"report", "summary", "simplification", "entities"}
    if set(content) != expected_content:
        raise ValueError("Assistant content has an invalid schema.")
    _nonempty_string(content["report"], "report")
    _nonempty_string(content["summary"], "summary")

    simplification = content["simplification"]
    if not isinstance(simplification, dict):
        raise ValueError("simplification must be an object.")
    if set(simplification) != {"clinical", "general", "child"}:
        raise ValueError("simplification has an invalid schema.")
    for level in ("clinical", "general", "child"):
        _nonempty_string(simplification[level], f"simplification.{level}")

    entities = content["entities"]
    if not isinstance(entities, list):
        raise ValueError("entities must be an array.")
    report_lower = content["report"].lower()
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("Every entity must be an object.")
        if set(entity) != {"term", "type", "meaning"}:
            raise ValueError("Entity has an invalid schema.")
        _nonempty_string(entity["term"], "entity.term")
        _nonempty_string(entity["meaning"], "entity.meaning")
        if entity["type"] not in ALLOWED_ENTITY_TYPES:
            raise ValueError(f"Invalid entity type: {entity['type']}")
        if entity["term"].lower() not in report_lower:
            raise ValueError(
                f"Entity term is absent from report: {entity['term']}"
            )

    number_pattern = re.compile(
        r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])"
    )
    report_numbers = set(number_pattern.findall(content["report"]))
    for level in ("clinical", "general", "child"):
        level_numbers = set(number_pattern.findall(simplification[level]))
        missing_numbers = report_numbers.difference(level_numbers)
        if missing_numbers:
            raise ValueError(
                f"{level} simplification dropped values: "
                f"{sorted(missing_numbers)}"
            )

    diagnoses = re.findall(
        r"Clinical consideration:\s*([^.]+)\.",
        content["report"],
        flags=re.IGNORECASE,
    )
    if not diagnoses:
        raise ValueError("Report has no detectable clinical consideration.")
    for diagnosis in diagnoses:
        for level in ("clinical", "general", "child"):
            if diagnosis.lower() not in simplification[level].lower():
                raise ValueError(
                    f"{level} simplification dropped diagnosis: {diagnosis}"
                )

    readability = child_readability_metrics(simplification["child"])
    if readability["grade"] > 13:
        raise ValueError(
            "Child simplification is too difficult: estimated grade "
            f"{readability['grade']}."
        )
    if readability["average_sentence_words"] > 15:
        raise ValueError(
            "Child simplification sentences are too long on average."
        )
    if readability["longest_sentence_words"] > 28:
        raise ValueError(
            "Child simplification contains an excessively long sentence."
        )


class CorpusQualityTracker:
    """Track linguistic diversity without retaining generated samples."""

    REPETITIVE_PHRASES: Final[tuple[str, ...]] = (
        "the findings are consistent with",
        "in plain language",
        "the recorded finding was",
        "the report lists",
        "put simply",
    )

    def __init__(self) -> None:
        self.examples = 0
        self.total_sentences = 0
        self.entity_meanings: dict[
            tuple[str, str],
            Counter[str],
        ] = defaultdict(Counter)
        self.sentence_openings: Counter[str] = Counter()
        self.phrase_counts: Counter[str] = Counter()
        self.child_grade_sum = 0.0
        self.uncertain_examples = 0
        self.recommendation_examples = 0
        self.report_styles: Counter[str] = Counter()

    @staticmethod
    def _normalize_meaning(meaning: str) -> str:
        return re.sub(r"\s+", " ", meaning.strip().lower())

    @staticmethod
    def _sentence_opening(sentence: str) -> str:
        normalized = re.sub(r"\d+(?:\.\d+)?", "<n>", sentence.lower())
        words = re.findall(r"[a-z<>]+", normalized)
        return " ".join(words[:5])

    def observe(self, sample: dict[str, Any]) -> None:
        self.examples += 1
        content = sample["messages"][2]["content"]
        report = content["report"]
        report_lower = report.lower()
        uncertainty_markers = (
            "likely",
            "probable",
            "suspected",
            "possible",
            "cannot exclude",
            "suggestive of",
            "appears consistent with",
            "favours",
            "likely represents",
            "differential diagnosis includes",
            "results suggest",
            "findings support",
            "evidence points toward",
        )
        if any(marker in report_lower for marker in uncertainty_markers):
            self.uncertain_examples += 1
        if "recommendations:" in report_lower:
            self.recommendation_examples += 1
        if "subjective:" in report_lower:
            self.report_styles["soap"] += 1
        elif "reason for admission:" in report_lower:
            self.report_styles["discharge"] += 1
        elif "microscopic description:" in report_lower:
            self.report_styles["pathology"] += 1
        elif "clinical history:" in report_lower:
            self.report_styles["radiology"] += 1
        elif "dictated report:" in report_lower:
            self.report_styles["dictated"] += 1
        elif "\n\n" in report:
            self.report_styles["narrative"] += 1
        else:
            self.report_styles["sectioned"] += 1
        for entity in content["entities"]:
            key = (entity["term"].lower(), entity["type"])
            meaning = self._normalize_meaning(entity["meaning"])
            self.entity_meanings[key][meaning] += 1

        combined = " ".join(
            (
                content["summary"],
                content["simplification"]["clinical"],
                content["simplification"]["general"],
                content["simplification"]["child"],
            )
        )
        sentences = [
            sentence.strip()
            for sentence in re.split(r"[.!?]+", combined)
            if sentence.strip()
        ]
        self.total_sentences += len(sentences)
        for sentence in sentences:
            opening = self._sentence_opening(sentence)
            if opening:
                self.sentence_openings[opening] += 1
        lower = combined.lower()
        for phrase in self.REPETITIVE_PHRASES:
            self.phrase_counts[phrase] += lower.count(phrase)
        metrics = child_readability_metrics(
            content["simplification"]["child"]
        )
        self.child_grade_sum += metrics["grade"]

    def finalize(self) -> dict[str, Any]:
        if self.examples == 0:
            raise ValueError("Corpus contains no examples.")

        explanation_failures: list[str] = []
        checked_entities = 0
        minimum_distinct_meanings: int | None = None
        maximum_dominant_share = 0.0
        for (term, entity_type), meanings in self.entity_meanings.items():
            occurrences = sum(meanings.values())
            if occurrences < 20:
                continue
            checked_entities += 1
            dominant_share = max(meanings.values()) / occurrences
            minimum_distinct_meanings = (
                len(meanings)
                if minimum_distinct_meanings is None
                else min(minimum_distinct_meanings, len(meanings))
            )
            maximum_dominant_share = max(
                maximum_dominant_share,
                dominant_share,
            )
            if len(meanings) < 3 or dominant_share > 0.65:
                explanation_failures.append(
                    f"{term}/{entity_type}: {len(meanings)} meanings, "
                    f"dominant share {dominant_share:.1%}"
                )
        if explanation_failures:
            preview = "; ".join(explanation_failures[:10])
            raise ValueError(
                "Entity explanation diversity validation failed: " + preview
            )

        most_common_opening, opening_count = (
            self.sentence_openings.most_common(1)[0]
        )
        opening_share = opening_count / max(1, self.total_sentences)
        if opening_share > 0.20:
            raise ValueError(
                "Repeated sentence-template validation failed: "
                f"{most_common_opening!r} appears in "
                f"{opening_share:.1%} of sentences."
            )

        phrase_rates = {
            phrase: count / self.examples
            for phrase, count in self.phrase_counts.items()
        }
        excessive = {
            phrase: rate
            for phrase, rate in phrase_rates.items()
            if rate > 0.35
        }
        if excessive:
            raise ValueError(
                "Excessive phrase repetition detected: "
                + ", ".join(
                    f"{phrase!r}={rate:.1%}"
                    for phrase, rate in excessive.items()
                )
            )

        average_grade = self.child_grade_sum / self.examples
        if not 3.0 <= average_grade <= 9.5:
            raise ValueError(
                "Corpus child readability is outside the target range: "
                f"estimated grade {average_grade:.2f}."
            )
        uncertainty_share = self.uncertain_examples / self.examples
        if uncertainty_share < 0.25:
            raise ValueError(
                "Too few reports contain realistic diagnostic uncertainty: "
                f"{uncertainty_share:.1%}."
            )
        if uncertainty_share > 0.85:
            raise ValueError(
                "Diagnostic uncertainty is unrealistically overused: "
                f"{uncertainty_share:.1%}."
            )
        recommendation_share = (
            self.recommendation_examples / self.examples
        )
        if recommendation_share < 0.55:
            raise ValueError(
                "Too few reports contain documented recommendations: "
                f"{recommendation_share:.1%}."
            )
        if len(self.report_styles) < 6:
            raise ValueError(
                "Insufficient clinician-documentation style diversity."
            )
        dominant_style, dominant_style_count = (
            self.report_styles.most_common(1)[0]
        )
        dominant_style_share = dominant_style_count / self.examples
        if dominant_style_share > 0.45:
            raise ValueError(
                "One report style is excessively dominant: "
                f"{dominant_style}={dominant_style_share:.1%}."
            )
        return {
            "entities_checked_for_explanation_diversity": checked_entities,
            "minimum_distinct_meanings_per_frequent_entity": (
                minimum_distinct_meanings or 0
            ),
            "maximum_dominant_entity_meaning_share": round(
                maximum_dominant_share,
                4,
            ),
            "most_common_sentence_opening": most_common_opening,
            "most_common_sentence_opening_share": round(opening_share, 4),
            "repetitive_phrase_rates": {
                phrase: round(rate, 4)
                for phrase, rate in phrase_rates.items()
            },
            "average_child_reading_grade": round(average_grade, 2),
            "reports_with_uncertainty_share": round(
                uncertainty_share,
                4,
            ),
            "reports_with_recommendations_share": round(
                recommendation_share,
                4,
            ),
            "detected_report_style_counts": _counter_dict(
                self.report_styles
            ),
        }


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def generate_dataset(
    output: Path,
    manifest_path: Path,
    count: int,
    seed: int,
) -> dict[str, Any]:
    """Stream validated examples to an atomically committed JSONL file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    digest = hashlib.sha256()
    specialty_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    report_type_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    rarity_counts: Counter[str] = Counter()
    gender_counts: Counter[str] = Counter()
    entity_type_counts: Counter[str] = Counter()
    report_style_counts: Counter[str] = Counter()
    quality_tracker = CorpusQualityTracker()
    min_report_words: int | None = None
    max_report_words = 0
    minimum_age: int | None = None
    maximum_age = 0

    with temporary.open("wb") as handle:
        for index in range(count):
            sample, metadata = build_sample(index, seed)
            quality_tracker.observe(sample)
            encoded = (
                json.dumps(
                    sample,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)

            specialty_counts[metadata["specialty"]] += 1
            condition_counts[metadata["condition"]] += 1
            report_type_counts[metadata["report_type"]] += 1
            difficulty_counts[metadata["difficulty"]] += 1
            rarity_counts[metadata["rarity"]] += 1
            gender_counts[metadata["gender"]] += 1
            report_style_counts[metadata["report_style"]] += 1
            entities = sample["messages"][2]["content"]["entities"]
            entity_type_counts.update(
                entity["type"] for entity in entities
            )
            age = int(metadata["age"])
            minimum_age = age if minimum_age is None else min(minimum_age, age)
            maximum_age = max(maximum_age, age)
            report = sample["messages"][2]["content"]["report"]
            word_count = len(report.split())
            min_report_words = (
                word_count
                if min_report_words is None
                else min(min_report_words, word_count)
            )
            max_report_words = max(max_report_words, word_count)
        handle.flush()
        os.fsync(handle.fileno())
    quality_metrics = quality_tracker.finalize()
    os.replace(temporary, output)

    manifest = {
        "dataset": output.name,
        "synthetic_data_notice": (
            "All reports are synthetic and are not records of real patients."
        ),
        "examples": count,
        "seed": seed,
        "sha256": digest.hexdigest(),
        "bytes": output.stat().st_size,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "specialty_counts": _counter_dict(specialty_counts),
        "condition_counts": _counter_dict(condition_counts),
        "report_type_counts": _counter_dict(report_type_counts),
        "difficulty_counts": _counter_dict(difficulty_counts),
        "rarity_counts": _counter_dict(rarity_counts),
        "gender_counts": _counter_dict(gender_counts),
        "entity_type_counts": _counter_dict(entity_type_counts),
        "report_style_counts": _counter_dict(report_style_counts),
        "linguistic_quality_metrics": quality_metrics,
        "age_range": {
            "minimum": minimum_age or 0,
            "maximum": maximum_age,
        },
        "report_word_range": {
            "minimum": min_report_words or 0,
            "maximum": max_report_words,
        },
        "allowed_entity_types": sorted(ALLOWED_ENTITY_TYPES),
    }
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.tmp"
    )
    with temporary_manifest.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_manifest, manifest_path)
    return manifest


def verify_jsonl(path: Path, expected_count: int) -> dict[str, Any]:
    """Perform an independent streaming parse and schema verification pass."""
    count = 0
    digest = hashlib.sha256()
    entity_types: Counter[str] = Counter()
    quality_tracker = CorpusQualityTracker()
    with path.open("rb") as handle:
        for line_number, encoded in enumerate(handle, start=1):
            if not encoded.strip():
                raise ValueError(f"Blank JSONL record at line {line_number}.")
            digest.update(encoded)
            try:
                sample = json.loads(encoded)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc
            validate_sample(sample)
            quality_tracker.observe(sample)
            entities = sample["messages"][2]["content"]["entities"]
            entity_types.update(entity["type"] for entity in entities)
            count += 1
    if count != expected_count:
        raise ValueError(f"Expected {expected_count} lines, found {count}.")
    quality_metrics = quality_tracker.finalize()
    return {
        "examples": count,
        "sha256": digest.hexdigest(),
        "entity_type_counts": _counter_dict(entity_types),
        "linguistic_quality_metrics": quality_metrics,
    }


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a validated synthetic medical SFT JSONL corpus."
    )
    parser.add_argument(
        "--count",
        type=_positive_integer,
        default=5000,
        help="number of examples to generate (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260731,
        help="deterministic random seed (default: 20260731)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSONL output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"statistics manifest path (default: {DEFAULT_MANIFEST})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate_dataset(
        args.output,
        args.manifest,
        args.count,
        args.seed,
    )
    verification = verify_jsonl(args.output, args.count)
    if verification["sha256"] != manifest["sha256"]:
        raise RuntimeError("Post-write SHA-256 verification failed.")
    print(
        f"Generated {verification['examples']} examples at {args.output} "
        f"(sha256={verification['sha256']})."
    )
    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
