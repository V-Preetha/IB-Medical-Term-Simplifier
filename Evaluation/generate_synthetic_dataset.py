import csv
import random
import re
from pathlib import Path


RNG = random.Random(42626)

SPECIALTIES = {
    "Cardiology": {
        "conditions": ["HTN", "CAD", "AF", "NSTEMI", "heart failure with preserved EF", "stable angina"],
        "medications": ["metoprolol", "atorvastatin", "aspirin", "clopidogrel", "furosemide", "apixaban"],
        "procedures": ["ECG", "echocardiogram", "coronary angiography", "CABG review", "stress test"],
        "anatomy": ["left ventricle", "right atrium", "coronary arteries", "mitral valve"],
        "labs": ["troponin", "BNP", "LDL", "INR", "potassium"],
    },
    "Neurology": {
        "conditions": ["TIA", "ischemic stroke", "migraine with aura", "focal seizure", "peripheral neuropathy"],
        "medications": ["levetiracetam", "aspirin", "atorvastatin", "gabapentin", "sumatriptan"],
        "procedures": ["MRI brain", "CT head", "EEG", "carotid Doppler", "lumbar puncture"],
        "anatomy": ["left MCA territory", "cerebellum", "basal ganglia", "optic nerve"],
        "labs": ["ESR", "CRP", "sodium", "glucose", "B12"],
    },
    "Pulmonology": {
        "conditions": ["COPD", "asthma exacerbation", "community-acquired pneumonia", "pleural effusion", "OSA"],
        "medications": ["salbutamol", "tiotropium", "budesonide", "azithromycin", "prednisone"],
        "procedures": ["CXR", "CT chest", "spirometry", "ABG", "bronchoscopy"],
        "anatomy": ["right lower lobe", "left hilum", "pleura", "bronchi"],
        "labs": ["SpO2", "pCO2", "WBC", "D-dimer", "procalcitonin"],
    },
    "Gastroenterology": {
        "conditions": ["GERD", "acute pancreatitis", "IBD flare", "cirrhosis", "upper GI bleed"],
        "medications": ["pantoprazole", "lactulose", "mesalamine", "ondansetron", "rifaximin"],
        "procedures": ["upper endoscopy", "colonoscopy", "abdominal ultrasound", "CT abdomen", "ERCP"],
        "anatomy": ["stomach", "duodenum", "colon", "liver", "pancreas"],
        "labs": ["ALT", "AST", "bilirubin", "lipase", "INR"],
    },
    "Nephrology": {
        "conditions": ["CKD stage 3", "AKI", "nephrotic-range proteinuria", "hyperkalemia", "renal colic"],
        "medications": ["losartan", "sevelamer", "furosemide", "sodium bicarbonate", "calcium gluconate"],
        "procedures": ["renal ultrasound", "urinalysis", "hemodialysis", "kidney biopsy", "CT KUB"],
        "anatomy": ["left kidney", "right ureter", "renal cortex", "bladder"],
        "labs": ["creatinine", "eGFR", "potassium", "urine protein", "BUN"],
    },
    "Endocrinology": {
        "conditions": ["DM", "hypothyroidism", "thyrotoxicosis", "DKA", "adrenal insufficiency"],
        "medications": ["metformin", "insulin glargine", "levothyroxine", "empagliflozin", "hydrocortisone"],
        "procedures": ["thyroid ultrasound", "HbA1c review", "glucose monitoring", "ACTH stimulation test"],
        "anatomy": ["thyroid gland", "pancreas", "adrenal glands", "pituitary"],
        "labs": ["HbA1c", "TSH", "free T4", "glucose", "ketones"],
    },
    "Oncology": {
        "conditions": ["breast carcinoma", "colon adenocarcinoma", "lymphoma", "lung nodule", "chemotherapy-induced anemia"],
        "medications": ["paclitaxel", "carboplatin", "ondansetron", "filgrastim", "tamoxifen"],
        "procedures": ["CT staging", "core biopsy", "PET-CT", "port placement", "pathology review"],
        "anatomy": ["axillary nodes", "sigmoid colon", "right lung", "bone marrow"],
        "labs": ["CEA", "CA-125", "hemoglobin", "platelets", "ANC"],
    },
    "Orthopedics": {
        "conditions": ["distal radius fracture", "osteoarthritis", "ACL tear", "lumbar spondylosis", "septic arthritis rule-out"],
        "medications": ["ibuprofen", "acetaminophen", "cefazolin", "tramadol", "enoxaparin"],
        "procedures": ["X-ray", "MRI knee", "ORIF", "arthroscopy", "joint aspiration"],
        "anatomy": ["right knee", "left wrist", "lumbar spine", "femoral head"],
        "labs": ["CRP", "ESR", "WBC", "vitamin D", "calcium"],
    },
    "Pediatrics": {
        "conditions": ["viral bronchiolitis", "otitis media", "febrile seizure", "dehydration", "type 1 DM"],
        "medications": ["amoxicillin", "ibuprofen", "oral rehydration solution", "insulin lispro", "albuterol"],
        "procedures": ["CBC", "urinalysis", "CXR", "nebulization", "growth chart review"],
        "anatomy": ["middle ear", "lungs", "abdomen", "oropharynx"],
        "labs": ["CBC", "glucose", "sodium", "ketones", "CRP"],
    },
    "Dermatology": {
        "conditions": ["atopic dermatitis", "cellulitis", "psoriasis", "urticaria", "melanocytic nevus"],
        "medications": ["hydrocortisone cream", "cephalexin", "cetirizine", "clobetasol", "mupirocin"],
        "procedures": ["skin biopsy", "dermoscopy", "wound swab", "patch testing", "cryotherapy"],
        "anatomy": ["forearm", "scalp", "lower leg", "trunk"],
        "labs": ["WBC", "eosinophils", "CRP", "wound culture"],
    },
    "Gynecology": {
        "conditions": ["uterine fibroid", "PID", "endometriosis", "ovarian cyst", "abnormal uterine bleeding"],
        "medications": ["doxycycline", "medroxyprogesterone", "ibuprofen", "tranexamic acid", "ceftriaxone"],
        "procedures": ["pelvic ultrasound", "Pap smear", "endometrial biopsy", "laparoscopy", "urine pregnancy test"],
        "anatomy": ["uterus", "right ovary", "cervix", "endometrium"],
        "labs": ["hemoglobin", "beta-hCG", "WBC", "CA-125", "TSH"],
    },
    "Psychiatry": {
        "conditions": ["major depressive disorder", "GAD", "bipolar disorder", "panic attacks", "insomnia"],
        "medications": ["sertraline", "escitalopram", "quetiapine", "lithium", "lorazepam"],
        "procedures": ["mental status exam", "PHQ-9", "GAD-7", "suicide risk assessment", "medication review"],
        "anatomy": ["sleep cycle", "cognition", "affect", "thought process"],
        "labs": ["TSH", "lithium level", "CBC", "CMP", "vitamin D"],
    },
    "Ophthalmology": {
        "conditions": ["diabetic retinopathy", "glaucoma", "cataract", "uveitis", "macular edema"],
        "medications": ["latanoprost", "timolol drops", "prednisolone drops", "cyclopentolate", "artificial tears"],
        "procedures": ["slit lamp exam", "OCT", "fundus photography", "tonometry", "visual field test"],
        "anatomy": ["retina", "optic disc", "cornea", "macula"],
        "labs": ["HbA1c", "IOP", "visual acuity", "CRP", "ESR"],
    },
    "ENT": {
        "conditions": ["sinusitis", "tonsillitis", "sensorineural hearing loss", "vertigo", "epistaxis"],
        "medications": ["amoxicillin-clavulanate", "fluticasone spray", "betahistine", "cetirizine", "tranexamic acid"],
        "procedures": ["nasal endoscopy", "audiometry", "CT sinuses", "tympanometry", "cautery"],
        "anatomy": ["maxillary sinus", "tympanic membrane", "nasal septum", "tonsils"],
        "labs": ["CBC", "CRP", "audiogram threshold", "INR"],
    },
    "Emergency Medicine": {
        "conditions": ["chest pain rule-out MI", "syncope", "sepsis evaluation", "ankle sprain", "acute abdomen"],
        "medications": ["IV fluids", "morphine", "ondansetron", "ceftriaxone", "nitroglycerin"],
        "procedures": ["ECG", "CT", "FAST ultrasound", "CBC", "lactate"],
        "anatomy": ["chest wall", "abdomen", "right ankle", "left flank"],
        "labs": ["troponin", "lactate", "CBC", "creatinine", "glucose"],
    },
    "Infectious Diseases": {
        "conditions": ["cellulitis", "UTI", "bacteremia", "dengue fever", "tuberculosis evaluation"],
        "medications": ["ceftriaxone", "vancomycin", "doxycycline", "oseltamivir", "isoniazid"],
        "procedures": ["blood culture", "urine culture", "CXR", "HIV test", "sputum AFB"],
        "anatomy": ["urinary tract", "right leg", "lungs", "lymph nodes"],
        "labs": ["WBC", "CRP", "platelets", "procalcitonin", "blood culture"],
    },
}

REPORT_TYPES = [
    "Discharge summary",
    "Radiology report",
    "CT report",
    "MRI report",
    "Ultrasound report",
    "Blood test report",
    "Pathology report",
    "Consultation note",
    "Progress note",
    "Operative note",
    "Prescription summary",
]

DIFFICULTIES = [
    ("Short", 30, 80),
    ("Medium", 80, 200),
    ("Long", 200, 500),
]

ABBREVIATIONS = ["HTN", "DM", "CAD", "COPD", "CKD", "MI", "AF", "CABG", "ECG", "MRI", "CT", "CBC", "HbA1c"]


def pick(items):
    return RNG.choice(items)


def dose_for(medication):
    units = ["mg", "mcg", "units", "mL"]
    if "insulin" in medication:
        return f"{RNG.randint(4, 24)} units"
    if "drops" in medication or "spray" in medication or "cream" in medication:
        return f"{RNG.randint(1, 2)} application"
    return f"{RNG.choice([2.5, 5, 10, 20, 25, 40, 50, 75, 100, 250, 500])} {pick(units[:2])}"


def vital_signs():
    systolic = RNG.randint(96, 168)
    diastolic = RNG.randint(58, 98)
    hr = RNG.randint(58, 122)
    rr = RNG.randint(12, 26)
    temp = round(RNG.uniform(36.1, 38.9), 1)
    spo2 = RNG.randint(91, 100)
    return f"Vitals: BP {systolic}/{diastolic}, HR {hr}, RR {rr}, T {temp} C, SpO2 {spo2}% on room air."


def lab_value(label):
    ranges = {
        "troponin": (0.01, 1.2, "ng/mL"),
        "BNP": (40, 980, "pg/mL"),
        "LDL": (62, 188, "mg/dL"),
        "INR": (0.9, 3.5, ""),
        "potassium": (3.1, 5.9, "mmol/L"),
        "ESR": (4, 78, "mm/hr"),
        "CRP": (1, 96, "mg/L"),
        "sodium": (128, 146, "mmol/L"),
        "glucose": (68, 318, "mg/dL"),
        "B12": (180, 880, "pg/mL"),
        "SpO2": (91, 100, "%"),
        "pCO2": (31, 68, "mmHg"),
        "WBC": (3.2, 18.6, "x10^9/L"),
        "D-dimer": (180, 1800, "ng/mL"),
        "procalcitonin": (0.02, 6.4, "ng/mL"),
        "ALT": (12, 220, "U/L"),
        "AST": (14, 210, "U/L"),
        "bilirubin": (0.3, 4.8, "mg/dL"),
        "lipase": (28, 980, "U/L"),
        "creatinine": (0.7, 4.8, "mg/dL"),
        "eGFR": (18, 104, "mL/min/1.73m2"),
        "urine protein": (0.1, 5.8, "g/day"),
        "BUN": (8, 72, "mg/dL"),
        "HbA1c": (5.2, 12.8, "%"),
        "TSH": (0.02, 12.6, "mIU/L"),
        "free T4": (0.5, 2.7, "ng/dL"),
        "ketones": (0.0, 5.2, "mmol/L"),
        "CEA": (1.1, 36.0, "ng/mL"),
        "CA-125": (8, 180, "U/mL"),
        "hemoglobin": (7.8, 15.8, "g/dL"),
        "platelets": (68, 510, "x10^9/L"),
        "ANC": (0.6, 8.9, "x10^9/L"),
        "vitamin D": (9, 54, "ng/mL"),
        "calcium": (8.1, 10.8, "mg/dL"),
        "CBC": (4.0, 15.0, "WBC x10^9/L"),
        "eosinophils": (0.1, 1.8, "x10^9/L"),
        "wound culture": (0, 1, "growth flag"),
        "beta-hCG": (0, 5400, "mIU/mL"),
        "lithium level": (0.3, 1.4, "mmol/L"),
        "CMP": (7, 28, "BUN mg/dL"),
        "IOP": (10, 31, "mmHg"),
        "visual acuity": (20, 80, "/20 equivalent"),
        "audiogram threshold": (15, 70, "dB"),
        "lactate": (0.8, 5.8, "mmol/L"),
        "blood culture": (0, 1, "growth flag"),
    }
    low, high, unit = ranges.get(label, (1, 100, ""))
    if isinstance(low, int) and isinstance(high, int):
        value = RNG.randint(low, high)
    else:
        value = round(RNG.uniform(low, high), 1 if high > 10 else 2)
    return f"{label} {value} {unit}".strip()


def labs_for(spec):
    labels = RNG.sample(spec["labs"], k=min(3, len(spec["labs"])))
    if RNG.random() < 0.35 and "HbA1c" not in labels:
        labels.append("HbA1c")
    return "; ".join(lab_value(label) for label in labels)


def diagnosis_sentence(condition, secondary):
    variants = [
        f"Assessment favors {condition}; differential includes {secondary} and medication-related symptoms.",
        f"Primary diagnosis is {condition}, with {secondary} considered less likely after review.",
        f"Impression: {condition}. Also monitoring for {secondary} given overlapping clinical features.",
    ]
    return pick(variants)


def treatment_plan(meds, procedure):
    med1, med2 = RNG.sample(meds, k=min(2, len(meds)))
    return (
        f"Plan: continue {med1} {dose_for(med1)} daily, start {med2} {dose_for(med2)} as indicated, "
        f"repeat {procedure} if symptoms worsen, and arrange follow-up in {RNG.choice([1, 2, 4, 6, 8])} weeks."
    )


def report_core(specialty, report_type, difficulty, report_id):
    spec = SPECIALTIES[specialty]
    condition = pick(spec["conditions"])
    secondary = pick([c for c in spec["conditions"] if c != condition])
    procedure = pick(spec["procedures"])
    anatomy = pick(spec["anatomy"])
    med = pick(spec["medications"])
    abbrev = pick(ABBREVIATIONS)
    style = RNG.choice(["narrative", "shorthand", "bullets", "formal"])
    date = f"2026-{RNG.randint(1, 12):02d}-{RNG.randint(1, 28):02d}"
    duration = RNG.choice(["2 days", "1 week", "3 weeks", "6 months", "overnight"])
    labs = labs_for(spec)
    vitals = vital_signs()

    if report_type in {"Radiology report", "CT report", "MRI report", "Ultrasound report"}:
        modality = report_type.split()[0] if report_type != "Radiology report" else procedure
        base = (
            f"{modality} performed on {date} for {condition} symptoms. Findings show {anatomy} with "
            f"{RNG.choice(['mild', 'moderate', 'no acute', 'stable', 'focal'])} abnormality and no unexpected mass effect. "
            f"Comparison with prior exam notes {RNG.choice(['interval improvement', 'no significant change', 'slight progression'])}. "
            f"Impression: imaging is most consistent with {condition}; {secondary} remains a differential consideration."
        )
    elif report_type == "Blood test report":
        base = (
            f"Blood test panel dated {date}. {labs}. CBC and CMP reviewed where available. "
            f"Results support monitoring for {condition}; {secondary} is not excluded. "
            f"Clinical correlation advised with current medications including {med}."
        )
    elif report_type == "Pathology report":
        base = (
            f"Specimen from {anatomy} received on {date}. Microscopy shows {RNG.choice(['chronic inflammation', 'reactive change', 'benign tissue', 'atypical cells requiring correlation'])}. "
            f"No definite invasive malignancy identified unless otherwise clinically suspected. "
            f"Pathologic impression aligns with {condition}; correlate with {procedure} and labs: {labs}."
        )
    elif report_type == "Operative note":
        base = (
            f"Procedure: {procedure} for {condition}. The {anatomy} was inspected, prepared, and treated without immediate complication. "
            f"Estimated blood loss {RNG.randint(5, 250)} mL. Post-op plan includes {med} {dose_for(med)}, wound review, and monitoring labs: {labs}."
        )
    elif report_type == "Prescription summary":
        med2 = pick([m for m in spec["medications"] if m != med])
        base = (
            f"Prescription summary for {condition}: {med} {dose_for(med)} daily and {med2} {dose_for(med2)} as directed. "
            f"Review renal/hepatic dosing where relevant. Avoid duplicate therapy. Return sooner for worsening symptoms, rash, bleeding, or persistent fever."
        )
    elif report_type == "Discharge summary":
        base = (
            f"Admitted for {duration} with {condition} and history of {abbrev}. {vitals} "
            f"Key investigations: {procedure}; labs: {labs}. {diagnosis_sentence(condition, secondary)} "
            f"Discharged in stable condition. {treatment_plan(spec['medications'], procedure)}"
        )
    elif report_type == "Consultation note":
        base = (
            f"{specialty} consult requested for {condition}. Symptoms present for {duration}; exam localized to {anatomy}. "
            f"{vitals} Reviewed {procedure} and labs: {labs}. {diagnosis_sentence(condition, secondary)} "
            f"{treatment_plan(spec['medications'], procedure)}"
        )
    elif report_type == "Progress note":
        base = (
            f"Progress note: {condition} is {RNG.choice(['improving', 'stable', 'slightly worse', 'not yet controlled'])}. "
            f"Pt reports {RNG.choice(['less pain', 'ongoing fatigue', 'mild dizziness', 'improved breathing', 'poor sleep'])}. {vitals} "
            f"Lab trend: {labs}. Continue {med} {dose_for(med)} and reassess after {procedure}."
        )
    else:
        base = (
            f"Clinical note for {condition}. Exam focuses on {anatomy}; {procedure} reviewed. {vitals} "
            f"Labs: {labs}. {treatment_plan(spec['medications'], procedure)}"
        )

    if style == "bullets":
        base = base.replace(". ", ".\n- ")
        base = "- " + base
    elif style == "shorthand":
        base = base.replace("Patient", "Pt").replace("with", "w/").replace("without", "w/o")
        base += f" Rx reviewed; no med allergy documented in this synthetic note."
    elif style == "formal":
        base = f"Clinical documentation entry {report_id}. " + base + " The note is synthetic and intended for benchmark evaluation only."

    return base


def expand_to_length(text, specialty, difficulty):
    target_low, target_high = difficulty[1], difficulty[2]
    if difficulty[0] == "Short":
        target_words = RNG.randint(target_low, target_high)
    elif difficulty[0] == "Medium":
        target_words = RNG.randint(110, target_high)
    else:
        target_words = RNG.randint(220, target_high)
    spec = SPECIALTIES[specialty]
    med_for_reconciliation = pick(spec["medications"])
    additions = [
        f"History also notes {pick(ABBREVIATIONS)} but no acute decompensation during this encounter.",
        f"Medication reconciliation included {med_for_reconciliation} {dose_for(med_for_reconciliation)}; adherence was discussed.",
        f"Safety-net advice: seek urgent care for chest pain, severe dyspnea, syncope, new weakness, uncontrolled pain, or bleeding.",
        f"Care team documented shared decision-making and explained why {pick(spec['procedures'])} was or was not required immediately.",
        f"Follow-up should verify symptom trajectory, repeat labs ({labs_for(spec)}), and confirm that numerical values remain stable.",
        f"Exam details: {pick(spec['anatomy'])} was described as {RNG.choice(['non-tender', 'mildly tender', 'stable', 'without focal deficit', 'mildly inflamed'])}.",
        f"Differential diagnosis remained broad but prioritized {pick(spec['conditions'])} over {pick(spec['conditions'])} based on available data.",
        f"The clinician documented medication risks, return precautions, and rationale for avoiding unnecessary escalation at this visit.",
        f"No contradictory diagnosis was recorded; the working impression, treatment plan, and follow-up instructions remained aligned.",
    ]
    while word_count(text) < target_words:
        text += " " + pick(additions)
    if word_count(text) > target_high:
        words = text.split()
        text = " ".join(words[: max(target_low, target_high - 12)]) + "."
    return text


def maybe_ocr_artifact(text):
    if RNG.random() >= 0.06:
        return text
    replacements = [
        ("O", "0"),
        ("l", "1"),
        ("BP", "B/P"),
        ("follow-up", "fo11ow-up"),
        ("stable", "stab1e"),
    ]
    for old, new in RNG.sample(replacements, k=2):
        text = text.replace(old, new, 1)
    if RNG.random() < 0.5:
        text = text.replace(". ", ".  \n", 2)
    return text


def word_count(text):
    return len(re.findall(r"\b[\w/%.-]+\b", text))


def csv_escape_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["report_id", "specialty", "report_type", "difficulty", "report"],
        )
        writer.writeheader()
        writer.writerows(rows)


def single_line(text):
    return re.sub(r"\s*\r?\n\s*", " | ", text).strip()


def generate_rows(count=1000):
    rows = []
    seen = set()
    specialties = list(SPECIALTIES)
    for report_id in range(1, count + 1):
        specialty = specialties[(report_id - 1) % len(specialties)]
        report_type = REPORT_TYPES[((report_id - 1) // len(specialties) + report_id) % len(REPORT_TYPES)]
        difficulty = DIFFICULTIES[(report_id - 1) % len(DIFFICULTIES)]
        text = report_core(specialty, report_type, difficulty, report_id)
        text = expand_to_length(text, specialty, difficulty)
        text = maybe_ocr_artifact(text)
        text = single_line(text)

        signature = re.sub(r"\d+", "#", text.lower())
        attempts = 0
        while signature in seen and attempts < 5:
            text = report_core(specialty, report_type, difficulty, report_id + attempts + 10000)
            text = expand_to_length(text, specialty, difficulty)
            text = single_line(text)
            signature = re.sub(r"\d+", "#", text.lower())
            attempts += 1
        seen.add(signature)

        rows.append(
            {
                "report_id": report_id,
                "specialty": specialty,
                "report_type": report_type,
                "difficulty": difficulty[0],
                "report": text,
            }
        )
    return rows


def main():
    output = Path(__file__).resolve().parent / "data" / "synthetic_medical_reports_singleline.csv"
    rows = generate_rows(1000)
    csv_escape_rows(rows, output)
    print(f"wrote={output}")
    print(f"rows={len(rows)}")
    print(f"specialties={len(set(row['specialty'] for row in rows))}")
    print(f"report_types={len(set(row['report_type'] for row in rows))}")
    print(f"min_words={min(word_count(row['report']) for row in rows)}")
    print(f"max_words={max(word_count(row['report']) for row in rows)}")


if __name__ == "__main__":
    main()
