from pathlib import Path
from types import SimpleNamespace
from modules.evaluator import ReportEvaluator
from modules.preprocessing import MedicalTextPreprocessor
from pipeline import MedicalSimplifierPipeline
class Extractor:
    def extract(self,path):return SimpleNamespace(file_type="text",method="direct_text",page_count=1,text="Assessment: Hypertension. BP 150/95 mmHg. Take amlodipine 5 mg.")
class NER:
    model_name="fake-openmed"
    def detect(self,text):
        start=text.index("Hypertension");return [{"term":"Hypertension","entity_type":"disease","start":start,"end":start+12,"confidence":.95}]
class Embedder:
    model_name="fake-modernbert"
    def embed(self,text):return [1.0,0.0]
class Simplifier:
    model_name="fake-qwen"
    def simplify(self,*args):return "High blood pressure (Hypertension). BP 150/95 mmHg. Take amlodipine 5 mg."
    def explain_terms(self,e):return [{"term":"Hypertension","meaning":"High blood pressure.","entity_type":"disease","confidence":.95}]
class Validator:
    model_name="fake-granite"
    def validate(self,*args):return {"passed":True,"hallucination_score":0.0,"semantic_similarity":1.0,"issues":[],"validator_model":self.model_name}
def test_pipeline_writes_outputs(tmp_path:Path):
    source=tmp_path/"sample.txt";source.write_text("x");pipeline=MedicalSimplifierPipeline(output_dir=tmp_path/"outputs",extractor=Extractor(),preprocessor=MedicalTextPreprocessor(),ner=NER(),embedder=Embedder(),simplifier=Simplifier(),validator=Validator(),evaluator=ReportEvaluator());result=pipeline.run(source)
    assert result["validation"]["passed"] and all(Path(p).is_file() for p in result["output_files"].values())
