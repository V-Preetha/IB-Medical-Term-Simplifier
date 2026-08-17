"""Framework-independent local pipeline orchestration."""
import logging
from pathlib import Path
from typing import Any
from modules.embedding import ClinicalEmbedder
from modules.evaluator import ReportEvaluator
from modules.extractor import TextExtractor
from modules.ner import OpenMedNER
from modules.preprocessing import MedicalTextPreprocessor
from modules.simplifier import QwenSimplifier
from modules.utils import save_outputs
from modules.validator import GraniteValidator
LOG=logging.getLogger(__name__)
class MedicalSimplifierPipeline:
    def __init__(self,*,output_dir:str|Path="outputs",extractor:Any=None,preprocessor:Any=None,ner:Any=None,embedder:Any=None,simplifier:Any=None,validator:Any=None,evaluator:Any=None)->None:
        self.output_dir=Path(output_dir); self.extractor=extractor or TextExtractor(); self.preprocessor=preprocessor or MedicalTextPreprocessor(); self.ner=ner or OpenMedNER("OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M"); self.embedder=embedder or ClinicalEmbedder("thomas-sounack/BioClinical-ModernBERT-base"); self.simplifier=simplifier or QwenSimplifier("Qwen/Qwen3-4B"); self.validator=validator or GraniteValidator("ibm-granite/granite-guardian-3.0-2b"); self.evaluator=evaluator or ReportEvaluator()
    def run(self,input_file:str|Path)->dict[str,Any]:
        path=Path(input_file).expanduser().resolve(); LOG.info("Processing %s",path); extracted=self.extractor.extract(path); clean=self.preprocessor.clean(extracted.text); entities=self.ner.detect(clean); original_vector=self.embedder.embed(clean); simplified=self.simplifier.simplify(clean,entities,original_vector).strip()
        if not simplified:raise RuntimeError("Qwen returned an empty simplification")
        terms=self.simplifier.explain_terms(entities); simplified_vector=self.embedder.embed(simplified); validation=self.validator.validate(clean,simplified,entities,original_vector,simplified_vector); evaluation=self.evaluator.evaluate(clean,simplified,entities,validation)
        result={"schema_version":"1.0","input_file":str(path),"input_detection":{"file_type":extracted.file_type,"extraction_method":extracted.method,"page_count":extracted.page_count},"raw_text":extracted.text,"clean_text":clean,"simplified_text":simplified,"medical_entities":entities,"highlighted_difficult_terms":terms,"embedding":{"model":self.embedder.model_name,"dimensions":len(original_vector),"vectors_saved":False},"models":{"ner":self.ner.model_name,"simplifier":self.simplifier.model_name,"validator":self.validator.model_name},"validation":validation,"evaluation_scores":evaluation}; save_outputs(result,self.output_dir); LOG.info("Complete; passed=%s",validation["passed"]); return result
