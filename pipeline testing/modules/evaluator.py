"""Readability and quality metrics."""
import re
from statistics import mean
from typing import Any
def _syllables(w:str)->int:
    n=len(re.findall(r"[aeiouy]+",w.lower())); return max(1,n-(1 if w.lower().endswith("e") and n>1 else 0))
class ReportEvaluator:
    def evaluate(self,original:str,simplified:str,entities:list[dict[str,Any]],validation:dict[str,Any])->dict[str,Any]:
        words=re.findall(r"[A-Za-z]+",simplified); wc=max(1,len(words)); sentences=max(1,len(re.findall(r"[.!?]+",simplified))); syllables=sum(_syllables(w) for w in words)
        ease=206.835-1.015*(wc/sentences)-84.6*(syllables/wc); grade=.39*(wc/sentences)+11.8*(syllables/wc)-15.59
        source,target=set(original.lower().split()),set(simplified.lower().split()); rouge=len(source&target)/max(1,len(source)); similarity=validation["semantic_similarity"]
        quality=100*(.4*similarity+.25*(1-validation["hallucination_score"])+.2*min(1,max(0,ease)/100)+.15*rouge)
        return {"readability":{"flesch_reading_ease":round(ease,2),"flesch_kincaid_grade":round(grade,2)},"semantic_similarity":similarity,"rouge_1_recall":round(rouge,4),"ner_statistics":{"entity_count":len(entities),"average_confidence":round(mean([e["confidence"] for e in entities]),4) if entities else 0.0},"validation_summary":"passed" if validation["passed"] else "needs_review","overall_quality_score":round(quality,2)}
