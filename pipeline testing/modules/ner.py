"""OpenMed NER adapter."""
from typing import Any
class OpenMedNER:
    def __init__(self, model_name: str, device: int=-1)->None: self.model_name,self.device,self._pipe=model_name,device,None
    def _load(self)->Any:
        if self._pipe is None:
            try:
                from transformers import pipeline
                self._pipe=pipeline("token-classification",model=self.model_name,tokenizer=self.model_name,aggregation_strategy="simple",device=self.device,trust_remote_code=True)
            except Exception as exc: raise RuntimeError(f"Could not load OpenMed NER '{self.model_name}': {exc}") from exc
        return self._pipe
    def detect(self,text:str)->list[dict[str,Any]]:
        found=[]
        for item in self._load()(text):
            start,end=int(item["start"]),int(item["end"]); label=str(item.get("entity_group",item.get("entity","medical_term"))).removeprefix("B-").removeprefix("I-").lower()
            found.append({"term":text[start:end].strip(),"entity_type":label,"start":start,"end":end,"confidence":round(float(item.get("score",0)),4)})
        return [x for x in found if x["term"]]
