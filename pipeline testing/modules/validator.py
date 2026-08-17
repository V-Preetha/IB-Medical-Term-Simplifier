"""IBM Granite validation with deterministic value checks."""
import json,re
from typing import Any
class GraniteValidator:
    VALUES=re.compile(r"\b(?:\d+(?:\.\d+)?(?:\s?(?:mg|mcg|g|mL|mmol/L|mg/dL|%|bpm|mmHg))?|no|not|without)\b",re.I)
    def __init__(self,model_name:str,max_new_tokens:int=384)->None:self.model_name,self.max_new_tokens,self._tokenizer,self._model=model_name,max_new_tokens,None,None
    def _load(self)->None:
        if self._model is not None:return
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self._tokenizer=AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True); self._model=AutoModelForCausalLM.from_pretrained(self.model_name,torch_dtype="auto",device_map="auto",trust_remote_code=True).eval()
    def validate(self,original:str,simplified:str,entities:list[dict[str,Any]],a:list[float],b:list[float])->dict[str,Any]:
        import torch
        similarity=float(torch.nn.functional.cosine_similarity(torch.tensor(a),torch.tensor(b),dim=0)); missing=sorted(set(x.casefold() for x in self.VALUES.findall(original))-set(x.casefold() for x in self.VALUES.findall(simplified)))
        prompt='Return JSON only: {"passed":bool,"hallucination_score":0..1,"issues":[strings]}. Flag invented facts or changed diagnoses, medication, doses, numbers, and negations.\nORIGINAL:\n'+original+'\nSIMPLIFIED:\n'+simplified
        try:
            self._load(); rendered=self._tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True); inputs=self._tokenizer(rendered,return_tensors="pt").to(self._model.device)
            with torch.inference_mode(): ids=self._model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False)
            raw=self._tokenizer.decode(ids[0,inputs.input_ids.shape[1]:],skip_special_tokens=True); match=re.search(r"\{.*\}",raw,re.S); judged=json.loads(match.group(0) if match else raw)
        except Exception as exc: judged={"passed":False,"hallucination_score":1.0,"issues":[f"Granite validation failed: {exc}"]}
        issues=[str(x) for x in judged.get("issues",[])];
        if missing:issues.append("Missing or changed values/negations: "+", ".join(missing))
        score=min(1,max(0,float(judged.get("hallucination_score",1))))
        return {"passed":bool(judged.get("passed")) and not missing and similarity>=.70,"hallucination_score":round(score,4),"semantic_similarity":round(similarity,4),"issues":issues,"validator_model":self.model_name}
