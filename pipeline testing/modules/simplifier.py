"""Qwen simplification and term explanations."""
import json,re
from typing import Any
class QwenSimplifier:
    def __init__(self,model_name:str,max_new_tokens:int=1024)->None:self.model_name,self.max_new_tokens,self._tokenizer,self._model=model_name,max_new_tokens,None,None
    def _load(self)->None:
        if self._model is not None:return
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self._tokenizer=AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True); self._model=AutoModelForCausalLM.from_pretrained(self.model_name,torch_dtype="auto",device_map="auto",trust_remote_code=True).eval()
    def _generate(self,system:str,user:str)->str:
        self._load(); import torch
        prompt=self._tokenizer.apply_chat_template([{"role":"system","content":system},{"role":"user","content":user}],tokenize=False,add_generation_prompt=True,enable_thinking=False); inputs=self._tokenizer(prompt,return_tensors="pt").to(self._model.device)
        with torch.inference_mode(): ids=self._model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False)
        return self._tokenizer.decode(ids[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()
    def simplify(self,text:str,entities:list[dict[str,Any]],embedding:list[float])->str:
        context=[{"term":e["term"],"type":e["entity_type"]} for e in entities]
        return self._generate("Simplify for patients. Never add facts or advice. Preserve medications, dosages, diagnoses, numbers, dates, and negations exactly. Return only the report.",f"Entities: {json.dumps(context)}\nEmbedding dimensions: {len(embedding)}\nREPORT:\n{text}")
    def explain_terms(self,entities:list[dict[str,Any]])->list[dict[str,Any]]:
        unique={e["term"].casefold():e for e in reversed(entities)}
        if not unique:return []
        raw=self._generate("Define terms briefly for patients. Return only JSON mapping each exact term to its meaning.",json.dumps([{"term":e["term"],"type":e["entity_type"]} for e in unique.values()])); match=re.search(r"\{.*\}",raw,re.S)
        try: meanings=json.loads(match.group(0) if match else raw)
        except ValueError: meanings={}
        return [{"term":e["term"],"meaning":str(meanings.get(e["term"],"Ask a clinician what this term means in this context.")),"entity_type":e["entity_type"],"confidence":e["confidence"]} for e in unique.values()]
