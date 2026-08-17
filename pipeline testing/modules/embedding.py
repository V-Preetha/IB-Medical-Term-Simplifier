"""BioClinical ModernBERT contextual embedding adapter."""
class ClinicalEmbedder:
    def __init__(self,model_name:str,max_length:int=8192)->None: self.model_name,self.max_length,self._tokenizer,self._model=model_name,max_length,None,None
    def _load(self)->None:
        if self._model is not None:return
        try:
            from transformers import AutoModel,AutoTokenizer
            self._tokenizer=AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True); self._model=AutoModel.from_pretrained(self.model_name,trust_remote_code=True).eval()
        except Exception as exc: raise RuntimeError(f"Could not load embedder '{self.model_name}': {exc}") from exc
    def embed(self,text:str)->list[float]:
        self._load(); import torch
        encoded=self._tokenizer(text,return_tensors="pt",truncation=True,max_length=self.max_length)
        with torch.inference_mode(): hidden=self._model(**encoded).last_hidden_state
        mask=encoded["attention_mask"].unsqueeze(-1)
        return ((hidden*mask).sum(1)/mask.sum(1).clamp(min=1))[0].float().cpu().tolist()
