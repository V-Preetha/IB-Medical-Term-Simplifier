from transformers import AutoTokenizer, AutoModel

model_name = "thomas-sounack/BioClinical-ModernBERT-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print("Loaded successfully!")