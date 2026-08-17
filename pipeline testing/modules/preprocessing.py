"""Conservative clinical text normalization."""
import re
import unicodedata

class MedicalTextPreprocessor:
    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text).replace("\x00", "").replace("\r", "\n")
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return re.sub(r"([,;:])(?=\S)", r"\1 ", text).strip()
    def sentences(self, text: str) -> list[str]:
        return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", text) if x.strip()]
