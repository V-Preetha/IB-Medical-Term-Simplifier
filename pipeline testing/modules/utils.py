"""Logging and output serialization."""
import json, logging, re
from pathlib import Path
from typing import Any

def configure_logging() -> None: logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
def _safe(value: object) -> str: return str(value).replace("|","\\|").replace("\n"," ")
def save_outputs(result: dict[str,Any], output_dir: Path) -> dict[str,str]:
    output_dir.mkdir(parents=True, exist_ok=True); stem = re.sub(r"[^\w-]+","_",Path(result["input_file"]).stem)
    txt, js, md = output_dir/f"{stem}_simplified.txt", output_dir/f"{stem}_result.json", output_dir/f"{stem}_report.md"
    paths = {"simplified_text":str(txt),"json":str(js),"markdown":str(md)}; result["output_files"] = paths
    txt.write_text(result["simplified_text"].strip()+"\n",encoding="utf-8")
    js.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    rows = "\n".join(f"| {_safe(t['term'])} | {_safe(t['meaning'])} | {t['confidence']:.2f} |" for t in result["highlighted_difficult_terms"]) or "| — | None detected | — |"
    issues = "\n".join(f"- {_safe(x)}" for x in result["validation"]["issues"]) or "- None"
    md.write_text(f"# Medical Report Simplification\n\n## Original Text\n\n{result['raw_text']}\n\n## Simplified Explanation\n\n{result['simplified_text']}\n\n## Difficult Medical Terms\n\n| Term | Meaning | Confidence |\n|---|---|---:|\n{rows}\n\n## Validation Results\n\n- Passed: **{result['validation']['passed']}**\n- Hallucination score: **{result['validation']['hallucination_score']:.3f}**\n- Semantic similarity: **{result['validation']['semantic_similarity']:.3f}**\n\n### Issues\n\n{issues}\n\n## Evaluation Scores\n\n```json\n{json.dumps(result['evaluation_scores'],indent=2)}\n```\n",encoding="utf-8")
    return paths
