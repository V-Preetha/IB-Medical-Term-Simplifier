"""Edit INPUT_FILE, then run: python simplify_report.py"""
from pathlib import Path
from modules.utils import configure_logging
from pipeline import MedicalSimplifierPipeline
INPUT_FILE="reports/sample.txt"
def main()->None:
    configure_logging(); base=Path(__file__).resolve().parent; source=Path(INPUT_FILE)
    if not source.is_absolute():source=base/source
    result=MedicalSimplifierPipeline(output_dir=base/"outputs").run(source); print(result["simplified_text"])
if __name__=="__main__":main()
