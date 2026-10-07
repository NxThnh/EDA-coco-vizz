"""Run the specified 20-image TinyLLaVA output-only dose probe on dev."""
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PYTHON=ROOT/'.venv-eda/Scripts/python.exe'
OUT=ROOT/'experiment_outputs/tiny_alpha_probe20_v2'
OUT.mkdir(parents=True,exist_ok=True)

for alpha in (0,.02,.05,.10,.20,.35,.50):
    output=OUT/f'tiny_pad_alpha_{alpha:.3f}.jsonl'
    command=[str(PYTHON),'vlm_experiment.py','--model','tiny',
             '--partition','dev_common_unmatched500','--condition','detailed512',
             '--limit','20','--layers','5','18','--alpha',str(alpha),
             '--output',str(output),'--resume']
    print('RUN',alpha,flush=True)
    subprocess.run(command,cwd=ROOT,check=True)
print('TINY_ALPHA_20_COMPLETE',flush=True)
