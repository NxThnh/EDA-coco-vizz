"""Measure TinyLLaVA output-length dose on the same ten locked dev images."""
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PYTHON=ROOT/'.venv-eda/Scripts/python.exe'
OUT=ROOT/'experiment_outputs/tiny_alpha_probe10'
OUT.mkdir(parents=True,exist_ok=True)

for alpha in (0,.02,.035,.05,.075,.10):
    label=f'{alpha:.3f}'
    output=OUT/f'tiny_pad_alpha_{label}.jsonl'
    command=[str(PYTHON),'vlm_experiment.py','--model','tiny',
             '--partition','dev_common_unmatched500','--condition','detailed512',
             '--limit','10','--layers','5','18','--alpha',str(alpha),
             '--output',str(output),'--resume']
    print('RUN',alpha,flush=True)
    subprocess.run(command,cwd=ROOT,check=True)
print('TINY_PROBE_COMPLETE',flush=True)
