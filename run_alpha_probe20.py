"""Run the fixed 20-image SmolVLM2 alpha probe on dev only."""
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PYTHON=ROOT/'.venv-eda/Scripts/python.exe'
OUT=ROOT/'experiment_outputs/alpha_probe20'
OUT.mkdir(parents=True,exist_ok=True)
MODES={
    'split_on':[],
    'split_768':['--smol-longest-edge','768'],
    'split_off':['--split-off'],
}
ALPHAS=(0,.02,.035,.05,.075,.10)

for mode,options in MODES.items():
    for alpha in ALPHAS:
        name=f'smol_{mode}_alpha_{alpha:.3f}.jsonl'
        command=[str(PYTHON),'vlm_experiment.py','--model','smol',
                 '--partition','dev_common_unmatched500','--condition','detailed512',
                 '--limit','20','--layers','5','18','--alpha',str(alpha),
                 '--output',str(OUT/name),'--resume',*options]
        print('RUN',mode,alpha,flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
print('PROBE_COMPLETE',flush=True)
