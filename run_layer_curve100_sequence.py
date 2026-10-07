"""Queue the two 100-image dev curve measurements after the Tiny dose probe."""
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PYTHON=ROOT/'.venv-eda/Scripts/python.exe'
READY=ROOT/'experiment_outputs/tiny_alpha_probe20_v2/tiny_pad_alpha_0.500.run.json'
while not READY.exists():
    time.sleep(15)
for model,command in (('smol',['measure_layer_curves100.py','--model','smol']),
                      ('tiny',['measure_layer_curves100_tiny_sdpa.py','--limit','100'])):
    print('MEASURE',model,flush=True)
    subprocess.run([str(PYTHON),*command],cwd=ROOT,check=True)
    subprocess.run([str(PYTHON),'analyze_layer_rule_v2.py'],cwd=ROOT,check=True)
print('LAYER_CURVES100_COMPLETE',flush=True)
