"""Queue descriptive all-step Smol VAR after both 100-image curve runs."""
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
READY=ROOT/'eda_results/hgai/layer_curves100_tiny_summary.csv'
while not READY.exists():
    time.sleep(15)
python=ROOT/'.venv-eda/Scripts/python.exe'
subprocess.run([str(python),'measure_var_all_steps_smol20.py'],cwd=ROOT,check=True)
print('ALL_STEP_VAR_COMPLETE',flush=True)
