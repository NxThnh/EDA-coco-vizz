"""Run output-only dev probes at frozen v2 bands and frozen alpha grids."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiment_outputs/v2_alpha_probe20'
OUT.mkdir(parents=True, exist_ok=True)
PYTHON = ROOT / '.venv-eda/Scripts/python.exe'
LOCKS = ROOT / 'eda_results/hgai'

for model, band, lock in (
    ('smol', (10, 23), 'smol_alpha_grid_lock.json'),
    ('tiny', (12, 28), 'tiny_alpha_grid_lock.json'),
):
    grid = json.loads((LOCKS / lock).read_text(encoding='utf-8'))['alpha_grid']
    for number, alpha in enumerate([0, *grid]):
        output = OUT / f'{model}_v2_dose_{number}.jsonl'
        command = [str(PYTHON), 'vlm_experiment.py', '--model', model,
                   '--partition', 'dev_common_unmatched500', '--condition', 'detailed512',
                   '--limit', '20', '--layers', str(band[0]), str(band[1]),
                   '--alpha', str(alpha), '--output', str(output), '--resume']
        print('DOSE', model, number, alpha, flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
