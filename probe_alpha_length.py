"""Short SmolVLM2 dose probe: output length only, before any CHAIR tuning."""
import json
import subprocess
from pathlib import Path
from statistics import mean

from hgai_cpu_audit import ROOT, sha

OUT = ROOT / 'experiment_outputs' / 'alpha_dose_probe'
PYTHON = ROOT / '.venv-eda' / 'Scripts' / 'python.exe'
ALPHAS = (0.02, 0.05, 0.10, 0.20, 0.30)
N = 10


def lengths(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if len(rows) < N:
        raise ValueError(f'Incomplete output: {path} ({len(rows)}/{N})')
    return [int(row['new_tokens']) for row in rows[:N]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = ROOT / 'experiment_outputs/hgai_campaign/smol_dev_common_unmatched500_detailed512_splitTrue_a0_l5-18.jsonl'
    base = lengths(baseline)
    summary = [{'alpha': 0, 'n': N, 'mean_new_tokens': mean(base),
                'min_new_tokens': min(base), 'max_new_tokens': max(base),
                'ratio_to_baseline': 1, 'output_sha256': sha(baseline)}]
    for alpha in ALPHAS:
        output = OUT / f'smol_dev10_alpha_{alpha:.2f}.jsonl'
        subprocess.run([str(PYTHON), 'vlm_experiment.py', '--model', 'smol',
                        '--partition', 'dev_common_unmatched500', '--limit', str(N),
                        '--alpha', str(alpha), '--layers', '5', '18',
                        '--output', str(output), '--resume'], cwd=ROOT, check=True)
        values = lengths(output)
        summary.append({'alpha': alpha, 'n': N, 'mean_new_tokens': mean(values),
                        'min_new_tokens': min(values), 'max_new_tokens': max(values),
                        'ratio_to_baseline': mean(values) / mean(base),
                        'images_below_half_baseline':sum(value < baseline_value/2 for value,baseline_value in zip(values,base)),
                        'output_sha256': sha(output)})
        (OUT / 'length_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
