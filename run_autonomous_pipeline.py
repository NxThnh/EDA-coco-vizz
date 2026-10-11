"""Autonomous pipeline runner for HGAI campaign execution.

Executes sequential experimental stages without requiring user manual intervention:
1. SmolVLM2 HGAI on eval_common_unmatched500 (500 images) with --resume
2. Scoring with score_vlm_runs.py (CHAIR-S, CHAIR-I, Recall, F1, Guard checks)
3. Paired Bootstrap CI comparison (10,000 replicates) vs full_baseline
4. Update experiment run status
5. Rebuild and publish final reports and notebooks
6. Full audit validation check
"""
import datetime
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv-eda/Scripts/python.exe'
LOG_PATH = ROOT / 'experiment_outputs/hgai_campaign/autonomous_pipeline.log'


def log(msg: str):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f'[{timestamp}] {msg}'
    print(formatted, flush=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(formatted + '\n')


def run_stage(cmd: list, label: str):
    log(f'=== STARTING STAGE: {label} ===')
    log(f'Command: {" ".join(cmd)}')
    start_t = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )
    for line in iter(proc.stdout.readline, ''):
        stripped = line.rstrip()
        if stripped:
            print(f'[{label}] {stripped}', flush=True)
            with LOG_PATH.open('a', encoding='utf-8') as f:
                f.write(f'[{label}] {stripped}\n')
    proc.stdout.close()
    ret = proc.wait()
    elapsed = time.perf_counter() - start_t
    if ret != 0:
        log(f'!!! STAGE FAILED ({label}) with code {ret} after {elapsed:.1f}s !!!')
        raise RuntimeError(f'Stage failed: {label} with returncode {ret}')
    log(f'=== STAGE COMPLETED: {label} in {elapsed:.1f}s ===\n')


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log('Autonomous HGAI Pipeline started.')

    # Stage 1: SmolVLM2 HGAI on eval_common_unmatched500
    eval_out = 'experiment_outputs/hgai_campaign/smol_eval_common_unmatched500_detailed512_splitTrue_a0.035_l10-23.jsonl'
    cmd_eval = [
        str(PYTHON), 'vlm_experiment.py',
        '--model', 'smol',
        '--task', 'chair',
        '--partition', 'eval_common_unmatched500',
        '--condition', 'detailed512',
        '--alpha', '0.035',
        '--layers', '10', '23',
        '--limit', '500',
        '--output', eval_out,
        '--resume'
    ]
    run_stage(cmd_eval, '1. SmolVLM2 Eval HGAI Generation (500 images)')

    # Stage 2: Score Eval HGAI run
    cmd_score = [
        str(PYTHON), 'score_vlm_runs.py',
        eval_out
    ]
    run_stage(cmd_score, '2. Score SmolVLM2 Eval HGAI')

    # Stage 3: Paired Bootstrap CI vs Eval Baseline
    eval_baseline = 'experiment_outputs/full_baseline/smol_chair_eval500_nf4_detailed512.jsonl'
    eval_ci_out = 'experiment_outputs/hgai_campaign/paired_smol_eval500_a0035_ci.json'
    cmd_compare = [
        str(PYTHON), 'score_vlm_runs.py',
        '--compare', eval_baseline, eval_out,
        '--ci-output', eval_ci_out
    ]
    run_stage(cmd_compare, '3. Paired Bootstrap CI (Baseline Eval vs HGAI Eval)')

    # Stage 4: Update experiment status snapshot
    cmd_status = [str(PYTHON), 'build_experiment_status.py']
    run_stage(cmd_status, '4. Update Experiment Status')

    # Stage 5: Rebuild reports & notebooks
    cmd_report = [str(PYTHON), 'build_hgai_report_final.py']
    run_stage(cmd_report, '5a. Build Final HGAI Report')

    cmd_nb = [str(PYTHON), 'build_hgai_notebook.py']
    run_stage(cmd_nb, '5b. Build HGAI Notebook')

    cmd_publish = [str(PYTHON), 'publish_hgai_report.py']
    run_stage(cmd_publish, '5c. Publish Reports & HTML')

    # Stage 6: Audit Validation
    cmd_val = [str(PYTHON), 'validate_hgai.py']
    run_stage(cmd_val, '6. Validate HGAI Repository Integrity')

    log('ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!')


if __name__ == '__main__':
    main()
