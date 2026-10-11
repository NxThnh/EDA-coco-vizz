"""Autonomous runner for TinyLLaVA-3.1B experimental campaign.

Executes sequential experimental stages:
1. TinyLLaVA Dev Baseline (alpha=0, 500 images) with --resume
2. Scoring TinyLLaVA Dev Baseline
3. TinyLLaVA Dev HGAI (alpha=0.045, layers 5-18, 500 images) with --resume
4. Scoring TinyLLaVA Dev HGAI
5. Paired Bootstrap CI (Baseline vs HGAI, 10,000 replicates)
6. Update experiment status, rebuild reports, and run integrity validation
"""
import datetime
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv-eda/Scripts/python.exe'
LOG_PATH = ROOT / 'experiment_outputs/hgai_campaign/tiny_campaign.log'


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
    log('TinyLLaVA HGAI Campaign started.')

    # Stage 1: TinyLLaVA Dev Baseline
    base_out = 'experiment_outputs/hgai_campaign/tiny_dev_common_unmatched500_detailed512_splitTrue_a0_l5-18.jsonl'
    cmd_base = [
        str(PYTHON), 'vlm_experiment.py',
        '--model', 'tiny',
        '--task', 'chair',
        '--partition', 'dev_common_unmatched500',
        '--condition', 'detailed512',
        '--alpha', '0',
        '--layers', '5', '18',
        '--limit', '500',
        '--output', base_out,
        '--resume'
    ]
    run_stage(cmd_base, '1. TinyLLaVA Dev Baseline (500 images)')

    # Stage 2: Score TinyLLaVA Dev Baseline
    cmd_score_base = [str(PYTHON), 'score_vlm_runs.py', base_out]
    run_stage(cmd_score_base, '2. Score TinyLLaVA Dev Baseline')

    # Stage 3: TinyLLaVA Dev HGAI (alpha=0.045, layers 5-18)
    hgai_out = 'experiment_outputs/hgai_campaign/tiny_dev_common_unmatched500_detailed512_splitTrue_a0.045_l5-18.jsonl'
    cmd_hgai = [
        str(PYTHON), 'vlm_experiment.py',
        '--model', 'tiny',
        '--task', 'chair',
        '--partition', 'dev_common_unmatched500',
        '--condition', 'detailed512',
        '--alpha', '0.045',
        '--layers', '5', '18',
        '--limit', '500',
        '--output', hgai_out,
        '--resume'
    ]
    run_stage(cmd_hgai, '3. TinyLLaVA Dev HGAI (500 images)')

    # Stage 4: Score TinyLLaVA Dev HGAI
    cmd_score_hgai = [str(PYTHON), 'score_vlm_runs.py', hgai_out]
    run_stage(cmd_score_hgai, '4. Score TinyLLaVA Dev HGAI')

    # Stage 5: Paired Bootstrap CI
    ci_out = 'experiment_outputs/hgai_campaign/paired_tiny_dev500_a0045_ci.json'
    cmd_compare = [
        str(PYTHON), 'score_vlm_runs.py',
        '--compare', base_out, hgai_out,
        '--ci-output', ci_out
    ]
    run_stage(cmd_compare, '5. Paired Bootstrap CI (Tiny Baseline vs HGAI)')

    # Stage 6: Update status & validate
    cmd_status = [str(PYTHON), 'build_experiment_status.py']
    run_stage(cmd_status, '6a. Update Experiment Status')

    cmd_report = [str(PYTHON), 'build_hgai_report_final.py']
    run_stage(cmd_report, '6b. Build Final Report')

    cmd_nb = [str(PYTHON), 'build_hgai_notebook.py']
    run_stage(cmd_nb, '6c. Build Notebook')

    cmd_publish = [str(PYTHON), 'publish_hgai_report.py']
    run_stage(cmd_publish, '6d. Publish Reports & HTML')

    cmd_val = [str(PYTHON), 'validate_hgai.py']
    run_stage(cmd_val, '6e. Validate Integrity')

    log('ALL TINYLLAVA PIPELINE STAGES COMPLETED SUCCESSFULLY!')


if __name__ == '__main__':
    main()
