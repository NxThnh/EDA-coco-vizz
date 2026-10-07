"""Refresh and verify reports from existing CPU outputs; fail fast on any step."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute-cpu',action='store_true',help='Run all 2,500 images through CPU processors again')
    parser.add_argument('--refresh-training-overlap',action='store_true',help='Rescan the downloaded annotation JSONs and final OneVision IDs')
    args=parser.parse_args()
    cpu=ROOT/'.venv-eda/Scripts/python.exe'
    if not cpu.is_file():raise FileNotFoundError('Create the workspace CPU venv and install requirements-hgai-cpu.txt first')
    def run(python,script,*arguments):
        subprocess.run([str(python),script,*arguments],cwd=ROOT,check=True)
    run(cpu,'lock_common_unmatched.py')
    if args.recompute_cpu:
        run(cpu,'hgai_cpu_audit.py')
        run(cpu,'verify_tiny_processor.py')
        run(cpu,'audit_smol_768.py')
    else:
        from hgai_cpu_audit import sha
        environment=json.loads((ROOT/'eda_results/hgai/processor_environment.json').read_text(encoding='utf-8'))
        if environment.get('geometry_postprocessing',{}).get('code_sha256') != sha(ROOT/'hgai_geometry_finalize.py'):
            run(cpu,'hgai_geometry_finalize.py')
    if args.refresh_training_overlap:run(cpu,'small_vlm_training_overlap.py')
    run(cpu,'sampling_fix_diagnostics.py')
    run(cpu,'build_experiment_status.py')
    run(cpu,'audit_source_availability.py')
    if not (ROOT/'eda_results/hgai/smol_tokens_768_per_image.csv').exists():
        run(cpu,'audit_smol_768.py')
    run(cpu,'audit_report_additions.py')
    if not (ROOT/'eda_results/hgai/layer_rule_lock.json').exists():
        raise FileNotFoundError('Missing layer rule lock')
    run(cpu,'build_hgai_report_final.py')
    run(cpu,'build_eda_notebook.py')
    run(cpu,'run_eda_notebook.py')
    run(cpu,'verify_hgai_report.py')
    run(cpu,'export_hgai_html.py')
    run(cpu,'validate_hgai.py')


if __name__=='__main__':main()
