"""Refresh the report once the background campaign completes or reports failure."""
import json
import subprocess
import time
from pathlib import Path
from filelock import FileLock
from hgai_cpu_audit import ROOT


def main():
    state_path=ROOT/'experiment_outputs/hgai_campaign/campaign_status.json'
    runtime=ROOT/'scratch/hgai_runtime'
    runtime.mkdir(parents=True,exist_ok=True)
    with FileLock(str(runtime/'report_monitor.lock'),timeout=0):
        while True:
            state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
            if state.get('status') in {'complete','failed'}:
                print('Campaign final status:',state['status'],flush=True)
                subprocess.run([str(ROOT/'.venv-eda/Scripts/python.exe'),'rebuild_hgai.py'],cwd=ROOT,check=True)
                print('Report refreshed; complete scored runs remain local until explicitly staged for Git.',flush=True)
                return
            time.sleep(30)


if __name__=='__main__':main()
