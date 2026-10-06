"""Read current local campaign progress without starting another GPU job."""
import json
from hgai_cpu_audit import ROOT


def main():
    campaign=ROOT/'experiment_outputs/hgai_campaign/campaign_status.json'
    if campaign.exists():
        state=json.loads(campaign.read_text(encoding='utf-8'))
        print('Campaign:',state['status'],'| job:',state.get('active_job'),'| PID:',state.get('pid'))
        print('Completed jobs:',len(state.get('completed_jobs',[])))
        if state.get('error'):print('Error:',state['error'])
    for config_path in sorted((ROOT/'experiment_outputs').glob('**/*.config.json')):
        output=config_path.with_name(config_path.name.replace('.config.json','.jsonl'))
        config=json.loads(config_path.read_text(encoding='utf-8'))
        if config['requested_rows']<config['benchmark_rows']:continue
        count=0
        if output.exists():
            for line in output.read_text(encoding='utf-8').splitlines():
                try:json.loads(line)
                except json.JSONDecodeError:break
                count+=1
        print(output.name,':',count,'/',config['requested_rows'])
    for partial in sorted((ROOT/'model_assets').glob('*/*.safetensors.part')):
        print('Downloading:',partial.parent.name,partial.name,round(partial.stat().st_size/1024**3,3),'GiB')
    log=ROOT/'scratch/hgai_runtime/campaign.stdout.log'
    if log.exists():
        print('Recent worker log:')
        print('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-5:]))
    print('Counts are progress, not benchmark scores. A saved status may be stale if its PID has stopped.')


if __name__=='__main__':main()
