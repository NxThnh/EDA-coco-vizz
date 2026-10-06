"""Snapshot actual run progress; never turn a planned/partial run into a score."""
import json
import time
from datetime import datetime,timezone

import pandas as pd
from hgai_cpu_audit import ROOT,OUT,sha


def main():
    records=[]
    for config_path in sorted((ROOT/'experiment_outputs').glob('**/*.config.json')):
        config=json.loads(config_path.read_text(encoding='utf-8'))
        output=config_path.with_name(config_path.name.replace('.config.json','.jsonl'))
        rows=0
        if output.exists():
            for line in output.read_text(encoding='utf-8').splitlines():
                try:json.loads(line)
                except json.JSONDecodeError:break
                rows+=1
        metrics_path=output.with_suffix('.metrics.json')
        metric=json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else None
        scored=bool(metric and rows==config['requested_rows'] and metric.get('output_sha256')==sha(output))
        model_key=config.get('model_key') or ('smol' if 'SmolVLM' in config['model'] else 'tiny' if 'TinyLLaVA' in config['model'] else 'llava')
        records.append({'model':config['model'],'model_key':model_key,'task':config['task'],'partition':config['partition'],
            'condition':config.get('condition','detailed512' if config['task']=='chair' else 'pope64'),
            'alpha':config.get('alpha',0),'precision':config['precision'],'requested_rows':config['requested_rows'],
            'benchmark_rows':config['benchmark_rows'],'generated_rows':rows,
            'status':'scored' if scored else 'generation_complete_unscored' if rows==config['requested_rows'] else 'partial' if rows else 'configured',
            'role':config['role'],'CHAIR_S_pct':metric.get('CHAIR_S_pct') if scored else None,
            'CHAIR_I_pct':metric.get('CHAIR_I_pct') if scored else None,'accuracy':metric.get('accuracy') if scored else None,
            'output_path':output.relative_to(ROOT).as_posix()})
    pd.DataFrame(records).to_csv(OUT/'experiment_run_status.csv',index=False,encoding='utf-8-sig')
    campaign=ROOT/'experiment_outputs/hgai_campaign/campaign_status.json'
    status={'updated_unix':time.time(),'runs':records,
            'snapshot_utc':datetime.now(timezone.utc).isoformat(),
            'campaign':json.loads(campaign.read_text()) if campaign.exists() else None,
            'not_a_score':'Partial runs are progress only; completed pilots remain pilots.'}
    (OUT/'experiment_run_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print('Actual run statuses:',len(records))


if __name__=='__main__':main()
