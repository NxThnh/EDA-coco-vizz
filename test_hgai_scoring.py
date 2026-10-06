"""CPU semantic checks for paired ratios, clustered POPE and output guards."""
import json
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from hgai_cpu_audit import ROOT,sha
from score_vlm_runs import bootstrap_statistics,bootstrap_pope,guards,verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit-only',action='store_true',help='CI without raw COCO assets')
    args=parser.parse_args()
    # CHAIR-I must weight mentions, not average per-image fractions (50%).
    samples=[{'CHAIR_S':1,'hallucinated':1,'generated':1,'tp':0,'predicted':1,'ground':1},
             {'CHAIR_S':0,'hallucinated':0,'generated':9,'tp':1,'predicted':1,'ground':1}]
    identical=bootstrap_statistics(samples,samples,1000)
    assert all(v['ci95_low']==v['ci95_high']==0 for v in identical.values())
    clean=[{**s,'CHAIR_S':0,'hallucinated':0} for s in samples]
    # Seed 1 draws [0, 1]: one hallucination / ten mentions = 10%, not 50%.
    assert bootstrap_statistics(samples,clean,1,seed=1)['CHAIR_I_pct']['delta_mean_bootstrap']==-10
    changed=bootstrap_statistics(samples,clean,1000)
    assert changed['CHAIR_S_pct']['ci95_low']==-100 and changed['CHAIR_S_pct']['ci95_high']==0
    assert changed['CHAIR_I_pct']['ci95_low']==-100 and changed['CHAIR_I_pct']['ci95_high']==0
    empty=[{**s,'generated':0,'hallucinated':0} for s in samples]
    assert bootstrap_statistics(samples,empty,1000)['CHAIR_I_pct']['undefined_replicates']==1000
    # Each image's balanced positive/negative questions stay in one cluster.
    rows=[{'image_id':image,'question_id':image*10+j,'label':label,'answer':'Yes.' if label=='yes' else 'No.'}
          for image in [1,2,3] for j,label in enumerate(['yes','yes','yes','no','no','no'])]
    inverted=[{**r,'answer':'No.' if r['label']=='yes' else 'Yes.'} for r in rows]
    delta,clusters=bootstrap_pope(rows,inverted,1000)
    assert clusters==3 and len(rows)==18
    for key in ['accuracy','precision','recall','F1']:
        assert delta[key]['ci95_low']==delta[key]['ci95_high']==-1
    assert delta['yes_rate']['ci95_low']==delta['yes_rate']['ci95_high']==0
    assert guards('I cannot tell from this image.')['refusal']
    assert not guards('No, there is no cat.')['refusal']
    assert guards('a cat on a mat '*8)['repeated_4gram_fraction']>.5
    assert guards('  ')['empty_output'] and guards('  ')['distinct_1'] is None
    # Real completed smoke proves the short-object-free output is not CHAIR-I=0.
    smoke=ROOT/'experiment_outputs/adapter_smoke/smol_alpha05_dev2.jsonl'
    real_smoke_checked=smoke.exists() and not args.unit_only
    if real_smoke_checked:
        config,actual=verify(smoke)
        metric=json.loads(smoke.with_suffix('.metrics.json').read_text())
        assert len(actual)==2 and metric['CHAIR_I_pct'] is None and metric['object_micro_F1']==0
    partial=ROOT/'experiment_outputs/full_baseline/smol_chair_eval500_nf4_detailed512.jsonl'
    if partial.exists():
        config=json.loads(partial.with_suffix('.config.json').read_text())
        if len(partial.read_text(encoding='utf-8').splitlines())<config['requested_rows']:
            try:verify(partial)
            except ValueError as error:assert 'Incomplete generation' in str(error)
            else:raise AssertionError('Partial benchmark was accepted')
    with TemporaryDirectory(prefix='hgai-unit-') as temporary:
        incomplete=Path(temporary)/'incomplete.jsonl'
        incomplete.write_text('{"image_id":1}\n',encoding='utf-8')
        incomplete.with_suffix('.config.json').write_text('{"requested_rows":2}',encoding='utf-8')
        try:verify(incomplete)
        except ValueError as error:assert 'Incomplete generation' in str(error)
        else:raise AssertionError('Partial benchmark was accepted')
    result={'status':'passed','paired_chair_ratio_and_undefined_denominator':True,
            'pope_image_clusters_and_known_deltas':True,'refusal_repetition_empty_guards':True,
            'partial_benchmark_rejected':True,'test_sha256':sha(Path(__file__)),
            'real_smoke_checked':real_smoke_checked,
            'scorer_sha256':sha(ROOT/'score_vlm_runs.py')}
    (ROOT/'eda_results/hgai/hgai_scoring_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
