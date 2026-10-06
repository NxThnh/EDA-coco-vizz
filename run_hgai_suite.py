"""Serial full-benchmark campaign: validated adapters, dev tuning, locked eval, CI.

Runs are resumable and flushed per image. Selection uses only dev500. A GPU job
is never labelled complete until all requested outputs and scoring exist.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
os.environ.setdefault('HF_HUB_OFFLINE','1')

from hgai_cpu_audit import ROOT,OUT,PREP,sha
from score_vlm_runs import score,compare

CAMPAIGN=ROOT/'experiment_outputs/hgai_campaign_v2'
STATE=CAMPAIGN/'campaign_status.json'
PYTHON=ROOT/'.venv-eda/Scripts/python.exe'
DESIGN_LOCK=ROOT/'eda_results/hgai/hgai_design_lock.json'
GUARDS={'max_recall_drop':.02,'max_F1_drop':.02,'max_refusal_increase':.05,'max_empty_increase':.05,
        'max_truncation_increase':.05,'max_repeat_increase':.05,'minimum_length_ratio':.5}


def atomic(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=['smol','tiny','llava'],default=['smol','tiny','llava'])
    parser.add_argument('--external-smol-baseline',type=Path)
    parser.add_argument('--bootstrap',type=int,default=10000)
    args=parser.parse_args()
    if not DESIGN_LOCK.exists():
        raise RuntimeError('Campaign paused: review the 10–20 image alpha dose probe and per-layer VAR/image-masking curves, then write hgai_design_lock.json before starting campaign v2.')
    design=json.loads(DESIGN_LOCK.read_text(encoding='utf-8'))
    alpha_grids=design['alpha_grid_by_model']
    for model in args.models:
        if model=='llava':continue
        alphas=alpha_grids.get(model,[])
        if not alphas or any(not 0<alpha<=.3 for alpha in alphas):
            raise ValueError('Each small model needs its own locked low-alpha grid')
    for model in args.models:
        if model=='llava':continue
        probe=design['dose_probe_by_model'][model]
        if sha(ROOT/probe['path'])!=probe['sha256']:
            raise ValueError('Dose probe changed after design lock: '+model)
    if sha(ROOT/design['layer_evidence_path'])!=design['layer_evidence_sha256']:
        raise ValueError('Layer evidence changed after design lock')
    if any(model not in design['layer_ranges_by_model'] for model in args.models if model!='llava'):
        raise ValueError('Missing measured layer ranges for a small model')
    CAMPAIGN.mkdir(parents=True,exist_ok=True)
    plan={'models':args.models,'precision':'nf4','dev_images':500,'eval_images':500,'pope_questions_per_variant':3000,
          'design_lock_sha256':sha(DESIGN_LOCK),'layer_ranges_by_model':design['layer_ranges_by_model'],
          'alpha_grid_by_model':alpha_grids,'guards':GUARDS,'bootstrap':args.bootstrap,
          'selection':'minimum dev CHAIR_S, then CHAIR_I, then maximum F1, then lower alpha; baseline included',
          'dev_manifest_sha256':sha(PREP/'dev_common_unmatched500.jsonl'),
          'eval_manifest_sha256':sha(PREP/'eval_common_unmatched500.jsonl'),
          'source_sha256':{f:sha(ROOT/f) for f in ['hgai_attention.py','vlm_experiment.py','tinyllava_model.py','score_vlm_runs.py','run_hgai_suite.py']},
          'external_smol_baseline':str(args.external_smol_baseline) if args.external_smol_baseline else None,
          'note':'NF4 campaign; it does not reproduce native FP16 paper numbers. Long-running local GPU work.'}
    plan_path=CAMPAIGN/'campaign_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Campaign plan/source changed; create a versioned campaign instead of silently resuming')
    if not plan_path.exists():atomic(plan_path,plan)
    state={'status':'running','pid':os.getpid(),'plan_sha256':sha(plan_path),'completed_jobs':[],
           'active_job':None,'started_unix':time.time(),'updated_unix':time.time()}
    if STATE.exists():
        previous=json.loads(STATE.read_text())
        state['completed_jobs']=previous.get('completed_jobs',[])
    def update(status,job=None,error=None):
        state.update(status=status,active_job=job,updated_unix=time.time())
        if error:state['error']=error
        atomic(STATE,state)
    def execute(command,label):
        for name,digest in plan['source_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Campaign code changed while running: '+name)
        update('running',label)
        print('JOB',label,flush=True)
        subprocess.run([str(PYTHON),*command],cwd=ROOT,check=True)
    def run(model,partition,alpha=0,layers=None,condition='detailed512',task='chair',variant='random',split_off=False,image_mode=None,smol_longest_edge=None):
        depth=24 if model=='smol' else 32
        layers=layers or [5,18]
        tag=f'{model}_{partition if task=="chair" else "pope_"+variant}_{condition}_split{not split_off}_a{alpha}_l{layers[0]}-{layers[1]}'
        if image_mode:tag+='_'+image_mode
        if smol_longest_edge:tag+=f'_edge{smol_longest_edge}'
        path=CAMPAIGN/(tag+'.jsonl')
        command=['vlm_experiment.py','--model',model,'--task',task,'--partition',partition,
                 '--condition',condition,'--alpha',str(alpha),'--layers',*[str(i) for i in layers],
                 '--limit','500' if task=='chair' else '3000','--output',str(path),'--resume']
        if task=='pope':command.extend(['--pope-variant',variant])
        if split_off:command.append('--split-off')
        if image_mode:command.extend(['--image-mode',image_mode])
        if smol_longest_edge:command.extend(['--smol-longest-edge',str(smol_longest_edge)])
        execute(command,tag)
        metric=score(path)
        if tag not in state['completed_jobs']:state['completed_jobs'].append(tag)
        update('running',None)
        return path,metric
    try:
        execute(['test_hgai_attention.py'],'adapter_semantic_validation')
        execute(['test_hgai_scoring.py'],'scorer_semantic_validation')
        if args.external_smol_baseline:
            if not args.external_smol_baseline.with_suffix('.run.json').exists():
                execute(['smol_baseline.py','--partition','eval_common_unmatched500','--limit','500',
                         '--output',str(args.external_smol_baseline),'--resume'],'resume_existing_smol_eval500')
            metric=score(args.external_smol_baseline)
            if metric['images']!=500:raise ValueError('External baseline is not full500')
        for model in args.models:
            while not (OUT/(model+'_weights_provenance.json')).exists():
                update('waiting_for_weights',model)
                print('Waiting for verified weights:',model,flush=True)
                time.sleep(30)
            if model=='llava':
                reference,_=run('llava','eval_random500')
                treated,_=run('llava','eval_random500',.5,[5,18])
                compare(reference,treated,CAMPAIGN/'llava_paper_protocol_nf4_ci.json',args.bootstrap)
                continue
            baseline_dev,baseline_metrics=run(model,'dev_common_unmatched500')
            candidates=[{'path':str(baseline_dev),'alpha':0,'layers':[5,18],'metrics':baseline_metrics,'guard_pass':True}]
            depth=24 if model=='smol' else 32
            for layers in design['layer_ranges_by_model'][model]:
                if len(layers)!=2 or not 0<=layers[0]<=layers[1]<depth:
                    raise ValueError('Invalid locked layer range')
                for alpha in alpha_grids[model]:
                    path,metrics=run(model,'dev_common_unmatched500',alpha,layers)
                    passed=(metrics['object_micro_recall']>=baseline_metrics['object_micro_recall']-GUARDS['max_recall_drop']
                        and metrics['object_micro_F1']>=baseline_metrics['object_micro_F1']-GUARDS['max_F1_drop']
                        and metrics['refusal_rate']<=baseline_metrics['refusal_rate']+GUARDS['max_refusal_increase']
                        and metrics['empty_output_rate']<=baseline_metrics['empty_output_rate']+GUARDS['max_empty_increase']
                        and metrics['truncation_rate']<=baseline_metrics['truncation_rate']+GUARDS['max_truncation_increase']
                        and metrics['mean_repeated_4gram_fraction']<=baseline_metrics['mean_repeated_4gram_fraction']+GUARDS['max_repeat_increase']
                        and metrics['mean_new_tokens']>=baseline_metrics['mean_new_tokens']*GUARDS['minimum_length_ratio'])
                    candidates.append({'path':str(path),'alpha':alpha,'layers':layers,'metrics':metrics,'guard_pass':bool(passed)})
            eligible=[r for r in candidates if r['guard_pass']]
            chosen=min(eligible,key=lambda r:(r['metrics']['CHAIR_S_pct'],r['metrics']['CHAIR_I_pct'] if r['metrics']['CHAIR_I_pct'] is not None else float('inf'),-r['metrics']['object_micro_F1'],r['alpha']))
            frozen={'model':model,'partition':'dev_common_unmatched500','images':500,'alpha':chosen['alpha'],
                    'layers_inclusive':chosen['layers'],'selection_plan_sha256':sha(plan_path),
                    'dev_manifest_sha256':plan['dev_manifest_sha256'],
                    'candidate_output_sha256':{r['path']:sha(Path(r['path'])) for r in candidates},'guards':GUARDS,
                    'selected_metrics':chosen['metrics'],'candidates':candidates,
                    'note':'Selected using complete dev500 only, before intervention eval/POPE. Alpha zero is a valid outcome.'}
            frozen_path=CAMPAIGN/(model+'_frozen_selection.json')
            if frozen_path.exists() and json.loads(frozen_path.read_text())!=frozen:raise ValueError('Frozen selection changed')
            if not frozen_path.exists():atomic(frozen_path,frozen)
            if model=='smol' and args.external_smol_baseline:baseline_eval=args.external_smol_baseline
            else:baseline_eval,_=run(model,'eval_common_unmatched500')
            intervention,_=run(model,'eval_common_unmatched500',chosen['alpha'],chosen['layers'])
            compare(baseline_eval,intervention,CAMPAIGN/(model+'_paired_eval_ci.json'),args.bootstrap)
            for variant in ['random','popular','adversarial']:
                base,_=run(model,'eval_common_unmatched500',task='pope',variant=variant)
                treated,_=run(model,'eval_common_unmatched500',chosen['alpha'],chosen['layers'],task='pope',variant=variant)
                compare(base,treated,CAMPAIGN/(model+'_pope_'+variant+'_paired_ci.json'),args.bootstrap)
            if model in {'smol','tiny'}:
                for condition in ['detailed64','concise64']:
                    base,_=run(model,'eval_common_unmatched500',condition=condition)
                    treated,_=run(model,'eval_common_unmatched500',chosen['alpha'],chosen['layers'],condition=condition)
                    compare(base,treated,CAMPAIGN/(model+'_'+condition+'_paired_ci.json'),args.bootstrap)
            if model=='smol':
                base,_=run(model,'eval_common_unmatched500',smol_longest_edge=768)
                treated,_=run(model,'eval_common_unmatched500',chosen['alpha'],chosen['layers'],smol_longest_edge=768)
                compare(base,treated,CAMPAIGN/'smol_edge768_paired_ci.json',args.bootstrap)
                base,_=run(model,'eval_common_unmatched500',split_off=True)
                treated,_=run(model,'eval_common_unmatched500',chosen['alpha'],chosen['layers'],split_off=True)
                compare(base,treated,CAMPAIGN/'smol_split_off_paired_ci.json',args.bootstrap)
        update('complete')
    except BaseException as error:
        update('failed',state.get('active_job'),str(error))
        raise


if __name__=='__main__':
    from filelock import FileLock
    CAMPAIGN.mkdir(parents=True,exist_ok=True)
    with FileLock(str(CAMPAIGN/'campaign.lock'),timeout=0):
        main()
