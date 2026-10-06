"""Shared scoring with operational refusal/repetition guards and paired image CI."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from hgai_cpu_audit import ROOT,PREP,OUT,sha
from score_hgai_baseline import score_chair,score_pope

REFUSAL=re.compile(r"\b(?:i|we)\s+(?:cannot|can't|can not|am unable to|are unable to|am not able to)\s+(?:tell|identify|determine|describe|see|discern)|\b(?:unable to identify|image is too (?:unclear|blurry|dark))\b",re.I)


def guards(text):
    words=re.findall(r"[a-z0-9]+(?:'[a-z]+)?",text.casefold())
    grams=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
    return {'refusal':bool(REFUSAL.search(text)),'lexical_tokens':len(words),
            'distinct_1':len(set(words))/len(words) if words else None,
            'repeated_4gram_fraction':1-len(set(grams))/len(grams) if grams else 0,
            'empty_output':not text.strip()}


def verify(path):
    config=json.loads(path.with_suffix('.config.json').read_text(encoding='utf-8'))
    rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    if not rows or len(rows)!=config['requested_rows']:
        raise ValueError('Incomplete generation; publish progress, not completed benchmark metrics')
    if config['task']=='chair':
        from lock_common_unmatched import prepare_common
        prepare_common()
        manifest=PREP/(config['partition']+'.jsonl')
        original=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
        if [r['image_id'] for r in rows]!=[r['image_id'] for r in original[:len(rows)]]:
            raise ValueError('Outputs differ from locked image prefix')
    else:
        variant=config['partition'].removeprefix('official-pope-')
        if variant not in {'random','popular','adversarial'}:raise ValueError('Unknown POPE variant')
        manifest=ROOT/'eda_resources/pope_official'/('coco_pope_'+variant+'.json')
        original=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
        for row,source in zip(rows,original):
            if row['question_id']!=source['question_id'] or row['label']!=source['label'] or row['prompt']!=source['text'] or row['image_id']!=int(Path(source['image']).stem.split('_')[-1]):
                raise ValueError('POPE question/label/image changed')
    if sha(manifest)!=config['manifest_sha256']:raise ValueError('Manifest hash changed')
    for name,digest in config.get('annotation_sha256',{}).items():
        if sha(ROOT/'coco/annotations'/name)!=digest:raise ValueError('Ground truth changed after generation')
    return config,rows


def score(path):
    config,rows=verify(path)
    metrics,details=score_chair(rows) if config['task']=='chair' else score_pope(rows)
    guard_rows=[{'image_id':r['image_id'],**({'question_id':r['question_id']} if 'question_id' in r else {}),
                 **guards(r.get('caption',r.get('answer','')))} for r in rows]
    values=[r['distinct_1'] for r in guard_rows if r['distinct_1'] is not None]
    metrics.update({'role':config['role'],'model':config['model'],'task':config['task'],
        'partition':config['partition'],'model_precision':config['precision'],'max_new_tokens':config['max_new_tokens'],
        'image_splitting':config['image_splitting'],'alpha':config.get('alpha',0),
        'condition':config.get('condition','detailed512' if config['task']=='chair' else 'pope64'),
        'refusal_rate':sum(r['refusal'] for r in guard_rows)/len(rows),
        'empty_output_rate':sum(r['empty_output'] for r in guard_rows)/len(rows),
        'mean_distinct_1':float(np.mean(values)) if values else None,
        'mean_repeated_4gram_fraction':float(np.mean([r['repeated_4gram_fraction'] for r in guard_rows])),
        'guard_definition':'English lexical refusal regex; macro per-response Distinct-1 and repeated 4-gram fraction; not gold refusal labels.',
        'output_sha256':sha(path),'config_sha256':sha(path.with_suffix('.config.json')),
        'scorer_sha256':sha(Path(__file__)),'chair_wrapper_sha256':sha(ROOT/'score_hgai_baseline.py'),
        'metric_sources':json.loads((OUT/'metrics_provenance.json').read_text())})
    if details is not None:
        for detail in details:
            generated=set(detail['mscoco_generated_words']);ground=set(detail['mscoco_gt_words'])
            detail['paired_statistics']={'CHAIR_S':detail['metrics']['CHAIRs'],
                'hallucinated':len(detail['mscoco_hallucinated_words']),'generated':len(detail['mscoco_generated_words']),
                'tp':len(generated & ground),'predicted':len(generated),'ground':len(ground)}
        path.with_suffix('.chair_details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
    path.with_suffix('.guard_details.json').write_text(json.dumps(guard_rows,indent=2),encoding='utf-8')
    path.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    return metrics


def bootstrap_statistics(a,b,reps=10000,seed=927):
    if len(a)!=len(b) or not a:raise ValueError('Nonempty equal paired sample required')
    names=['CHAIR_S','hallucinated','generated','tp','predicted','ground']
    aa=np.asarray([[r[k] for k in names] for r in a],dtype=float)
    bb=np.asarray([[r[k] for k in names] for r in b],dtype=float)
    indices=np.random.default_rng(seed).integers(0,len(a),size=(reps,len(a)))
    def evaluate(x):
        summed=x[indices].sum(axis=1)
        with np.errstate(divide='ignore',invalid='ignore'):
            return {'CHAIR_S_pct':100*summed[:,0]/len(a),
                    'CHAIR_I_pct':100*summed[:,1]/summed[:,2],
                    'object_F1':2*summed[:,3]/(summed[:,4]+summed[:,5])}
    av,bv=evaluate(aa),evaluate(bb)
    result={}
    for key in av:
        delta=bv[key]-av[key];valid=delta[np.isfinite(delta)]
        result[key]={'delta_mean_bootstrap':float(valid.mean()) if len(valid) else None,
                     'ci95_low':float(np.percentile(valid,2.5)) if len(valid) else None,
                     'ci95_high':float(np.percentile(valid,97.5)) if len(valid) else None,
                     'valid_replicates':int(len(valid)),'undefined_replicates':int(reps-len(valid))}
    return result


def bootstrap_pope(rows_a,rows_b,reps=10000,seed=927):
    """Resample images, keeping their positive/negative questions together."""
    from score_hgai_baseline import parse_pope
    image_ids=sorted({r['image_id'] for r in rows_a})
    if not image_ids or len(rows_a)!=len(rows_b):raise ValueError('Nonempty paired POPE required')
    lookup={image:i for i,image in enumerate(image_ids)}
    def counts(rows):
        result=np.zeros((len(image_ids),4),dtype=float)
        for r in rows:
            gold=int(r['label']=='yes');prediction=parse_pope(r['answer'])
            slot=0 if gold and prediction else 1 if not gold and not prediction else 2 if prediction else 3
            result[lookup[r['image_id']],slot]+=1
        return result
    indices=np.random.default_rng(seed).integers(0,len(image_ids),size=(reps,len(image_ids)))
    def evaluate(rows):
        tp,tn,fp,fn=counts(rows)[indices].sum(axis=1).T
        with np.errstate(divide='ignore',invalid='ignore'):
            return {'accuracy':(tp+tn)/(tp+tn+fp+fn),'precision':tp/(tp+fp),
                    'recall':tp/(tp+fn),'F1':2*tp/(2*tp+fp+fn),'yes_rate':(tp+fp)/(tp+tn+fp+fn)}
    aa,bb=evaluate(rows_a),evaluate(rows_b)
    result={}
    for key in aa:
        delta=bb[key]-aa[key];valid=delta[np.isfinite(delta)]
        result[key]={'delta_mean_bootstrap':float(valid.mean()) if len(valid) else None,
                    'ci95_low':float(np.percentile(valid,2.5)) if len(valid) else None,
                    'ci95_high':float(np.percentile(valid,97.5)) if len(valid) else None,
                    'valid_replicates':int(len(valid)),'undefined_replicates':int(reps-len(valid))}
    return result,len(image_ids)


def compare(baseline,intervention,output,reps):
    ca,ra=verify(baseline);cb,rb=verify(intervention)
    if ca['task']!=cb['task']:raise ValueError('Comparison tasks differ')
    for key in ['model','revision','manifest_sha256','precision','image_splitting','max_new_tokens','seed','weights']:
        if ca[key]!=cb[key]:raise ValueError('Unmatched comparison setting: '+key)
    if [r['image_id'] for r in ra]!=[r['image_id'] for r in rb] or [r['prompt'] for r in ra]!=[r['prompt'] for r in rb]:
        raise ValueError('Paired image order or prompt differs')
    if ca.get('image_mode','hgai_crop')!=cb.get('image_mode','hgai_crop'):raise ValueError('Image mode differs')
    ma,mb=score(baseline),score(intervention)
    if ca['task']=='chair':
        a=json.loads(baseline.with_suffix('.chair_details.json').read_text(encoding='utf-8'))
        b=json.loads(intervention.with_suffix('.chair_details.json').read_text(encoding='utf-8'))
        intervals=bootstrap_statistics([s['paired_statistics'] for s in a],[s['paired_statistics'] for s in b],reps)
        keys=['CHAIR_S_pct','CHAIR_I_pct','object_micro_F1'];images=len(ra)
    else:
        if [(r['question_id'],r['label']) for r in ra]!=[(r['question_id'],r['label']) for r in rb]:
            raise ValueError('POPE question/label pairs differ')
        intervals,images=bootstrap_pope(ra,rb,reps)
        keys=['accuracy','precision','recall','F1','yes_rate']
    result={'role':'paired_eval' if len(ra)==ca['benchmark_rows'] else 'paired_pilot',
            'task':ca['task'],'images':images,'rows':len(ra),'bootstrap_unit':'image',
            'bootstrap_replicates':reps,'seed':927,'delta_definition':'intervention minus baseline',
            'baseline_sha256':sha(baseline),'intervention_sha256':sha(intervention),'code_sha256':sha(Path(__file__)),
            'delta_point':{k:mb[k]-ma[k] if mb[k] is not None and ma[k] is not None else None for k in keys},
            'paired_ci':intervals}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path,nargs='?')
    parser.add_argument('--compare',type=Path,nargs=2)
    parser.add_argument('--ci-output',type=Path)
    parser.add_argument('--bootstrap',type=int,default=10000)
    args=parser.parse_args()
    if args.compare:
        if not args.ci_output or args.bootstrap<100:parser.error('Provide --ci-output and at least 100 bootstrap replicates')
        result=compare(*args.compare,args.ci_output,args.bootstrap)
    elif args.output:result=score(args.output)
    else:parser.error('Provide an output JSONL or --compare BASELINE HGAI')
    print(json.dumps(result,indent=2))
