"""Summarize TinyLLaVA dev-only dose outputs and lock the prescribed log grid."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_alpha_probe20 import distance

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'experiment_outputs/tiny_alpha_probe20_v2'
OUT=ROOT/'eda_results/hgai'
DOSES=(.02,.05,.10,.20,.35,.50)


def records(alpha):
    path=SOURCE/f'tiny_pad_alpha_{alpha:.3f}.jsonl'
    run=json.loads(path.with_suffix('.run.json').read_text(encoding='utf-8'))
    cfg=json.loads(path.with_suffix('.config.json').read_text(encoding='utf-8'))
    assert run['status']=='complete' and run['rows']==20
    assert hashlib.sha256(path.read_bytes()).hexdigest()==run['output_sha256']
    assert cfg['model_key']=='tiny' and cfg['requested_rows']==20
    assert cfg['condition']=='detailed512' and cfg['layers_inclusive']==[5,18]
    assert cfg['alpha']==alpha and cfg['max_new_tokens']==512
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==20 and len({r['image_id'] for r in rows})==20
    return rows,path


def repetition(caption):
    words=caption.lower().split()
    groups=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
    return len(groups)>len(set(groups))


def main():
    baseline,base_path=records(0)
    ids=[r['image_id'] for r in baseline]
    assert all(r['new_tokens']>0 for r in baseline)
    results=[]
    source_hashes={base_path.name:hashlib.sha256(base_path.read_bytes()).hexdigest()}
    for dose in DOSES:
        treated,path=records(dose)
        assert [r['image_id'] for r in treated]==ids
        source_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        ratios=[r['new_tokens']/b['new_tokens'] for r,b in zip(treated,baseline)]
        identical=[r['caption']==b['caption'] for r,b in zip(treated,baseline)]
        results.append({'alpha':dose,'images':20,'mean_length_ratio':float(np.mean(ratios)),
                        'median_length_ratio':float(np.median(ratios)),
                        'less_than_half_count':sum(value<.5 for value in ratios),
                        'identical_caption_count':sum(identical),
                        'mean_word_edit_distance':float(np.mean([distance(b['caption'],r['caption'])
                            for b,r in zip(baseline,treated)])),
                        'ceiling_512_count':sum(r['new_tokens']>=512 for r in treated),
                        'repetition_4gram_count':sum(repetition(r['caption']) for r in treated)})
    pd.DataFrame(results).to_csv(OUT/'tiny_alpha_probe20_v2_summary.csv',index=False,encoding='utf-8-sig')
    low=next((r['alpha'] for r in results if 20-r['identical_caption_count']>=10),None)
    high=next((r['alpha'] for r in reversed(results) if r['less_than_half_count']<=4),None)
    if low is None or high is None or low>high:
        print('NO_VALID_LOG_GRID',low,high,flush=True)
        return
    if low==high:
        print('NO_DISTINCT_FIVE_LEVEL_GRID',low,flush=True)
        return
    grid=[float(value) for value in np.geomspace(low,high,5)]
    payload={'status':'locked_from_output_only_dev_probe','model':'bczhou/TinyLLaVA-3.1B',
             'partition':'dev_common_unmatched500','images':20,'layers_inclusive':[5,18],
             'condition':'detailed512','candidate_doses':list(DOSES),
             'alpha_low_rule':'Smallest tested dose with at least 10/20 captions different from paired baseline.',
             'alpha_high_rule':'Largest tested dose with at most 4/20 captions below half paired baseline length.',
             'alpha_low':low,'alpha_high':high,'grid':'Five equally spaced levels on log scale, inclusive endpoints.',
             'alpha_grid':grid,'selection_data':'Output only; no CHAIR or eval.',
             'source_sha256':source_hashes}
    digest=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    path=OUT/'tiny_alpha_grid_lock.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps({**payload,'rule_sha256':digest},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('LOCKED',low,high,grid,digest,flush=True)


if __name__=='__main__':main()
