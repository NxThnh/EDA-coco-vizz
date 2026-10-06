"""Model-specific annotation overlap, separating exact manifests from upstream pools."""
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from hgai_cpu_audit import ROOT, OUT, PREP, sha


def scan(path, vg_to_coco):
    direct=set();aliases=set();prefixes=Counter();records=0
    with path.open('rb') as f:
        for line in f:
            if not line.lstrip().startswith(b'"image":'):continue
            value=json.loads(line.split(b':',1)[1].strip().rstrip(b','))
            if not isinstance(value,str):continue
            records+=1
            prefixes[value.split('/')[0]]+=1
            match=re.search(r'(\d+)\.(?:jpe?g|png)$',value,re.I)
            if not match:continue
            i=int(match.group(1))
            if re.search(r'(?:^|/)coco(?:/|_)',value,re.I) or 'COCO_' in value:direct.add(i)
            elif value.split('/')[0] in {'gqa','vg'} and i in vg_to_coco:aliases.add(vg_to_coco[i])
    return direct,aliases,{'records_with_image':records,'prefix_counts':dict(prefixes),'sha256':sha(path)}


def main():
    with zipfile.ZipFile(ROOT/'scratch/image_data.json.zip') as z:
        metadata=json.loads(z.read('image_data.json'))
    mapping={int(r['image_id']):int(r['coco_id']) for r in metadata if r.get('coco_id') is not None}
    files={'llava_sft': 'llava_v1_5_mix665k.json',
           'share_sft': 'sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json',
           'share_pretrain': 'share-captioner_coco_lcs_sam_1246k_1107.json',
           'share_cap100k': 'sharegpt4v_instruct_gpt4-vision_cap100k.json'}
    pools={};sources={}
    for key,name in files.items():
        d,a,provenance=scan(ROOT/'scratch'/name,mapping)
        pools[key]=(d,a);sources[key]=provenance
        sources[key]['path']='scratch/'+name
        print('Scanned:',key,'direct',len(d),'aliases',len(a),flush=True)
    if (ROOT/'scratch/onevision_sharegpt4v_ids.csv').exists():
        ids=pd.read_csv(ROOT/'scratch/onevision_sharegpt4v_ids.csv',dtype=str)
        coco=ids[ids.source_slice=='sharegpt4v(coco)'].source_id
        pools['smol_onevision_coco']=(set(int(i) for i in coco if i.isdigit()),set())
        sources['smol_onevision_coco']={'path':'scratch/onevision_sharegpt4v_ids.csv',
                                      'sha256':sha(ROOT/'scratch/onevision_sharegpt4v_ids.csv'),
                                      'status':'upstream_candidate_pool_not_exact_training_subset'}
    else:
        pools['smol_onevision_coco']=(set(),set())
        sources['smol_onevision_coco']={'status':'id_metadata_unavailable'}
    union_pools={key:d|a for key,(d,a) in pools.items()}
    data=json.loads((ROOT/'coco/annotations/captions_val2014.json').read_text(encoding='utf-8'))
    partitions={}
    for path in PREP.glob('*.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);partitions[r['image_id']]=r['partition']
    rows=[]
    for image in data['images']:
        i=image['id'];r={'image_id':i,'partition':partitions.get(i,'unused_val')}
        for key,(d,a) in pools.items():
            r[key+'_direct']=int(i in d);r[key+'_alias']=int(i in a);r[key+'_union']=int(i in union_pools[key])
        r['tiny_llava_recipe_known']=r['llava_sft_union']
        r['tiny_share_recipe_known']=int(r['share_sft_union'] or r['share_pretrain_union'])
        r['smol_checked_upstream_candidate']=r['smol_onevision_coco_union']
        rows.append(r)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'small_vlm_training_overlap.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for scope,group in [('val2014',frame),*[(p,g) for p,g in frame.groupby('partition') if p!='unused_val']]:
        for model,column,status in [
            ('LLaVA-1.5-7B','llava_sft_union','exact_SFT_manifest_ID_join'),
            ('TinyLLaVA/LLaVA-recipe','tiny_llava_recipe_known','recipe_conditional_exact_SFT_ID_join'),
            ('TinyLLaVA/ShareGPT4V-recipe','tiny_share_recipe_known','recipe_conditional_pretrain_and_SFT_ID_join'),
            ('SmolVLM2/OneVision-ShareGPT4V-COCO','smol_checked_upstream_candidate','upstream_pool_candidate_not_verified_training_exposure')]:
            summary.append({'scope':scope,'model_or_recipe':model,'images':len(group),
                            'matched_ids':int(group[column].sum()),'matched_pct':100*group[column].mean(),
                            'status':status})
    pd.DataFrame(summary).to_csv(OUT/'small_vlm_training_overlap_summary.csv',index=False,encoding='utf-8-sig')
    result={'sources':sources,'tiny_checkpoint_recipe':'Model card lists both recipes; no exact per-checkpoint full training record supplied. Report both conditionally.',
            'smol_training_sources':['LLaVA-OneVision','M4-Instruct','MAmmoTH-VL','LLaVA-Video-178K','FineVideo','Video-STaR','VRipt','VISTA-400K','MovieChat','ShareGPT4Video'],
            'smol_scope':'Only exact upstream OneVision ShareGPT4V-COCO IDs checked; other slices, base SmolVLM training and sampled 3.3M training selection remain unknown.',
            'not_clean':'Absence of a match never proves absence from all pretraining or unseen sources.',
            'code_sha256':sha(Path(__file__))}
    (OUT/'small_vlm_training_overlap.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
