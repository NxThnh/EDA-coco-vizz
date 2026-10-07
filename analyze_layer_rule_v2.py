"""Apply immutable v1/v2 layer rules to dev-only measured curves."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'eda_results/hgai'
V1=json.loads((OUT/'layer_rule_lock.json').read_text(encoding='utf-8'))
V2=json.loads((OUT/'layer_rule_v2_lock.json').read_text(encoding='utf-8'))
for lock in (V1,V2):
    payload={k:v for k,v in lock.items() if k!='rule_sha256'}
    digest=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert digest==lock['rule_sha256']


def band(frame,version):
    frame=frame.sort_values('layer')
    assert frame.layer.tolist()==list(range(len(frame)))
    depth=len(frame)
    var=frame['var'].to_numpy(float)
    mask=frame['mask_sensitivity'].to_numpy(float)
    if depth<8 or not np.isfinite(var).all() or not np.isfinite(mask).all():return None
    normalized=[]
    for values in (var,mask):
        if version==1:
            low,high=values.min(),values.max()
        else:
            low,high=np.percentile(values,[10,90])
        if high<=low:return None
        normalized.append(np.clip((values-low)/(high-low),0,1))
    v,s=normalized
    threshold=.6 if version==1 else .5
    minimum=max(2,math.ceil(.1*depth))
    candidates=[]
    start=None
    for index,passed in enumerate([*((v>=threshold)&(s>=threshold)),False]):
        if passed and start is None:start=index
        if not passed and start is not None:
            if index-start>=minimum:
                candidates.append((float(np.minimum(v[start:index],s[start:index]).sum()),start,index-1))
            start=None
    if not candidates:return None
    _,start,end=sorted(candidates,key=lambda row:(-row[0],row[1]))[0]
    return [start,end]


def analyze(path,model,bootstrap=1000,seed=927):
    raw=pd.read_csv(path)
    columns=['image_id','layer','var','mask_sensitivity']
    assert all(column in raw for column in columns)
    ids=raw.image_id.drop_duplicates().tolist()
    depth=int(raw.layer.max())+1
    assert len(raw)==len(ids)*depth
    assert raw.groupby('image_id').layer.nunique().eq(depth).all()
    mean=raw.groupby('layer',as_index=False)[['var','mask_sensitivity']].mean()
    rows={'model':model,'images':len(ids),'layers':depth,'seed':seed,
          'source':str(path.relative_to(ROOT)).replace('\\','/'),
          'v1_rule_sha256':V1['rule_sha256'],'v2_rule_sha256':V2['rule_sha256'],
          'v1_band':band(mean,1),'v2_band':band(mean,2)}
    if bootstrap:
        ordered=raw.set_index(['image_id','layer']).loc[[(id_,layer) for id_ in ids for layer in range(depth)]]
        var=ordered['var'].to_numpy().reshape(len(ids),depth)
        mask=ordered['mask_sensitivity'].to_numpy().reshape(len(ids),depth)
        rng=np.random.default_rng(seed)
        counts=Counter()
        for _ in range(bootstrap):
            sample=rng.integers(0,len(ids),len(ids))
            frame=pd.DataFrame({'layer':range(depth),'var':var[sample].mean(0),
                                'mask_sensitivity':mask[sample].mean(0)})
            result=band(frame,2)
            counts['none' if result is None else f'{result[0]}-{result[1]}']+=1
        rows['bootstrap_replicates']=bootstrap
        rows['v2_bootstrap_frequencies']=[{'band':key,'count':count,'frequency':count/bootstrap}
                                           for key,count in sorted(counts.items())]
    return rows,mean


def plot_curves(mean,model,band_value,path):
    layers=mean.layer.to_numpy()
    var=mean['var'].to_numpy(float)
    mask=mean['mask_sensitivity'].to_numpy(float)
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),sharex=True)
    for values,label in ((var,'VAR'),(mask,'Image-mask sensitivity')):
        axes[0].plot(layers,values,'o-',markersize=3,label=label)
        p10,p90=np.percentile(values,[10,90])
        axes[1].plot(layers,np.clip((values-p10)/(p90-p10),0,1),'o-',markersize=3,label=label)
    axes[1].axhline(.5,linestyle='--',color='#a54b43',linewidth=1,label='v2 threshold 0.50')
    if band_value is not None:
        axes[1].axvspan(band_value[0]-.4,band_value[1]+.4,color='#bbd8e5',alpha=.35)
    axes[0].set(title='Raw curves',xlabel='Decoder layer',ylabel='Measured value')
    axes[1].set(title='P10/P90 normalized curves',xlabel='Decoder layer',ylabel='Clipped value',ylim=(-.04,1.04))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'{model}: 100 seeded dev images')
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)


if __name__=='__main__':
    jobs=[(OUT/'layer_curves_per_image.csv','SmolVLM2-2.2B',20),
          (OUT/'layer_curves100_smol_per_image.csv','SmolVLM2-2.2B',100),
          (OUT/'layer_curves100_tiny_sdpa_per_image.csv','bczhou/TinyLLaVA-3.1B',100)]
    for path,model,expected in jobs:
        if not path.exists():continue
        result,mean=analyze(path,model,1000 if expected==100 else 0)
        assert result['images']==expected
        stem='layer_rule_v2_20_smol' if expected==20 else 'layer_rule_v2_100_'+('smol' if 'Smol' in model else 'tiny')
        (OUT/(stem+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        mean.to_csv(OUT/(stem+'_mean.csv'),index=False,encoding='utf-8-sig')
        if expected==100:
            plot_curves(mean,model,result['v2_band'],OUT/(stem+'_curves.png'))
        print(stem,result['v1_band'],result['v2_band'],flush=True)
