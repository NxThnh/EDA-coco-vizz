"""Derive reader-facing figures and descriptive tables from frozen artifacts."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'eda_results/hgai'
LOCK=json.loads((OUT/'layer_rule_lock.json').read_text(encoding='utf-8'))


def band(frame,threshold,drop_zero):
    selected=frame[frame.layer.ne(0)].copy() if drop_zero else frame.copy()
    v=selected['var'].to_numpy(float)
    s=selected['mask_sensitivity'].to_numpy(float)
    if len(v)<8 or not np.isfinite(v).all() or not np.isfinite(s).all() or np.ptp(v)==0 or np.ptp(s)==0:
        return None
    v=(v-v.min())/np.ptp(v)
    s=(s-s.min())/np.ptp(s)
    passed=(v>=threshold)&(s>=threshold)
    width=max(2,math.ceil(.10*len(v)))
    choices=[]
    start=None
    for i,ok in enumerate([*passed,False]):
        if ok and start is None:start=i
        elif not ok and start is not None:
            if i-start>=width:
                choices.append((float(np.minimum(v[start:i],s[start:i]).sum()),
                                int(selected.iloc[start].layer),int(selected.iloc[i-1].layer)))
            start=None
    if not choices:return None
    _,lo,hi=sorted(choices,key=lambda item:(-item[0],item[1]))[0]
    return [lo,hi]


def curves():
    frame=pd.read_csv(OUT/'layer_curves_summary.csv').sort_values('layer')
    assert frame.layer.tolist()==list(range(len(frame)))
    assert LOCK['threshold']==.6
    rows=[]
    for threshold in (.5,.6,.7):
        for drop_zero in (False,True):
            found=band(frame,threshold,drop_zero)
            rows.append({'threshold':threshold,'include_layer_0':not drop_zero,
                         'band':'không tìm thấy ranh giới' if found is None else f'{found[0]}–{found[1]}',
                         'layers_considered':len(frame)-int(drop_zero)})
    pd.DataFrame(rows).to_csv(OUT/'layer_rule_sensitivity.csv',index=False,encoding='utf-8-sig')
    assert rows[2]['band']=='không tìm thấy ranh giới'
    layers=frame.layer.to_numpy()
    raw_var=frame['var'].to_numpy(float)
    raw_mask=frame['mask_sensitivity'].to_numpy(float)
    norm_var=(raw_var-raw_var.min())/np.ptp(raw_var)
    norm_mask=(raw_mask-raw_mask.min())/np.ptp(raw_mask)
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),sharex=True)
    axes[0].plot(layers,raw_var,'o-',label='VAR',markersize=3)
    axes[0].plot(layers,raw_mask,'s-',label='Độ nhạy che ảnh',markersize=3)
    axes[0].set(title='Giá trị thô',xlabel='Layer',ylabel='Giá trị đo')
    axes[1].plot(layers,norm_var,'o-',label='VAR',markersize=3)
    axes[1].plot(layers,norm_mask,'s-',label='Độ nhạy che ảnh',markersize=3)
    axes[1].axhline(.6,color='#af4b40',linestyle='--',linewidth=1,label='Ngưỡng 0,60')
    axes[1].set(title='Chuẩn hóa min–max',xlabel='Layer',ylabel='Giá trị chuẩn hóa',ylim=(-.04,1.05))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('SmolVLM2, 20 ảnh dev, bước sinh đầu')
    fig.tight_layout()
    fig.savefig(OUT/'layer_curves_raw_normalized.png',dpi=220)
    plt.close(fig)


def repetition(text):
    words=text.lower().split()
    groups=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
    return (len(groups)-len(set(groups)))/len(groups) if groups else 0.0


def alpha_probe():
    folder=ROOT/'experiment_outputs/alpha_probe20'
    modes={'split_on':'Chia ô mặc định','split_768':'Chia ô 768','split_off':'Tắt chia ô'}
    rows=[]
    for mode,label in modes.items():
        base=[json.loads(line) for line in (folder/f'smol_{mode}_alpha_0.000.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(base)==20
        for dose in (.02,.035,.05,.075,.10):
            path=folder/f'smol_{mode}_alpha_{dose:.3f}.jsonl'
            current=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            assert len(current)==20
            assert [r['image_id'] for r in current]==[r['image_id'] for r in base]
            ratios=[r['new_tokens']/b['new_tokens'] for r,b in zip(current,base)]
            repeats=[repetition(r['caption']) for r in current]
            rows.append({'mode':mode,'mode_label':label,'alpha':dose,'images':20,
                         'median_length_ratio':float(np.median(ratios)),
                         'ceiling_512_count':sum(r['new_tokens']>=512 for r in current),
                         'repetition_any_count':sum(value>0 for value in repeats),
                         'mean_repeated_4gram_fraction':float(np.mean(repeats))})
    pd.DataFrame(rows).to_csv(OUT/'alpha_probe20_extra.csv',index=False,encoding='utf-8-sig')


def tiny_alpha_probe():
    folder=ROOT/'experiment_outputs/tiny_alpha_probe10'
    doses=(0,.02,.035,.05,.075,.10)
    paths=[folder/f'tiny_pad_alpha_{dose:.3f}.jsonl' for dose in doses]
    if not all(path.exists() and len(path.read_text(encoding='utf-8').splitlines())==10 for path in paths):
        return False
    groups=[[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] for path in paths]
    base=groups[0]
    assert len(set(r['image_id'] for r in base))==10
    rows=[]
    for dose,group in zip(doses,groups):
        assert [r['image_id'] for r in group]==[r['image_id'] for r in base]
        ratios=[r['new_tokens']/b['new_tokens'] for r,b in zip(group,base)]
        repeats=[repetition(r['caption']) for r in group]
        rows.append({'alpha':dose,'images':len(group),
                     'median_length_ratio':float(np.median(ratios)),
                     'mean_output_tokens':float(np.mean([r['new_tokens'] for r in group])),
                     'less_than_half_count':sum(value<.5 for value in ratios),
                     'ceiling_512_count':sum(r['new_tokens']>=512 for r in group),
                     'repetition_any_count':sum(value>0 for value in repeats),
                     'mean_repeated_4gram_fraction':float(np.mean(repeats))})
    pd.DataFrame(rows).to_csv(OUT/'tiny_alpha_probe10_summary.csv',index=False,encoding='utf-8-sig')
    return True


if __name__=='__main__':
    curves()
    alpha_probe()
    tiny_alpha_probe()
