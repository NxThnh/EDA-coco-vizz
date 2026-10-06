"""Summarize paired output lengths from the short SmolVLM2 alpha probe."""
import json
from pathlib import Path
from statistics import mean, median

from hgai_cpu_audit import ROOT, sha

FOLDER=ROOT/'experiment_outputs/alpha_dose_probe'
BASE=ROOT/'experiment_outputs/hgai_campaign/smol_dev_common_unmatched500_detailed512_splitTrue_a0_l5-18.jsonl'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()][:10]


def main():
    baseline=rows(BASE)
    assert len(baseline)==10
    base=[row['new_tokens'] for row in baseline]
    result=[]
    for alpha in (0,.02,.035,.05,.075,.10,.20,.30):
        label=f'{alpha:.3f}' if alpha in {.035,.075} else f'{alpha:.2f}'
        path=BASE if alpha==0 else FOLDER/f'smol_dev10_alpha_{label}.jsonl'
        current=rows(path)
        if len(current)!=10 or [r['image_id'] for r in current]!=[r['image_id'] for r in baseline]:
            raise ValueError(f'Incomplete or unmatched output: {path}')
        lengths=[row['new_tokens'] for row in current]
        result.append({'alpha':alpha,'images':10,'mean_new_tokens':mean(lengths),
                       'median_new_tokens':median(lengths),
                       'ratio_to_baseline':mean(lengths)/mean(base),
                       'images_below_half_baseline':sum(x<b/2 for x,b in zip(lengths,base)),
                       'images_below_32_tokens':sum(x<32 for x in lengths),
                       'lengths':lengths,'output_sha256':sha(path)})
    dest=FOLDER/'paired_length_analysis.json'
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(dest)


if __name__=='__main__':main()
