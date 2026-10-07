"""Enumerate every contiguous dev control band at both prescribed widths."""
import json
import math
import random
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'eda_results/hgai'
manifest=ROOT/'data_prepared/hgai/dev_common_unmatched500.jsonl'
rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
indices=random.Random(42).sample(range(len(rows)),125)
pd.DataFrame([{'order':position,'manifest_index':index,'image_id':rows[index]['image_id']}
              for position,index in enumerate(indices)]).to_csv(
    OUT/'layer_band_scan_dev125_ids.csv',index=False,encoding='utf-8-sig')

bands=[]
for model,depth in [('smol',24),('tiny',32)]:
    result=OUT/f'layer_rule_v2_100_{model}.json'
    if not result.exists() and model=='smol':result=OUT/'layer_rule_v2_20_smol.json'
    selected=json.loads(result.read_text(encoding='utf-8'))['v2_band'] if result.exists() else None
    selected_width=selected[1]-selected[0]+1 if selected is not None else 14
    for kind,width in [('v2_width',selected_width),('quarter_depth',math.ceil(depth/4))]:
        for start in range(depth-width+1):
            end=start+width-1
            bands.append({'model':model,'depth':depth,'width_kind':kind,'width':width,
                          'start_inclusive':start,'end_inclusive':end,
                          'selected_v2_band':selected==[start,end],
                          'dev_images':125,'dev_seed':42})
pd.DataFrame(bands).to_csv(OUT/'layer_band_scan_plan.csv',index=False,encoding='utf-8-sig')
print('BANDS',len(bands),flush=True)
