"""TinyLLaVA 100-image curves with native SDPA and float32 attention readout."""
import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import AttentionInterface
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from hgai_attention import decoder_layers
from measure_layer_curves100 import ROOT,OUT,PREP,load,inputs_for

TARGET=OUT/'layer_curves100_tiny_sdpa_per_image.csv'


def capture_query(model,chosen,mask,cache,positions):
    layers=decoder_layers(model)
    lookup={id(layer.self_attn):index for index,layer in enumerate(layers)}
    observed={}
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    if getattr(original,'_hgai_dispatch',False):raise RuntimeError('Unexpected active attention adapter')
    def dispatch(module,query,key,value,attention_mask,dropout=0.0,scaling=None,**kwargs):
        output,weights=original(module,query,key,value,attention_mask,
                                dropout=dropout,scaling=scaling,**kwargs)
        index=lookup.get(id(module))
        if index is None:return output,weights
        groups=query.shape[1]//key.shape[1]
        assert query.shape[1]%key.shape[1]==0
        full_key=key.repeat_interleave(groups,dim=1) if groups>1 else key
        scores=torch.matmul(query[:,:,-1:,:].float(),full_key.float().transpose(-2,-1))
        scores*=scaling if scaling is not None else query.shape[-1]**-0.5
        if attention_mask is not None:
            applied=attention_mask[..., -1:, :scores.shape[-1]]
            scores=scores.masked_fill(~applied,float('-inf')) if applied.dtype==torch.bool else scores+applied.float()
        probabilities=torch.softmax(scores,dim=-1)
        assert torch.isfinite(probabilities).all()
        pos=torch.as_tensor(positions,device=probabilities.device,dtype=torch.long)
        observed[index]=float(probabilities[0,:,-1,:].index_select(-1,pos).sum(-1).mean().item())
        return output,weights
    AttentionInterface.register('sdpa',dispatch)
    try:
        with torch.inference_mode():
            result=model(input_ids=chosen,attention_mask=mask,past_key_values=cache,
                         use_cache=False,output_hidden_states=True,return_dict=True)
    finally:
        AttentionInterface.register('sdpa',original)
    assert len(observed)==len(layers)==32
    return result,[observed[index] for index in range(32)]


def main(limit):
    torch.set_num_threads(2)
    torch.manual_seed(927)
    manifest=PREP/'dev_common_unmatched500.jsonl'
    source=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    indices=random.Random(927).sample(range(len(source)),100)
    rows=[source[index] for index in indices]
    selection=pd.read_csv(OUT/'layer_curves100_dev_ids.csv')
    assert selection.image_id.tolist()==[r['image_id'] for r in rows]
    if TARGET.exists():
        done=pd.read_csv(TARGET)
        ids=done.image_id.drop_duplicates().tolist()
        assert ids==[r['image_id'] for r in rows[:len(ids)]]
        assert done.groupby('image_id').layer.nunique().eq(32).all()
        assert np.isfinite(done[['var','mask_sensitivity']].to_numpy(float)).all()
    else:ids=[]
    model,processor,tokenizer,image_id,prompt=load('tiny',backend='sdpa')
    for index,row in enumerate(rows[len(ids):limit],start=len(ids)+1):
        with Image.open(ROOT/row['image_path']) as raw:
            image=ImageOps.exif_transpose(raw).convert('RGB')
        chosen=None
        original_positions=None
        hidden={}
        for condition,source_image in [('original',image),('masked',Image.new('RGB',image.size,(127,127,127)))]:
            inputs,positions=inputs_for('tiny',model,processor,tokenizer,source_image,image_id,prompt)
            if original_positions is None:original_positions=positions
            assert positions==original_positions
            with torch.inference_mode():
                prefix=model(**inputs,use_cache=True,return_dict=True)
                if chosen is None:chosen=prefix.logits[:,-1,:].argmax(dim=-1,keepdim=True)
                mask=torch.ones((1,inputs['attention_mask'].shape[-1]+1),device=chosen.device,dtype=torch.long)
                if condition=='original':
                    query,var=capture_query(model,chosen,mask,prefix.past_key_values,positions)
                else:
                    query=model(input_ids=chosen,attention_mask=mask,past_key_values=prefix.past_key_values,
                                use_cache=False,output_hidden_states=True,return_dict=True)
            hidden[condition]=[value[:,-1,:].float().cpu().numpy()[0] for value in query.hidden_states[1:]]
            del inputs,prefix,query
            gc.collect();torch.cuda.empty_cache()
        assert len(var)==len(hidden['original'])==len(hidden['masked'])==32
        records=[]
        for layer,(value,a,b) in enumerate(zip(var,hidden['original'],hidden['masked'])):
            sensitivity=float(np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-8))
            assert np.isfinite(value) and np.isfinite(sensitivity)
            records.append({'image_id':row['image_id'],'layer':layer,'var':value,
                            'mask_sensitivity':sensitivity})
        pd.DataFrame(records).to_csv(TARGET,mode='a',header=not TARGET.exists(),index=False,
                                     encoding='utf-8-sig' if not TARGET.exists() else 'utf-8')
        print('tiny SDPA MEASURED',index,'/',limit,row['image_id'],flush=True)
    if limit==100:
        frame=pd.read_csv(TARGET)
        assert len(frame)==3200 and np.isfinite(frame[['var','mask_sensitivity']].to_numpy(float)).all()
        frame.groupby('layer',as_index=False)[['var','mask_sensitivity']].mean().to_csv(
            OUT/'layer_curves100_tiny_sdpa_summary.csv',index=False,encoding='utf-8-sig')
        print('tiny SDPA COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,choices=[1,100],default=100)
    main(parser.parse_args().limit)
