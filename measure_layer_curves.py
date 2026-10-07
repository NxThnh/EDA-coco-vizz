"""Measure the frozen dev-only Smol VAR and image-mask curves, then apply the locked rule."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import BitsAndBytesConfig, SmolVLMForConditionalGeneration, SmolVLMProcessor

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'eda_results/hgai'
PREP=ROOT/'data_prepared/hgai'
LOCK=json.loads((OUT/'layer_rule_lock.json').read_text(encoding='utf-8'))
APPROVAL=json.loads((OUT/'layer_rule_approval.json').read_text(encoding='utf-8'))
digest=LOCK['rule_sha256']
payload={k:v for k,v in LOCK.items() if k!='rule_sha256'}
assert hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()==digest
assert APPROVAL['rule_sha256']==digest
assert LOCK['status']=='approved_for_dev_measurement'
assert LOCK['measurement_protocol']['partition']=='dev_common_unmatched500'


def select_band(frame):
    var=frame['var'].to_numpy(dtype=float)
    sensitivity=frame['mask_sensitivity'].to_numpy(dtype=float)
    depth=len(var)
    if depth<8 or not np.isfinite(var).all() or not np.isfinite(sensitivity).all():
        return None
    if np.ptp(var)==0 or np.ptp(sensitivity)==0:
        return None
    v=(var-var.min())/np.ptp(var)
    s=(sensitivity-sensitivity.min())/np.ptp(sensitivity)
    both=(v>=LOCK['threshold'])&(s>=LOCK['threshold'])
    width=max(2,math.ceil(.10*depth))
    candidates=[]
    start=None
    for i,passed in enumerate([*both,False]):
        if passed and start is None:start=i
        elif not passed and start is not None:
            end=i-1
            if end-start+1>=width:
                score=float(np.minimum(v[start:i],s[start:i]).sum())
                candidates.append((score,start,end))
            start=None
    if not candidates:return None
    _,start,end=sorted(candidates,key=lambda t:(-t[0],t[1]))[0]
    return {'start_inclusive':start,'end_inclusive':end,'minimum_width':width}


def main(limit):
    torch.set_num_threads(2)
    torch.manual_seed(927)
    model_path=ROOT/'model_assets/SmolVLM2-2.2B-Instruct'
    processor=SmolVLMProcessor.from_pretrained(model_path,local_files_only=True)
    processor.image_processor.do_image_splitting=True
    processor.image_processor.size={'longest_edge':1536}
    image_id=processor.tokenizer.convert_tokens_to_ids('<image>')
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16,llm_int8_enable_fp32_cpu_offload=True)
    model=SmolVLMForConditionalGeneration.from_pretrained(model_path,local_files_only=True,
        torch_dtype=torch.float16,low_cpu_mem_usage=True,quantization_config=quant,
        attn_implementation='eager',device_map={'':0}).eval()
    manifest=PREP/'dev_common_unmatched500.jsonl'
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()[:20]]
    text=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},
           {'type':'text','text':LOCK['measurement_protocol']['prompt']}]}],add_generation_prompt=True)
    measured=[]
    for index,row in enumerate(rows[:limit],1):
        with Image.open(ROOT/row['image_path']) as raw:
            image=ImageOps.exif_transpose(raw).convert('RGB')
        hidden={}
        chosen=None
        original_positions=None
        for condition,source in [('original',image),('masked',Image.new('RGB',image.size,(127,127,127)))]:
            inputs=processor(text=text,images=[source],return_tensors='pt')
            positions=(inputs.input_ids[0]==image_id).nonzero().flatten().tolist()
            if original_positions is None:original_positions=positions
            assert positions==original_positions
            device=model.get_input_embeddings().weight.device
            inputs={k:v.to(device,dtype=torch.float16) if v.is_floating_point() else v.to(device) for k,v in inputs.items()}
            with torch.inference_mode():
                prefix=model(**inputs,use_cache=True,return_dict=True)
                if chosen is None:chosen=prefix.logits[:,-1,:].argmax(dim=-1,keepdim=True)
                attention_mask=torch.ones((1,inputs['input_ids'].shape[-1]+1),dtype=torch.long,device=device)
                query=model(input_ids=chosen,attention_mask=attention_mask,
                            past_key_values=prefix.past_key_values,use_cache=False,
                            output_attentions=(condition=='original'),output_hidden_states=True,return_dict=True)
            hidden[condition]=[h[:,-1,:].float().cpu().numpy()[0] for h in query.hidden_states[1:]]
            if condition=='original':
                assert query.attentions is not None
                pos=torch.tensor(positions,device=device,dtype=torch.long)
                var=[float(a[0,:,-1,:].index_select(-1,pos).float().sum(-1).mean().item())
                     for a in query.attentions]
            del inputs,prefix,query
            gc.collect();torch.cuda.empty_cache()
        assert len(var)==len(hidden['original'])==len(hidden['masked'])
        for layer,(value,a,b) in enumerate(zip(var,hidden['original'],hidden['masked'])):
            sensitivity=float(np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-8))
            measured.append({'image_id':row['image_id'],'layer':layer,'var':value,
                             'mask_sensitivity':sensitivity})
        print('MEASURED',index,'/',limit,row['image_id'],flush=True)
    if limit!=20:
        print('Smoke only; no lock result written',flush=True)
        return
    pd.DataFrame(measured).to_csv(OUT/'layer_curves_per_image.csv',index=False,encoding='utf-8-sig')
    summary=pd.DataFrame(measured).groupby('layer',as_index=False)[['var','mask_sensitivity']].mean()
    summary.to_csv(OUT/'layer_curves_summary.csv',index=False,encoding='utf-8-sig')
    band=select_band(summary)
    result={'rule_sha256':digest,'images':20,'layers':len(summary),'band':band or 'không tìm thấy ranh giới',
            'source':'eda_results/hgai/layer_curves_summary.csv',
            'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (OUT/'layer_boundary_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('BOUNDARY',result['band'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--limit',type=int,default=20)
    main(p.parse_args().limit)
