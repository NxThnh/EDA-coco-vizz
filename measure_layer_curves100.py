"""Measure first-step VAR and image-mask sensitivity on 100 seeded dev images."""
import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import (AutoImageProcessor, AutoTokenizer, BitsAndBytesConfig,
                          SmolVLMForConditionalGeneration, SmolVLMProcessor)

from tinyllava_model import TinyPhiConfig, TinyPhiForCausalLM, validate_checkpoint_mapping

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'eda_results/hgai'
PREP=ROOT/'data_prepared/hgai'
PROMPT='Please help me describe the image in detail.'


def load(model_key,backend='eager'):
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True)
    common={'local_files_only':True,'torch_dtype':torch.float16,'low_cpu_mem_usage':True,
            'quantization_config':quant,'attn_implementation':backend,'device_map':{'':0}}
    if model_key=='smol':
        folder=ROOT/'model_assets/SmolVLM2-2.2B-Instruct'
        processor=SmolVLMProcessor.from_pretrained(folder,local_files_only=True)
        processor.image_processor.do_image_splitting=True
        processor.image_processor.size={'longest_edge':1536}
        model=SmolVLMForConditionalGeneration.from_pretrained(folder,**common).eval()
        tokenizer=processor.tokenizer
        image_id=tokenizer.convert_tokens_to_ids('<image>')
        prompt=processor.apply_chat_template([{'role':'user','content':[
            {'type':'image'},{'type':'text','text':PROMPT}]}],add_generation_prompt=True)
        return model,processor,tokenizer,image_id,prompt
    folder=ROOT/'model_assets/TinyLLaVA-3.1B'
    config=TinyPhiConfig.from_pretrained(folder,local_files_only=True)
    model=TinyPhiForCausalLM.from_pretrained(folder,config=config,**common).eval()
    validate_checkpoint_mapping(model,folder)
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    processor=AutoImageProcessor.from_pretrained(ROOT/'model_assets/siglip-so400m-patch14-384',local_files_only=True)
    return model,processor,tokenizer,None,PROMPT


def inputs_for(model_key,model,processor,tokenizer,image,image_id,prompt):
    if model_key=='smol':
        inputs=processor(text=prompt,images=[image],return_tensors='pt')
        positions=(inputs.input_ids[0]==image_id).nonzero().flatten().tolist()
        device=model.get_input_embeddings().weight.device
        inputs={key:(value.to(device,dtype=torch.float16) if value.is_floating_point() else value.to(device))
                for key,value in inputs.items()}
        return inputs,positions
    side=max(image.size)
    pad=Image.new('RGB',(side,side),tuple(int(value*255) for value in processor.image_mean))
    pad.paste(image,((side-image.width)//2,(side-image.height)//2))
    pixels=processor(images=pad,return_tensors='pt').pixel_values
    return model.prepare_visual_inputs(tokenizer,pixels,prompt)


def main(model_key):
    if model_key!='smol':
        raise ValueError('Use measure_layer_curves100_tiny_sdpa.py for TinyLLaVA; eager attention yielded NaNs.')
    torch.set_num_threads(2)
    torch.manual_seed(927)
    manifest=PREP/'dev_common_unmatched500.jsonl'
    all_rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    selected_indices=random.Random(927).sample(range(len(all_rows)),100)
    rows=[all_rows[index] for index in selected_indices]
    selection_path=OUT/'layer_curves100_dev_ids.csv'
    selection=pd.DataFrame([{'order':position,'manifest_index':source_index,
                             'image_id':all_rows[source_index]['image_id']}
                            for position,source_index in enumerate(selected_indices)])
    if selection_path.exists():
        assert pd.read_csv(selection_path).equals(selection)
    else:
        selection.to_csv(selection_path,index=False,encoding='utf-8-sig')
    path=OUT/f'layer_curves100_{model_key}_per_image.csv'
    if path.exists():
        done=pd.read_csv(path)
        done_ids=done.image_id.drop_duplicates().tolist()
        assert done_ids==[r['image_id'] for r in rows[:len(done_ids)]]
        assert done.groupby('image_id').layer.nunique().eq(24 if model_key=='smol' else 32).all()
    else:done_ids=[]
    model,processor,tokenizer,image_id,prompt=load(model_key)
    depth=24 if model_key=='smol' else 32
    for index,row in enumerate(rows[len(done_ids):],start=len(done_ids)+1):
        with Image.open(ROOT/row['image_path']) as raw:
            image=ImageOps.exif_transpose(raw).convert('RGB')
        hidden={}
        chosen=None
        original_positions=None
        for condition,source in [('original',image),('masked',Image.new('RGB',image.size,(127,127,127)))]:
            inputs,positions=inputs_for(model_key,model,processor,tokenizer,source,image_id,prompt)
            if original_positions is None:original_positions=positions
            assert positions==original_positions
            with torch.inference_mode():
                prefix=model(**inputs,use_cache=True,return_dict=True)
                if chosen is None:chosen=prefix.logits[:,-1,:].argmax(dim=-1,keepdim=True)
                attention_mask=torch.ones((1,inputs['attention_mask'].shape[-1]+1),
                                          dtype=torch.long,device=chosen.device)
                query=model(input_ids=chosen,attention_mask=attention_mask,
                            past_key_values=prefix.past_key_values,use_cache=False,
                            output_attentions=(condition=='original'),output_hidden_states=True,
                            return_dict=True)
            hidden[condition]=[value[:,-1,:].float().cpu().numpy()[0] for value in query.hidden_states[1:]]
            if condition=='original':
                assert query.attentions is not None and len(query.attentions)==depth
                pos=torch.tensor(positions,device=chosen.device,dtype=torch.long)
                var=[float(attn[0,:,-1,:].index_select(-1,pos).float().sum(-1).mean().item())
                     for attn in query.attentions]
            del inputs,prefix,query
            gc.collect();torch.cuda.empty_cache()
        records=[]
        for layer,(value,a,b) in enumerate(zip(var,hidden['original'],hidden['masked'])):
            records.append({'image_id':row['image_id'],'layer':layer,'var':value,
                            'mask_sensitivity':float(np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-8))})
        pd.DataFrame(records).to_csv(path,mode='a',header=not path.exists(),index=False,
                                     encoding='utf-8-sig' if not path.exists() else 'utf-8')
        print(model_key,'MEASURED',index,'/ 100',row['image_id'],flush=True)
    summary=pd.read_csv(path).groupby('layer',as_index=False)[['var','mask_sensitivity']].mean()
    summary.to_csv(OUT/f'layer_curves100_{model_key}_summary.csv',index=False,encoding='utf-8-sig')
    print(model_key,'COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['smol'],required=True)
    main(parser.parse_args().model)
