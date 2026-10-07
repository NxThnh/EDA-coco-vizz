"""Descriptive mean visual-attention ratio over all generation steps on 20 dev images."""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from transformers import BitsAndBytesConfig, SmolVLMForConditionalGeneration, SmolVLMProcessor

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'eda_results/hgai'
TARGET=OUT/'var_all_steps_smol20_per_image.csv'
BASE=ROOT/'experiment_outputs/alpha_probe20/smol_split_on_alpha_0.000.jsonl'


def main():
    torch.set_num_threads(2)
    torch.manual_seed(927)
    folder=ROOT/'model_assets/SmolVLM2-2.2B-Instruct'
    processor=SmolVLMProcessor.from_pretrained(folder,local_files_only=True)
    processor.image_processor.do_image_splitting=True
    processor.image_processor.size={'longest_edge':1536}
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True)
    model=SmolVLMForConditionalGeneration.from_pretrained(folder,local_files_only=True,
        torch_dtype=torch.float16,low_cpu_mem_usage=True,quantization_config=quant,
        attn_implementation='eager',device_map={'':0}).eval()
    rows=[json.loads(line) for line in BASE.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==20
    if TARGET.exists():
        done=pd.read_csv(TARGET).image_id.drop_duplicates().tolist()
        assert done==[r['image_id'] for r in rows[:len(done)]]
    else:done=[]
    image_id=processor.tokenizer.convert_tokens_to_ids('<image>')
    eos=model.generation_config.eos_token_id
    eos_ids={int(eos)} if isinstance(eos,int) else {int(value) for value in eos}
    for index,row in enumerate(rows[len(done):],start=len(done)+1):
        with Image.open(ROOT/row['image_path']) as raw:
            image=ImageOps.exif_transpose(raw).convert('RGB')
        prompt=processor.apply_chat_template([{'role':'user','content':[
            {'type':'image'},{'type':'text','text':row['prompt']}]}],add_generation_prompt=True)
        inputs=processor(text=prompt,images=[image],return_tensors='pt')
        positions=(inputs.input_ids[0]==image_id).nonzero().flatten()
        device=model.get_input_embeddings().weight.device
        inputs={key:(value.to(device,dtype=torch.float16) if value.is_floating_point() else value.to(device))
                for key,value in inputs.items()}
        positions=positions.to(device)
        sums=np.zeros(24,dtype=np.float64)
        steps=0
        with torch.inference_mode():
            prefix=model(**inputs,use_cache=True,return_dict=True)
            chosen=prefix.logits[:,-1,:].argmax(dim=-1,keepdim=True)
            cache=prefix.past_key_values
            prefix_length=inputs['input_ids'].shape[-1]
            for step in range(512):
                mask=torch.ones((1,prefix_length+step+1),device=device,dtype=torch.long)
                query=model(input_ids=chosen,attention_mask=mask,past_key_values=cache,
                            use_cache=True,output_attentions=True,return_dict=True)
                values=[float(attn[0,:,-1,:].index_select(-1,positions).float().sum(-1).mean().item())
                        for attn in query.attentions]
                assert len(values)==24 and np.isfinite(values).all()
                sums+=values
                steps+=1
                cache=query.past_key_values
                next_token=query.logits[:,-1,:].argmax(dim=-1,keepdim=True)
                if int(chosen.item()) in eos_ids:break
                chosen=next_token
                del query
        records=[{'image_id':row['image_id'],'layer':layer,'var_all_steps':value/steps,
                  'steps_measured':steps} for layer,value in enumerate(sums)]
        pd.DataFrame(records).to_csv(TARGET,mode='a',header=not TARGET.exists(),index=False,
                                     encoding='utf-8-sig' if not TARGET.exists() else 'utf-8')
        print('VAR_ALL_STEPS',index,'/ 20',row['image_id'],steps,flush=True)
        del inputs,prefix,cache
        gc.collect();torch.cuda.empty_cache()
    summary=pd.read_csv(TARGET).groupby('layer',as_index=False).var_all_steps.mean()
    summary.to_csv(OUT/'var_all_steps_smol20_summary.csv',index=False,encoding='utf-8-sig')
    first=pd.read_csv(OUT/'layer_curves_summary.csv')[['layer','var']]
    fig,ax=plt.subplots(figsize=(8,3.5))
    ax.plot(first.layer,first['var'],'o-',markersize=3,label='First generated step')
    ax.plot(summary.layer,summary.var_all_steps,'s-',markersize=3,label='Mean over generated steps')
    ax.set(xlabel='Decoder layer',ylabel='Attention mass on image tokens',
           title='SmolVLM2: visual attention by layer, 20 dev images')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(OUT/'var_all_steps_smol20.png',dpi=220)
    plt.close(fig)
    print('VAR_ALL_STEPS_COMPLETE',flush=True)


if __name__=='__main__':main()
