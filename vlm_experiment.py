"""Shared, resumable CHAIR/POPE baseline and HGAI runner for all three VLMs."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
os.environ.setdefault('HF_HUB_OFFLINE','1')

from PIL import Image, ImageOps
from hgai_cpu_audit import ROOT,OUT,PREP,sha
from lock_common_unmatched import prepare_common

MODELS={'smol':('HuggingFaceTB/SmolVLM2-2.2B-Instruct','SmolVLM2-2.2B-Instruct','smol'),
        'tiny':('bczhou/TinyLLaVA-3.1B','TinyLLaVA-3.1B','tiny'),
        'llava':('llava-hf/llava-1.5-7b-hf','llava-1.5-7b-hf','llava')}
DETAIL='Please help me describe the image in detail.'
CONCISE='Describe the visible content in one concise sentence. If the image is too unclear to identify reliably, say "I cannot tell from this image."'


def run(args):
    import torch
    import transformers
    from transformers import (SmolVLMProcessor,SmolVLMForConditionalGeneration,LlavaProcessor,
                              LlavaForConditionalGeneration,AutoTokenizer,AutoImageProcessor,BitsAndBytesConfig)
    from hgai_attention import HGAIAdapter,decoder_layers
    args.image_mode=args.image_mode or ('pad' if args.model=='tiny' else 'hgai_crop')
    if args.model=='tiny' and args.image_mode!='pad':
        raise ValueError('Pinned legacy TinyLLaVA uses pad384; crop is only an EDA geometry control, not this checkpoint protocol')
    if transformers.__version__!='4.53.3' or not torch.cuda.is_available():
        raise RuntimeError('Use pinned Transformers 4.53.3 and a CUDA GPU')
    if args.limit<1:raise ValueError('Limit must be positive')
    prepare_common()
    repo,folder,weight_key=MODELS[args.model]
    model_path=ROOT/'model_assets'/folder
    weights=json.loads((OUT/(weight_key+'_weights_provenance.json')).read_text())
    for item in weights:
        p=model_path/item['file']
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise ValueError('Missing/corrupt pinned weight: '+str(p))
    if args.task=='chair':
        manifest=PREP/(args.partition+'.jsonl')
        rows=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
        prompt=CONCISE if args.condition=='concise64' else DETAIL
        for row in rows:row['prompt']=prompt
        max_tokens=512 if args.condition=='detailed512' else 64
        partition=args.partition
    else:
        manifest=ROOT/'eda_resources/pope_official'/('coco_pope_'+args.pope_variant+'.json')
        source=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
        rows=[{'question_id':r['question_id'],'image_id':int(Path(r['image']).stem.split('_')[-1]),
               'image_path':'coco/val2014/'+r['image'],'prompt':r['text'],'label':r['label']} for r in source]
        max_tokens=64
        partition='official-pope-'+args.pope_variant
    benchmark_rows=len(rows)
    if args.limit>benchmark_rows:raise ValueError('Limit exceeds benchmark')
    rows=rows[:args.limit]
    depth=24 if args.model=='smol' else 32
    layers=list(range(args.layers[0],args.layers[1]+1))
    if args.layers[0]<0 or args.layers[1]<args.layers[0] or args.layers[1]>=depth:
        raise ValueError('Invalid inclusive layer range')
    condition=args.condition if args.task=='chair' else 'pope64'
    name=f'{args.model}_{partition}_{condition}_{args.precision}_{args.image_mode}_split{not args.split_off}_alpha{args.alpha}_layers{args.layers[0]}-{args.layers[1]}_n{args.limit}'
    output=args.output or ROOT/'experiment_outputs/vlm_runs'/(name+'.jsonl')
    output.parent.mkdir(parents=True,exist_ok=True)
    config_path=output.with_suffix('.config.json')
    sources=['vlm_experiment.py','hgai_attention.py','tinyllava_model.py']
    config={'schema_version':3,'model':repo,'model_key':args.model,'revision':weights[0]['revision'],
            'task':args.task,'partition':partition,'condition':condition,'manifest_sha256':sha(manifest),
            'weights':weights,'precision':args.precision,'image_splitting':not args.split_off if args.model=='smol' else None,
            'image_mode':args.image_mode,'max_new_tokens':max_tokens,'requested_rows':args.limit,'benchmark_rows':benchmark_rows,
            'role':'feasibility_pilot' if args.limit<benchmark_rows else 'baseline' if args.alpha==0 else 'hgai',
            'alpha':args.alpha,'layers_inclusive':args.layers,'aggregation':'mean','intervention':'none' if args.alpha==0 else 'HGAI',
            'seed':927,'do_sample':False,'num_beams':1,'use_cache':True,'transformers':transformers.__version__,
            'code_sha256':{f:sha(ROOT/f) for f in sources},
            'assets_sha256':{p.name:sha(p) for p in model_path.iterdir() if p.suffix in {'.json','.model','.txt','.jinja'}},
            'annotation_sha256':{name:sha(ROOT/'coco/annotations'/name) for name in ['instances_val2014.json','captions_val2014.json']}}
    done=[]
    if output.exists():
        if not args.resume:raise FileExistsError('Output exists; use identical --resume or another path')
        if json.loads(config_path.read_text())!=config:raise ValueError('Resume config/source bytes differ')
        done=[json.loads(s) for s in output.read_text(encoding='utf-8').splitlines()]
        key='image_id' if args.task=='chair' else 'question_id'
        if len(done)>len(rows) or [r[key] for r in done]!=[r[key] for r in rows[:len(done)]]:
            raise ValueError('Output is not a locked manifest prefix')
    elif config_path.exists():
        if not args.resume or json.loads(config_path.read_text())!=config:
            raise ValueError('Existing config without output; inspect before resuming')
    else:config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    if len(done)==len(rows):
        print('Already complete',output,flush=True);return
    torch.set_num_threads(2);torch.manual_seed(927)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
          bnb_4bit_compute_dtype=torch.float16,llm_int8_enable_fp32_cpu_offload=True) if args.precision=='nf4' else None
    loading={'local_files_only':True,'torch_dtype':torch.float16,'low_cpu_mem_usage':True,
             'quantization_config':quant,'attn_implementation':'sdpa'}
    if args.model=='llava':loading.update(device_map='auto',max_memory={0:'2700MiB','cpu':'10GiB'})
    else:loading['device_map']={'':0}
    print('Loading',args.model,'precision',args.precision,flush=True)
    if args.model=='smol':
        processor=SmolVLMProcessor.from_pretrained(model_path,local_files_only=True)
        processor.image_processor.do_image_splitting=not args.split_off
        model=SmolVLMForConditionalGeneration.from_pretrained(model_path,**loading).eval()
        tokenizer=processor.tokenizer
    elif args.model=='llava':
        processor=LlavaProcessor.from_pretrained(model_path,local_files_only=True)
        processor.patch_size=14;processor.num_additional_image_tokens=1;processor.vision_feature_select_strategy='default'
        if hasattr(processor.image_processor,'do_pad'):processor.image_processor.do_pad=False
        model=LlavaForConditionalGeneration.from_pretrained(model_path,**loading).eval()
        tokenizer=processor.tokenizer
    else:
        from tinyllava_model import TinyPhiConfig,TinyPhiForCausalLM,validate_checkpoint_mapping
        cfg=TinyPhiConfig.from_pretrained(model_path,local_files_only=True)
        model=TinyPhiForCausalLM.from_pretrained(model_path,config=cfg,**loading).eval()
        validate_checkpoint_mapping(model,model_path)
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        processor=AutoImageProcessor.from_pretrained(ROOT/'model_assets/siglip-so400m-patch14-384',local_files_only=True)
    actual_depth=len(decoder_layers(model))
    if actual_depth!=depth:raise ValueError('Checkpoint decoder depth differs')
    adapter=HGAIAdapter(model,layers,args.alpha)
    start=time.perf_counter()
    with adapter,output.open('a' if done else 'x',encoding='utf-8') as stream,torch.inference_mode():
        for index,row in enumerate(rows[len(done):],start=len(done)):
            with Image.open(ROOT/row['image_path']) as raw:image=ImageOps.exif_transpose(raw).convert('RGB')
            if args.model=='tiny' or args.image_mode=='pad':
                mean=processor.image_mean if args.model=='tiny' else processor.image_processor.image_mean
                side=max(image.size);padded=Image.new('RGB',(side,side),tuple(int(v*255) for v in mean))
                padded.paste(image,((side-image.width)//2,(side-image.height)//2));image=padded
            if args.model=='tiny':
                pixels=processor(images=image,return_tensors='pt').pixel_values
                inputs,positions=model.prepare_visual_inputs(tokenizer,pixels,row['prompt'])
                prefix=0 # HF inputs_embeds-only generation returns only newly generated IDs.
            else:
                if args.model=='smol':
                    text=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':row['prompt']}]}],add_generation_prompt=True)
                else:
                    text="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n"+row['prompt']+' ASSISTANT:'
                inputs=processor(text=text,images=[image],return_tensors='pt')
                image_id=model.config.image_token_id if args.model=='smol' else model.config.image_token_index
                positions=(inputs.input_ids[0]==image_id).nonzero().flatten().tolist()
                if not positions:raise ValueError('No expanded visual placeholders')
                input_device=model.get_input_embeddings().weight.device
                inputs={k:v.to(input_device,dtype=torch.float16) if v.is_floating_point() else v.to(input_device) for k,v in inputs.items()}
                prefix=inputs['input_ids'].shape[-1]
            adapter.set_image_positions(positions)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
            generated=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,num_beams=1,
                                     use_cache=True,pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize();tokens=generated[0,prefix:]
            answer=tokenizer.decode(tokens,skip_special_tokens=True).strip()
            record={k:v for k,v in row.items() if k not in {'valid_captions','prompt'}}
            record.update({'prompt':row['prompt'],'caption' if args.task=='chair' else 'answer':answer,
                           'new_tokens':len(tokens),'image_tokens':len(positions),'truncated':len(tokens)==max_tokens,
                           'elapsed_seconds':time.perf_counter()-tick,'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),
                           'hgai_modified_calls':adapter.modified_calls})
            stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            print(args.model,args.task,index+1,'/',len(rows),len(tokens),'tokens',round(record['elapsed_seconds'],2),'s',flush=True)
            del inputs,generated,tokens
    output.with_suffix('.run.json').write_text(json.dumps({'status':'complete','rows':len(rows),'role':config['role'],
         'output_sha256':sha(output),'config_sha256':sha(config_path),'seconds_this_run':time.perf_counter()-start,
         'device':torch.cuda.get_device_name(),'torch':torch.__version__,'device_map':str(getattr(model,'hf_device_map',{})),
         'adapter_calls':adapter.calls,'modified_calls':adapter.modified_calls},indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--task',choices=['chair','pope'],default='chair')
    parser.add_argument('--partition',choices=['dev_common_unmatched500','eval_common_unmatched500','eval_random500'],default='dev_common_unmatched500')
    parser.add_argument('--pope-variant',choices=['random','popular','adversarial'],default='random')
    parser.add_argument('--condition',choices=['detailed512','detailed64','concise64'],default='detailed512')
    parser.add_argument('--precision',choices=['nf4','fp16'],default='nf4')
    parser.add_argument('--image-mode',choices=['hgai_crop','pad'],default=None)
    parser.add_argument('--split-off',action='store_true')
    parser.add_argument('--alpha',type=float,default=0)
    parser.add_argument('--layers',type=int,nargs=2,default=[5,18])
    parser.add_argument('--limit',type=int,default=500)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())


if __name__=='__main__':main()
