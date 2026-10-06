"""Local baseline caption/POPE inference, with an explicit NF4 feasibility condition.

No HGAI intervention is applied. Partial runs are pilots, never 500-image scores.
Use --resume only with the identical hashed config; outputs are flushed per image.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from PIL import Image, ImageOps
from hgai_cpu_audit import ROOT, PREP, OUT, sha
from lock_common_unmatched import prepare_common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition', default='dev_common_unmatched500',
                        choices=['dev_common_unmatched500', 'eval_common_unmatched500', 'eval_random500'])
    parser.add_argument('--task', choices=['chair', 'pope'], default='chair')
    parser.add_argument('--pope-variant', choices=['random', 'popular', 'adversarial'], default='random')
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--precision', choices=['nf4', 'fp16'], default='nf4')
    parser.add_argument('--split-off', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('limit must be positive')
    prepare_common()
    model_path = ROOT / 'model_assets/SmolVLM2-2.2B-Instruct'
    provenance_path = OUT / 'smol_weights_provenance.json'
    if not provenance_path.exists():
        raise FileNotFoundError('Run fetch_smol_baseline.py to fetch and verify official pinned weights.')
    weights = json.loads(provenance_path.read_text())
    for item in weights:
        path = model_path / item['file']
        if not path.exists() or path.stat().st_size != item['bytes']:
            raise ValueError('Verified local weight file missing or truncated: ' + item['file'])
    if args.task == 'chair':
        manifest = PREP / (args.partition + '.jsonl')
        rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
        if args.limit > len(rows):
            parser.error('CHAIR limit exceeds the manifest')
        rows = rows[:args.limit]
        for r in rows:
            r['prompt'] = 'Please help me describe the image in detail.'
        max_tokens = 512
    else:
        manifest = ROOT / 'eda_resources/pope_official' / ('coco_pope_' + args.pope_variant + '.json')
        questions = [json.loads(line) for line in manifest.read_text().splitlines()]
        if args.limit > len(questions):
            parser.error('POPE limit exceeds the official question set')
        # Questions remain in official order, labels and negative sampling unchanged.
        rows = [{'question_id': r['question_id'], 'image_id': int(Path(r['image']).stem.split('_')[-1]),
                 'image_path': 'coco/val2014/' + r['image'], 'prompt': r['text'], 'label': r['label']}
                for r in questions[:args.limit]]
        max_tokens = 64
    if not all((ROOT / r['image_path']).is_file() for r in rows):
        raise FileNotFoundError('Benchmark image missing')
    name = args.task + ('_' + args.partition if args.task == 'chair' else '_' + args.pope_variant)
    name += f'_{args.precision}_split_{"off" if args.split_off else "on"}_n{args.limit}'
    output = args.output or (ROOT / 'experiment_outputs/smol_baseline' / (name + '.jsonl'))
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = output.with_suffix('.config.json')
    config = {'model': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct', 'revision': weights[0]['revision'],
              'task': args.task, 'partition': args.partition if args.task == 'chair' else 'official-pope-' + args.pope_variant,
              'manifest_sha256': sha(manifest), 'weights': weights,
              'processor_config_sha256': sha(model_path / 'preprocessor_config.json'),
              'chat_template_sha256': sha(model_path / 'chat_template.json'),
              'transformers': '4.53.3', 'seed': 927, 'requested_rows': args.limit,
              'benchmark_rows': len(questions) if args.task == 'pope' else 500,
              'precision': args.precision, 'quantization': 'NF4 all supported linear layers, double quant, FP16 compute' if args.precision == 'nf4' else None,
              'image_splitting': not args.split_off, 'max_new_tokens': max_tokens,
              'do_sample': False, 'num_beams': 1, 'use_cache': True, 'intervention': 'none',
              'role': 'feasibility_pilot' if args.limit < (3000 if args.task == 'pope' else 500) else 'baseline',
              'code_sha256': sha(Path(__file__))}
    completed = []
    if output.exists():
        if not args.resume:
            raise FileExistsError('Output exists; choose a new output or explicitly --resume')
        if json.loads(config_path.read_text()) != config:
            raise ValueError('Resume configuration differs; use a versioned output')
        completed = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
        key = 'image_id' if args.task == 'chair' else 'question_id'
        if [r[key] for r in completed] != [r[key] for r in rows[:len(completed)]]:
            raise ValueError('Existing output is not a valid manifest prefix')
    else:
        if config_path.exists():
            raise FileExistsError('Config already exists without output; inspect failed run before retrying')
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    if len(completed) == len(rows):
        print('Already complete:', output, flush=True)
        return
    import torch
    import transformers
    from transformers import SmolVLMForConditionalGeneration, SmolVLMProcessor, BitsAndBytesConfig
    if transformers.__version__ != '4.53.3' or not torch.cuda.is_available():
        raise RuntimeError('Use pinned Transformers 4.53.3 and a CUDA GPU for this runner')
    torch.set_num_threads(2)
    torch.manual_seed(927)
    processor = SmolVLMProcessor.from_pretrained(model_path, local_files_only=True)
    processor.image_processor.do_image_splitting = not args.split_off
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16) if args.precision == 'nf4' else None
    print('Loading official local SmolVLM2 weights:', args.precision, flush=True)
    model = SmolVLMForConditionalGeneration.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16,
                                                  device_map={'': 0}, low_cpu_mem_usage=True,
                                                  quantization_config=quant, attn_implementation='sdpa')
    model.eval()
    print('Loaded GPU footprint GiB:', model.get_memory_footprint() / 2**30, flush=True)
    image_token = processor.tokenizer.convert_tokens_to_ids('<image>')
    start = time.perf_counter()
    with output.open('a' if completed else 'x', encoding='utf-8') as stream, torch.inference_mode():
        for index, r in enumerate(rows[len(completed):], start=len(completed)):
            with Image.open(ROOT / r['image_path']) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': r['prompt']}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=text, images=[image], return_tensors='pt')
            actual_image_tokens = int((inputs.input_ids == image_token).sum())
            inputs = {k: v.to('cuda', dtype=torch.float16) if v.is_floating_point() else v.to('cuda') for k, v in inputs.items()}
            prefix = inputs['input_ids'].shape[-1]
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); tick = time.perf_counter()
            generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                       num_beams=1, use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
            torch.cuda.synchronize()
            tokens = generated[0, prefix:]
            answer = processor.decode(tokens, skip_special_tokens=True).strip()
            result = {k: v for k, v in r.items() if k not in {'valid_captions', 'prompt'}}
            result.update({'prompt': r['prompt'], 'caption' if args.task == 'chair' else 'answer': answer,
                           'new_tokens': len(tokens), 'image_tokens': actual_image_tokens,
                           'truncated': len(tokens) == max_tokens,
                           'elapsed_seconds': time.perf_counter() - tick,
                           'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated()})
            stream.write(json.dumps(result, ensure_ascii=False) + '\n'); stream.flush()
            print('Generated', index + 1, '/', len(rows), 'tokens', len(tokens),
                  'seconds', round(result['elapsed_seconds'], 2), flush=True)
            del inputs, generated, tokens
    summary = {'status': 'complete', 'rows': len(rows), 'role': config['role'],
               'seconds_this_run': time.perf_counter() - start, 'output_sha256': sha(output),
               'config_sha256': sha(config_path), 'device': torch.cuda.get_device_name(),
               'torch': torch.__version__, 'transformers': transformers.__version__}
    output.with_suffix('.run.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
