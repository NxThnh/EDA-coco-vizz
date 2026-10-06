"""CPU-only HGAI EDA: locked COCO splits, processors, caption tokens and geometry.

Run with .venv-eda/Scripts/python.exe hgai_cpu_audit.py after fetching CPU assets.
No language or vision model weights are loaded. Existing EDA manifests are retained.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
OUT = ROOT / 'eda_results/hgai'
PREP = ROOT / 'data_prepared/hgai'
EXT = ROOT / 'eda_results/vlm_extensions'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def csv_write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'mean': float(a.mean()), 'min': float(a.min()),
            **{f'p{q}': float(np.percentile(a, q)) for q in [50, 90, 95, 99]},
            'max': float(a.max())}


def prepare():
    """Lock once. Re-running validates source hashes; it never silently resamples."""
    OUT.mkdir(parents=True, exist_ok=True)
    PREP.mkdir(parents=True, exist_ok=True)
    lock_path = PREP / 'manifest_lock.json'
    sources = [ROOT / 'coco/annotations/captions_val2014.json',
               EXT / 'llava15_vg_gqa_alias_overlap.csv',
               EXT / 'val_image_hashes.csv']
    fingerprints = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sources}
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding='utf-8'))
        if lock['source_sha256'] != fingerprints:
            raise ValueError('Source changed after HGAI manifest lock; create a versioned experiment explicitly.')
        for name, digest in lock['manifest_sha256'].items():
            if sha(PREP / name) != digest:
                raise ValueError('Locked manifest altered: ' + name)
        return lock
    caps = load('coco/annotations/captions_val2014.json')
    overlap = pd.read_csv(EXT / 'llava15_vg_gqa_alias_overlap.csv').set_index('image_id')
    hashes = pd.read_csv(EXT / 'val_image_hashes.csv')
    hashes = hashes[hashes.dataset == 'COCO'].set_index('image_id')
    ids = sorted(im['id'] for im in caps['images'] if int(hashes.loc[im['id'], 'decode_ok']) == 1)
    # Main set retains the existing random COCO500, making earlier EDA comparable.
    old = pd.read_csv(EXT / 'evaluation_manifest_seed42.csv')
    main = sorted(old.loc[old.dataset == 'COCO', 'image_id'].astype(int))
    used_hashes = {hashes.loc[i, 'md5'] for i in main}
    eligible = [i for i in ids if not overlap.loc[i, 'direct_coco_path']
                and not overlap.loc[i, 'vg_gqa_alias'] and hashes.loc[i, 'md5'] not in used_hashes]
    sensitivity = sorted(random.Random(43).sample(eligible, 500))
    used_hashes.update(hashes.loc[i, 'md5'] for i in sensitivity)
    dev_pool = [i for i in ids if hashes.loc[i, 'md5'] not in used_hashes]
    dev = sorted(random.Random(44).sample(dev_pool, 500))
    grouped = defaultdict(list)
    for ann in caps['annotations']:
        grouped[ann['image_id']].append(ann)
    names = {}
    rows_summary = []
    for partition, chosen, seed in [('eval_random500', main, 42),
                                   ('eval_unmatched500', sensitivity, 43), ('dev500', dev, 44)]:
        selected = set(chosen)
        records = []
        for image in sorted(caps['images'], key=lambda r: r['id']):
            i = image['id']
            if i not in selected:
                continue
            direct, alias = bool(overlap.loc[i, 'direct_coco_path']), bool(overlap.loc[i, 'vg_gqa_alias'])
            records.append({'dataset': 'COCO', 'partition': partition, 'image_id': i,
                            'file_name': image['file_name'], 'image_path': 'coco/val2014/' + image['file_name'],
                            'width': image['width'], 'height': image['height'],
                            'md5': hashes.loc[i, 'md5'], 'llava_direct_coco': direct,
                            'llava_vg_gqa_alias': alias, 'llava_known_match': direct or alias,
                            'valid_captions': [a['caption'] for a in sorted(grouped[i], key=lambda a: a['id'])]})
        filename = partition + '.jsonl'
        path = PREP / filename
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
        csv_write(PREP / (partition + '.csv'), [{k: v for k, v in r.items() if k != 'valid_captions'} for r in records])
        names[filename] = sha(path)
        rows_summary.append({'partition': partition, 'images': len(records), 'seed': seed,
                             'direct_coco': sum(r['llava_direct_coco'] for r in records),
                             'vg_gqa_alias': sum(r['llava_vg_gqa_alias'] for r in records),
                             'known_union': sum(r['llava_known_match'] for r in records),
                             'unmatched': sum(not r['llava_known_match'] for r in records), 'sha256': names[filename]})
    lock = {'schema_version': 1, 'created_date': '2026-10-06', 'source_sha256': fingerprints,
            'manifest_sha256': names, 'partitions': rows_summary,
            'notes': ['Main COCO500 retains prior seed42 random manifest.',
                      'Unmatched means LLaVA mix665K COCO/VG/GQA only; not clean across all models.',
                      'Dev is fixed at 500; pairwise image ID and exact-byte disjoint from both eval sets.',
                      'Remaining validation images are unused reservoir, not tuning dev.',
                      'Old COCO/VizWiz EDA manifest remains a context artifact; no VizWiz HGAI run is scheduled.']}
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding='utf-8')
    csv_write(OUT / 'manifest_summary.csv', rows_summary)
    return lock


def records():
    lock = prepare()
    filenames = set(lock['manifest_sha256'])
    if (PREP / 'manifest_lock_common_v9.json').exists():
        from lock_common_unmatched import prepare_common
        filenames.update(prepare_common()['manifest_sha256'])
    for name in sorted(filenames):
        path = PREP / name
        for line in path.read_text(encoding='utf-8').splitlines():
            yield json.loads(line)


def caption_tokens(tokenizers):
    rows = []
    for dataset, source in [('COCO', 'coco/annotations/captions_val2014.json'),
                            ('VizWiz-context', 'vizwiz/annotations/val.json')]:
        data = load(source)
        grouped = defaultdict(list)
        for ann in data['annotations']:
            if not (ann.get('is_precanned') or ann.get('is_rejected')) and ann['caption'].strip():
                grouped[ann['image_id']].append(ann)
        selected = [(i, min(anns, key=lambda r: r['id'])) for i, anns in sorted(grouped.items())]
        for model, tokenizer in tokenizers.items():
            lengths = []
            for start in range(0, len(selected), 512):
                batch = selected[start:start + 512]
                encoded = tokenizer([ann['caption'] for _, ann in batch], add_special_tokens=False,
                                    truncation=False, return_attention_mask=False)['input_ids']
                for (i, ann), tokens in zip(batch, encoded):
                    lengths.append(len(tokens))
                    rows.append({'dataset': dataset, 'model': model, 'image_id': i,
                                 'annotation_id': ann['id'], 'caption_tokens': len(tokens)})
            print('Caption tokens:', dataset, model, stats(lengths), flush=True)
    csv_write(OUT / 'caption_tokens_per_image.csv', rows)
    summaries = []
    frame = pd.DataFrame(rows)
    for (dataset, model), group in frame.groupby(['dataset', 'model']):
        summaries.append({'dataset': dataset, 'model': model, **stats(group.caption_tokens)})
    csv_write(OUT / 'caption_token_summary.csv', summaries)


def geometry():
    data = load('coco/annotations/instances_val2014.json')
    imgs = {r['id']: r for r in data['images']}
    centers = defaultdict(list)
    for ann in data['annotations']:
        image = imgs[ann['image_id']]
        x, y, w, h = ann['bbox']
        # 0=center; 1=corner, normalized by image half diagonal.
        distance = math.hypot((x + w / 2) - image['width'] / 2,
                              (y + h / 2) - image['height'] / 2) / (math.hypot(image['width'], image['height']) / 2)
        centers[ann['image_id']].append((distance, ann['area'], bool(ann['iscrowd'])))
    rows = []
    for i, image in imgs.items():
        c = centers[i]
        rows.append({'image_id': i, 'instances': len(c), 'noncrowd_instances': sum(not x[2] for x in c),
                     'mean_bbox_center_offset': float(np.mean([x[0] for x in c])) if c else None,
                     'largest_object_center_offset': max(c, key=lambda x: x[1])[0] if c else None,
                     'area_weighted_center_offset': sum(x[0]*x[1] for x in c)/sum(x[1] for x in c) if c and sum(x[1] for x in c) else None})
    csv_write(OUT / 'coco_object_centering.csv', rows)
    summary = {'images': len(rows), 'instances': sum(r['instances'] for r in rows),
               'images_without_instances': sum(r['instances'] == 0 for r in rows),
               'formula': 'Euclidean bbox-center to image-center distance / half image diagonal; crowd included and flagged.',
               'largest_object': stats([r['largest_object_center_offset'] for r in rows if r['instances']]),
               'limitation': 'Measured bbox geometry; not image quality or evidence of a CHAIR correlation. VizWiz has no bbox ground truth.'}
    (OUT / 'coco_object_centering.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def processor_audit(limit=None, missing_only=False):
    import transformers
    from transformers import AutoTokenizer, SmolVLMProcessor, SiglipImageProcessor
    import torch
    if transformers.__version__ != '4.53.3':
        raise RuntimeError('Use the pinned Transformers 4.53.3 CPU environment for this audit.')
    torch.set_num_threads(2)
    smol_path = ROOT / 'model_assets/SmolVLM2-2.2B-Instruct'
    tiny_path = ROOT / 'model_assets/TinyLLaVA-3.1B'
    smol = SmolVLMProcessor.from_pretrained(smol_path, local_files_only=True)
    tiny_tok = AutoTokenizer.from_pretrained(tiny_path, local_files_only=True, trust_remote_code=False)
    tiny_ip = SiglipImageProcessor.from_pretrained(ROOT / 'model_assets/siglip-so400m-patch14-384', local_files_only=True)
    image_id = smol.tokenizer.convert_tokens_to_ids('<image>')
    print('Loaded CPU processors:', transformers.__version__, 'image seq len', smol.image_seq_len, flush=True)
    rows = []
    selected = list(records())
    previous = None
    if missing_only and (OUT / 'processor_per_image.csv').exists():
        previous = pd.read_csv(OUT / 'processor_per_image.csv')
        complete = previous.groupby(['partition', 'image_id']).size()
        done = set(complete[complete == 3].index)
        selected = [r for r in selected if (r['partition'], r['image_id']) not in done]
    if limit:
        selected = selected[:limit]
    for index, row in enumerate(selected):
        with Image.open(ROOT / row['image_path']) as raw:
            image = ImageOps.exif_transpose(raw).convert('RGB')
        w, h = image.size
        for splitting in [False, True]:
            smol.image_processor.do_image_splitting = splitting
            encoded = smol(images=[image], text='<image>Describe the image.', return_tensors='pt')
            # Count actual expanded image placeholders, excluding boundary/grid tokens.
            visual = int((encoded['input_ids'] == image_id).sum().item())
            tiles = int(encoded['pixel_values'].shape[1])
            mask = encoded['pixel_attention_mask']
            pixel_valid_fraction = float(mask.float().mean().item())
            assert visual == tiles * smol.image_seq_len, (visual, tiles, smol.image_seq_len)
            # Pixel mask describes post-resize/tiling pad area, not lost source content.
            rows.append({'partition': row['partition'], 'image_id': row['image_id'], 'model': 'SmolVLM2-2.2B',
                         'mode': 'split_on' if splitting else 'split_off', 'width': w, 'height': h,
                         'image_tokens': visual, 'tiles_including_global': tiles,
                         'processor_input_tokens': encoded['input_ids'].shape[-1],
                         'padding_area_pct': 100*(1-pixel_valid_fraction), 'crop_area_lost_pct': 0,
                         'distortion_factor': 1, 'long_edge_to_384_ratio': max(w,h)/384,
                         'resize_longest_edge_config': smol.image_processor.size['longest_edge']})
            del encoded
        # TinyLLaVA config is pad. SigLIP processor resizes the padded square to 384.
        side = max(w,h)
        pad = Image.new('RGB', (side, side), tuple(int(v*255) for v in tiny_ip.image_mean))
        pad.paste(image, ((side-w)//2, (side-h)//2))
        encoded = tiny_ip(images=pad, return_tensors='pt')
        assert tuple(encoded.pixel_values.shape[-2:]) == (384,384)
        rows.append({'partition': row['partition'], 'image_id': row['image_id'], 'model': 'TinyLLaVA-3.1B',
                     'mode': 'pad_384', 'width': w, 'height': h, 'image_tokens': (384//14)**2,
                     'tiles_including_global': 1, 'processor_input_tokens': None,
                     'padding_area_pct': 100*(1-min(w,h)/max(w,h)), 'crop_area_lost_pct': 0,
                     'distortion_factor': 1, 'long_edge_to_384_ratio': max(w,h)/384,
                     'resize_longest_edge_config': 384})
        if index % 100 == 0:
            print('Processor audit:', index+1, '/', len(selected), flush=True)
    if limit:
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
        return
    frame = pd.DataFrame(rows)
    if previous is not None:
        replaced = {(r['partition'], r['image_id']) for r in rows}
        keep = [(p, i) not in replaced for p, i in zip(previous.partition, previous.image_id)]
        frame = pd.concat([previous.loc[keep], frame], ignore_index=True)
    frame.to_csv(OUT / 'processor_per_image.csv', index=False, encoding='utf-8-sig')
    summary, distribution = [], []
    for (partition, model, mode), group in frame.groupby(['partition', 'model', 'mode']):
        summary.append({'partition': partition, 'model': model, 'mode': mode, **stats(group.image_tokens),
                        'mean_padding_pct': group.padding_area_pct.mean(),
                        'median_long_edge_to_384': group.long_edge_to_384_ratio.median()})
        for value, count in group.image_tokens.value_counts().sort_index().items():
            distribution.append({'partition': partition, 'model': model, 'mode': mode,
                                 'image_tokens': int(value), 'images': int(count), 'pct':100*count/len(group)})
    csv_write(OUT / 'image_token_summary.csv', summary)
    csv_write(OUT / 'image_token_distribution.csv', distribution)
    smol.image_processor.do_image_splitting = True
    examples = []
    for w,h in [(640,480),(640,640),(384,384),(23,30)]:
        for splitting in [False,True]:
            smol.image_processor.do_image_splitting = splitting
            e=smol(images=[Image.new('RGB',(w,h))],text='<image>Describe the image.',return_tensors='pt')
            examples.append({'width':w,'height':h,'splitting':splitting,
                             'image_tokens':int((e.input_ids==image_id).sum()),'tiles':int(e.pixel_values.shape[1])})
    csv_write(OUT/'image_token_examples.csv',examples)
    if not missing_only or not (OUT / 'caption_tokens_per_image.csv').exists():
        caption_tokens({'SmolVLM2-2.2B': smol.tokenizer, 'TinyLLaVA-3.1B': tiny_tok})
    environment = {'transformers': transformers.__version__, 'torch':torch.__version__,
                   'image_seq_len': smol.image_seq_len, 'smol_image_token_id': image_id,
                   'smol_processor': smol.image_processor.to_dict(), 'tiny_processor':tiny_ip.to_dict(),
                   'scope': 'Actual CPU processor on all locked manifests. No model forward pass.',
                   'source_code_sha256':sha(Path(__file__)), 'manifest_lock_sha256':sha(PREP/'manifest_lock.json'),
                   'processor_images': int(len(frame) / 3),
                   'incremental_run_images':len(selected) if missing_only else None}
    if (PREP / 'manifest_lock_common_v9.json').exists():
        environment['common_manifest_lock_sha256'] = sha(PREP / 'manifest_lock_common_v9.json')
    (OUT/'processor_environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    from hgai_geometry_finalize import main as finalize_geometry
    finalize_geometry()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', type=int)
    parser.add_argument('--missing-only', action='store_true', help='Process newly locked images; retain previous processor rows')
    args = parser.parse_args()
    lock = prepare()
    if (PREP / 'manifest_lock_common_v9.json').exists():
        from lock_common_unmatched import prepare_common
        common = prepare_common()
        lock = {**lock, 'partitions': [*lock['partitions'], *common['partitions']]}
    print('Locked partitions:', [(p['partition'],p['images']) for p in lock['partitions']], flush=True)
    geometry()
    if not args.prepare_only:
        processor_audit(args.smoke, args.missing_only)
