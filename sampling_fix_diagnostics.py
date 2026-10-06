"""Measured selection shifts and evidence for the published SmolVLM2 recipe."""
import json
import re

import pandas as pd
import yaml

from hgai_cpu_audit import ROOT, OUT, PREP, sha, csv_write
from lock_common_unmatched import prepare_common


def main():
    lock = prepare_common()
    source = ROOT / 'eda_resources/smol_training/onevision_less_mammoth_more_videos_balanced.yaml'
    entries = yaml.safe_load(source.read_text())['datasets']
    recipe = [{'name': r['name'], 'modality': r['modality'],
               'sampling_strategy': r['sampling_strategy'],
               'local_json_path': r['json_path'],
               'source_rows_comment': int(re.search(r'(\d+) samples', r['_comment']).group(1))}
              for r in entries]
    csv_write(OUT / 'smol_published_recipe.csv', recipe)
    provenance = json.loads((ROOT / 'eda_resources/smol_training/revision.json').read_text())
    evidence = {'repo': 'huggingface/smollm', 'revision': provenance[0]['revision'],
                'mixture_entries': len(entries), 'recipe_sha256': sha(source),
                'training_script': 'vision/smolvlm2/scripts/train/multinode25.sh',
                'exact_training_manifest_status': 'not_verified',
                'evidence': ['Published launch script selects this mixture YAML.',
                             'YAML specifies source pools and percentages, including ShareGPT4V COCO and LLaVA at 100%.',
                             'Source JSON paths are author-local /fsx paths; no exact selected ID list accompanies this YAML.',
                             'A published recipe is not a checkpoint-specific complete exposure manifest.'],
                'remaining_unknown': ['Selected IDs and private/modified source JSON contents',
                                      'Checkpoint linkage to exact training run',
                                      'Inherited SmolVLM training and vision/language pretraining exposure']}
    (OUT / 'smol_training_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    processor = pd.read_csv(OUT / 'processor_per_image.csv')
    captions = pd.read_csv(OUT / 'caption_tokens_per_image.csv')
    center = pd.read_csv(OUT / 'coco_object_centering.csv').set_index('image_id')
    diagnostics = []
    for name in ['eval_random500', 'eval_unmatched500', 'dev500',
                 'eval_common_unmatched500', 'dev_common_unmatched500']:
        rows = [json.loads(line) for line in (PREP / (name + '.jsonl')).read_text(encoding='utf-8').splitlines()]
        frame = pd.DataFrame(rows)
        smol = processor[(processor.partition == name) & (processor['mode'] == 'split_on')]
        tiny = processor[(processor.partition == name) & (processor['mode'] == 'pad_384')]
        token = captions[(captions.dataset == 'COCO') & (captions.model == 'SmolVLM2-2.2B') & captions.image_id.isin(frame.image_id)]
        diagnostics.append({'partition': name, 'images': len(frame),
                            'portrait_pct': 100 * (frame.height > frame.width).mean(),
                            'square_pct': 100 * (frame.height == frame.width).mean(),
                            'mean_tiny_pad_pct': tiny.padding_area_pct.mean(),
                            'mean_smol_image_tokens_split_on': smol.image_tokens.mean(),
                            'median_reference_caption_tokens_smol': token.caption_tokens.median(),
                            'median_largest_object_center_offset': center.loc[frame.image_id].largest_object_center_offset.median(),
                            'images_without_instances': int(center.loc[frame.image_id].instances.eq(0).sum())})
    csv_write(OUT / 'sampling_geometry_shift.csv', diagnostics)
    data = json.loads((ROOT / 'coco/annotations/instances_val2014.json').read_text())
    image_classes = {}
    for row in data['annotations']:
        image_classes.setdefault(row['image_id'], set()).add(row['category_id'])
    classes = []
    for cat in data['categories']:
        row = {'category': cat['name']}
        for name in ['eval_random500', 'eval_common_unmatched500', 'dev_common_unmatched500']:
            ids = [json.loads(line)['image_id'] for line in (PREP / (name + '.jsonl')).read_text(encoding='utf-8').splitlines()]
            row[name + '_images'] = sum(cat['id'] in image_classes.get(i, set()) for i in ids)
            row[name + '_pct'] = 100 * row[name + '_images'] / len(ids)
        row['common_minus_random_pp'] = row['eval_common_unmatched500_pct'] - row['eval_random500_pct']
        classes.append(row)
    csv_write(OUT / 'sampling_object_distribution.csv', classes)
    print('Sampling fix verified:', lock['total_common_unmatched'], 'unmatched pool;', len(entries), 'published Smol mixture entries')


if __name__ == '__main__':
    main()
