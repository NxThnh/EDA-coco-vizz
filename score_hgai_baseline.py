"""Score genuine caption/question outputs; preserve CHAIR lexical rules and POPE parser.

CHAIR's Python-2 print syntax is ported at import time. Its normalization and
formula are unchanged; only file IO is adapted to JSONL and val2014-only images.
"""
import argparse
import ast
import json
import re
import warnings
from pathlib import Path

import nltk
import numpy as np

from hgai_cpu_audit import ROOT, OUT, PREP, sha


def chair_class():
    folder = ROOT / 'eda_resources/chair_official'
    normalizer = json.loads((OUT / 'chair_normalizer_provenance.json').read_text())
    assert sha(folder / 'pattern_inflect.py') == normalizer['sha256']
    assert sha(folder / 'synonyms.txt') == normalizer['synonyms_sha256']
    sources = json.loads((OUT / 'metrics_provenance.json').read_text())
    chair_source = next(r for r in sources if r['source_path'] == 'utils/chair.py')
    assert sha(folder / 'chair.py') == chair_source['sha256']
    pattern_tree = ast.parse((folder / 'pattern_inflect.py').read_text(encoding='utf-8'))
    selected = []
    for node in pattern_tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                (t.id.startswith('singular_') or t.id == 'plural_prepositions') for t in node.targets):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == 'singularize':
            selected.append(node)
    namespace = {'re': re, 'NOUN': 'NN'}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(folder / 'pattern_inflect.py'), 'exec'), namespace)
    assert namespace['singularize']('women') == 'woman'
    assert namespace['singularize']('buses') == 'bus'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        from lib2to3.refactor import RefactoringTool
    source = str(RefactoringTool(['lib2to3.fixes.fix_print']).refactor_string(
        (folder / 'chair.py').read_text(encoding='utf-8') + '\n', 'official_chair.py'))
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CHAIR')
    for node in ast.walk(cls):
        if isinstance(node, ast.Constant) and node.value == 'data/synonyms.txt':
            node.value = str(folder / 'synonyms.txt')
    context = {'nltk': nltk, 'singularize': namespace['singularize']}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(folder / 'chair.py'), 'exec'), context)
    return context['CHAIR']


def score_chair(rows):
    nltk.data.path.insert(0, str(ROOT / 'model_assets/nltk_data'))
    ids = [r['image_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate image outputs cannot be scored as independent captions.')
    evaluator = chair_class()(ids, str(ROOT / 'coco/annotations'))
    # IDs in these runs are exclusively val2014, disjoint from train2014.
    # Restrict IO to val annotations: the ground-truth union per selected ID is identical.
    data = json.loads((ROOT / 'coco/annotations/instances_val2014.json').read_text())
    names = {r['id']: r['name'] for r in data['categories']}
    evaluator.imid_to_objects = {i: set() for i in ids}
    for r in data['annotations']:
        if r['image_id'] in evaluator.imid_to_objects:
            evaluator.imid_to_objects[r['image_id']].add(evaluator.inverse_synonym_dict[names[r['category_id']]])
    del data
    caps = json.loads((ROOT / 'coco/annotations/captions_val2014.json').read_text())
    for r in caps['annotations']:
        if r['image_id'] in evaluator.imid_to_objects:
            evaluator.imid_to_objects[r['image_id']].update(evaluator.caption_to_words(r['caption'])[1])
    del caps
    irrelevant = {name: 0 for name in ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'CIDEr', 'SPICE', 'ROUGE_L']}
    evaluator.caps = [{'image_id': r['image_id'], 'caption': r['caption'], **irrelevant} for r in rows]
    evaluator.metrics = irrelevant
    evaluator._load_generated_captions_into_evaluator = lambda path: None
    recognized = sum(len(evaluator.caption_to_words(r['caption'])[1]) for r in rows)
    if recognized:
        sentences = evaluator.compute_chair('')['sentences']
    else:
        # Original formula has 0/0 for CHAIR-I. Keep it undefined, CHAIR-S=0.
        sentences = [{'image_id': r['image_id'], 'caption': r['caption'],
                      'mscoco_gt_words': sorted(evaluator.imid_to_objects[r['image_id']]),
                      'mscoco_generated_words': [], 'mscoco_hallucinated_words': [],
                      'hallucination_idxs': [], 'words': evaluator.caption_to_words(r['caption'])[3],
                      'metrics': {'CHAIRs': 0, 'CHAIRi': 0}} for r in rows]
    for sentence in sentences:
        sentence['metrics'] = {k: v for k, v in sentence['metrics'].items() if k in {'CHAIRs', 'CHAIRi'}}
    hallucinated_caps = sum(s['metrics']['CHAIRs'] for s in sentences)
    hallucinated_words = sum(len(s['mscoco_hallucinated_words']) for s in sentences)
    generated_words = sum(len(s['mscoco_generated_words']) for s in sentences)
    gt = sum(len(s['mscoco_gt_words']) for s in sentences)
    true = sum(len(set(s['mscoco_generated_words']) & set(s['mscoco_gt_words'])) for s in sentences)
    predicted = sum(len(set(s['mscoco_generated_words'])) for s in sentences)
    recall = true / gt if gt else None
    precision = true / predicted if predicted else None
    f1 = 2 * recall * precision / (recall + precision) if recall is not None and precision is not None and recall + precision else 0
    # Original evaluator divides by zero on object-free outputs; expose undefined CHAIR-I explicitly.
    result = {'images': len(rows), 'hallucinated_captions': hallucinated_caps,
              'hallucinated_object_mentions': hallucinated_words, 'generated_object_mentions': generated_words,
              'CHAIR_S_pct': 100 * hallucinated_caps / len(rows),
              'CHAIR_I_pct': 100 * hallucinated_words / generated_words if generated_words else None,
              'object_micro_recall': recall, 'object_micro_precision': precision, 'object_micro_F1': f1,
              'mean_new_tokens': float(np.mean([r['new_tokens'] for r in rows])),
              'truncation_rate': sum(r['truncated'] for r in rows) / len(rows),
              'mean_seconds': float(np.mean([r['elapsed_seconds'] for r in rows])),
              'peak_gpu_GiB': max(r['peak_allocated_gpu_bytes'] for r in rows) / 2**30}
    return result, sentences


def parse_pope(answer):
    # Exact normalization from RUCAIBox/POPE evaluate.py, including case behavior.
    text = answer.split('.')[0].replace(',', '')
    words = text.split(' ')
    return int(not ('No' in words or 'not' in words or 'no' in words))


def score_pope(rows):
    pred = [parse_pope(r['answer']) for r in rows]
    labels = [int(r['label'] == 'yes') for r in rows]
    tp = sum(p == 1 and y == 1 for p, y in zip(pred, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(pred, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(pred, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(pred, labels))
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {'questions': len(rows), 'images': len({r['image_id'] for r in rows}),
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'accuracy': (tp + tn) / len(rows),
            'precision': precision, 'recall': recall,
            'F1': 2 * precision * recall / (precision + recall) if precision + recall else 0,
            'yes_rate': sum(pred) / len(pred)}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    path = args.output
    config = json.loads(path.with_suffix('.config.json').read_text())
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if len(rows) != config['requested_rows']:
        raise ValueError('Generation is incomplete; do not publish a completed-run score.')
    if not rows:
        raise ValueError('Cannot score an empty run.')
    if config['task'] == 'chair':
        from lock_common_unmatched import prepare_common
        prepare_common()
        manifest = PREP / (config['partition'] + '.jsonl')
        original = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
        if [r['image_id'] for r in rows] != [r['image_id'] for r in original[:len(rows)]]:
            raise ValueError('Caption output IDs differ from locked manifest prefix.')
    else:
        variant = config['partition'].removeprefix('official-pope-')
        if variant not in {'random', 'popular', 'adversarial'}:
            raise ValueError('Unknown official POPE variant.')
        manifest = ROOT / 'eda_resources/pope_official' / ('coco_pope_' + variant + '.json')
        original = [json.loads(line) for line in manifest.read_text().splitlines()]
        for generated, expected in zip(rows, original):
            if (generated['question_id'] != expected['question_id'] or generated['label'] != expected['label']
                    or generated['prompt'] != expected['text']
                    or generated['image_id'] != int(Path(expected['image']).stem.split('_')[-1])):
                raise ValueError('POPE output misaligned with official question, image or label.')
    if sha(manifest) != config['manifest_sha256']:
        raise ValueError('Evaluation manifest changed after generation.')
    if config['task'] == 'pope':
        result, sentences = score_pope(rows)
    else:
        result, sentences = score_chair(rows)
    result.update({'role': config['role'], 'model': config['model'], 'task': config['task'],
                   'partition': config['partition'], 'model_precision': config['precision'],
                   'image_splitting': config['image_splitting'], 'max_new_tokens': config['max_new_tokens'],
                   'output_sha256': sha(path), 'config_sha256': sha(path.with_suffix('.config.json')),
                   'evaluator_code_sha256': sha(Path(__file__)),
                   'metric_sources': json.loads((OUT / 'metrics_provenance.json').read_text()),
                   'normalizer': json.loads((OUT / 'chair_normalizer_provenance.json').read_text()) if config['task'] == 'chair' else 'official POPE evaluate.py parser',
                   'note': 'A pilot subset is not a full-benchmark score; NF4 is a separate precision condition.'})
    path.with_suffix('.metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if sentences is not None:
        path.with_suffix('.chair_details.json').write_text(json.dumps(sentences, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
