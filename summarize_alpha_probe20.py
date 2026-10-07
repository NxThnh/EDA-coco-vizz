"""Summarize the locked, paired 20-image dev alpha probe without CHAIR scores."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiment_outputs/alpha_probe20'
DEST = ROOT / 'eda_results/hgai'
MODES = ('split_on', 'split_768', 'split_off')
ALPHAS = (0.02, 0.035, 0.05, 0.075, 0.10)


def distance(a, b):
    """Word-level Levenshtein distance, preserving case and punctuation."""
    left, right = a.split(), b.split()
    previous = list(range(len(right) + 1))
    for i, word in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != other)))
        previous = current
    return previous[-1]


def rows(mode, alpha):
    path = SOURCE / f'smol_{mode}_alpha_{alpha:.3f}.jsonl'
    run = path.with_suffix('.run.json')
    assert run.is_file(), path
    log = json.loads(run.read_text(encoding='utf-8'))
    assert log['status'] == 'complete' and log['output_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest(), path
    config = json.loads(path.with_suffix('.config.json').read_text(encoding='utf-8'))
    assert config['alpha'] == alpha and config['requested_rows'] == 20
    assert config['max_new_tokens'] == 512 and config['layers_inclusive'] == [5, 18]
    assert config['smol_longest_edge'] == (768 if mode == 'split_768' else None)
    assert config['image_splitting'] == (mode != 'split_off')
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert len(records) == 20, path
    assert len({record['image_id'] for record in records}) == 20, path
    return records


def main():
    summary = []
    comparisons = []
    identical_everywhere = set(ALPHAS)
    source_hashes = {}
    for mode in MODES:
        baseline = rows(mode, 0)
        baseline_path = SOURCE / f'smol_{mode}_alpha_0.000.jsonl'
        source_hashes[baseline_path.name] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        ids = [record['image_id'] for record in baseline]
        for alpha in ALPHAS:
            treated = rows(mode, alpha)
            assert [record['image_id'] for record in treated] == ids
            source = SOURCE / f'smol_{mode}_alpha_{alpha:.3f}.jsonl'
            source_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
            pair = []
            for original, changed in zip(baseline, treated):
                assert original['new_tokens'] > 0
                identical = original['caption'] == changed['caption']
                rec = {'mode': mode, 'alpha': alpha, 'image_id': original['image_id'],
                       'baseline_tokens': original['new_tokens'],
                       'treated_tokens': changed['new_tokens'],
                       'length_ratio': changed['new_tokens'] / original['new_tokens'],
                       'less_than_half': changed['new_tokens'] < original['new_tokens'] / 2,
                       'identical_caption': identical,
                       'word_edit_distance': distance(original['caption'], changed['caption'])}
                pair.append(rec)
                comparisons.append(rec)
            if not all(rec['identical_caption'] for rec in pair):
                identical_everywhere.discard(alpha)
            summary.append({'mode': mode, 'alpha': alpha, 'images': len(pair),
                            'mean_length_ratio': sum(rec['length_ratio'] for rec in pair) / len(pair),
                            'less_than_half_count': sum(rec['less_than_half'] for rec in pair),
                            'identical_caption_count': sum(rec['identical_caption'] for rec in pair),
                            'mean_word_edit_distance': sum(rec['word_edit_distance'] for rec in pair) / len(pair)})
    pd.DataFrame(comparisons).to_csv(DEST / 'alpha_probe20_pairs.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(DEST / 'alpha_probe20_summary.csv', index=False, encoding='utf-8-sig')
    kept = [alpha for alpha in ALPHAS if alpha not in identical_everywhere]
    assert kept
    lock = {'status': 'alpha_grid_locked_before_dev500',
            'model': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
            'partition': 'dev_common_unmatched500',
            'probe_images': 20,
            'probe_layers_inclusive': [5, 18],
            'processor_modes': list(MODES),
            'probe_measure': 'Paired generated token ratio, strict below-half count, exact caption equality, and whitespace-word Levenshtein distance; no CHAIR.',
            'baseline_alpha': 0,
            'alpha_grid': kept,
            'selection_rationale': (f'{len(kept)} of {len(ALPHAS)} candidate doses retained; only doses whose captions equal baseline on all 20 images in all three modes are removed. Frozen dev500 guards make later exclusions.'),
            'probe_path': 'eda_results/hgai/alpha_probe20_summary.csv',
            'source_sha256': source_hashes,
            'not_a_benchmark': True}
    (DEST / 'smol_alpha_grid_lock.json').write_text(json.dumps(lock, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    config_path = DEST / 'experiment_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config['small_model_tuning']['smol_alpha_grid'] = kept
    config['small_model_tuning']['alpha_probe_status'] = '20 paired dev images, three image-processor modes, detailed512; see alpha_probe20_summary.csv'
    config['implementation_status']['HGAI_and_tuning'] = ('Adapter semantics/GQA/cache validated on CPU and Smol real weights. '
        'The 20-image alpha probe locks doses before dev500; dev CHAIR, recall/F1, and length guards remain separate.')
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    design_path = DEST / 'report_pending_design.json'
    design = json.loads(design_path.read_text(encoding='utf-8'))
    design['alpha_probe']['status'] = 'complete'
    design['alpha_probe']['current_lock_unchanged'] = False
    design_path.write_text(json.dumps(design, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('SUMMARY_COMPLETE', len(summary), 'mode-dose rows; locked alpha grid:', kept)


if __name__ == '__main__':
    main()
