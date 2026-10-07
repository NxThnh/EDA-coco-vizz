"""Summarize paired caption lengths at v2 bands without reading CHAIR or eval."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_alpha_probe20 import distance
from summarize_tiny_alpha_probe20_v2 import repetition

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiment_outputs/v2_alpha_probe20'
OUT = ROOT / 'eda_results/hgai'


def records(model, number, alpha, band):
    path = SOURCE / f'{model}_v2_dose_{number}.jsonl'
    metadata = json.loads(path.with_suffix('.run.json').read_text(encoding='utf-8'))
    config = json.loads(path.with_suffix('.config.json').read_text(encoding='utf-8'))
    assert metadata['status'] == 'complete' and metadata['rows'] == 20
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['output_sha256']
    assert config['model_key'] == model and config['layers_inclusive'] == list(band)
    assert abs(config['alpha'] - alpha) < 1e-12 and config['condition'] == 'detailed512'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 20 and len({row['image_id'] for row in rows}) == 20
    return rows


def main():
    def repeated_fraction(caption):
        words = caption.lower().split()
        groups = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
        return (len(groups)-len(set(groups)))/len(groups) if groups else 0.0
    output = []
    for model, band, lock in (
        ('smol', (10, 23), 'smol_alpha_grid_lock.json'),
        ('tiny', (12, 28), 'tiny_alpha_grid_lock.json'),
    ):
        grid = json.loads((OUT / lock).read_text(encoding='utf-8'))['alpha_grid']
        baseline = records(model, 0, 0, band)
        for number, alpha in enumerate(grid, 1):
            treated = records(model, number, alpha, band)
            assert [r['image_id'] for r in baseline] == [r['image_id'] for r in treated]
            ratios = [r['new_tokens'] / b['new_tokens'] for r, b in zip(treated, baseline)]
            output.append({'model': model, 'alpha': alpha,
                'mean_length_ratio': float(np.mean(ratios)),
                'median_length_ratio': float(np.median(ratios)),
                'less_than_half_count': sum(r < .5 for r in ratios),
                'identical_caption_count': sum(r['caption'] == b['caption'] for r, b in zip(treated, baseline)),
                'mean_word_edit_distance': float(np.mean([distance(b['caption'], r['caption'])
                    for r, b in zip(treated, baseline)])),
                'ceiling_512_count': sum(r['new_tokens'] >= 512 for r in treated),
                'repetition_4gram_count': sum(repetition(r['caption']) for r in treated),
                'mean_repeated_4gram_fraction': float(np.mean([repeated_fraction(r['caption'])
                    for r in treated]))})
    pd.DataFrame(output).to_csv(OUT / 'v2_alpha_probe20_summary.csv', index=False,
                                encoding='utf-8-sig')


if __name__ == '__main__':
    main()
