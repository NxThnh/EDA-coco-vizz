"""Descriptive layer diagnostics from measured dev curves; never changes rule locks."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'eda_results/hgai'


def main():
    records = []
    overlaps = []
    for key, label, depth, selected, scaled in (
        ('smol', 'SmolVLM2-2.2B', 24, (10, 23), (3, 14)),
        ('tiny', 'bczhou/TinyLLaVA-3.1B', 32, (12, 28), (5, 18)),
    ):
        path = OUT / f'layer_curves100_{key}_summary.csv'
        if key == 'tiny':
            path = OUT / 'layer_curves100_tiny_sdpa_summary.csv'
        data = pd.read_csv(path).sort_values('layer')
        assert data.layer.tolist() == list(range(depth))
        increase = data.mask_sensitivity.diff()
        records.extend({'model': label, 'layer': int(layer),
                        'mask_sensitivity_increment': float(value)}
                       for layer, value in zip(data.layer[1:], increase[1:]))
        selected_set = set(range(selected[0], selected[1] + 1))
        for name, band in [('5–18 tuyệt đối', (5, 18)), ('co theo độ sâu', scaled)]:
            common = sorted(selected_set & set(range(band[0], band[1] + 1)))
            overlaps.append({'Mô hình': label, 'Dải v2': f'{selected[0]}–{selected[1]}',
                             'Đối chứng': name, 'Dải đối chứng': f'{band[0]}–{band[1]}',
                             'Số layer chung': len(common),
                             'Layer chung': ', '.join(map(str, common))})
    pd.DataFrame(records).to_csv(OUT / 'layer_mask_increment.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(overlaps).to_csv(OUT / 'report_layer_band_overlap.csv', index=False, encoding='utf-8-sig')
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False, layout='constrained')
    for axis, (key, label) in zip(axes, [('smol', 'SmolVLM2-2.2B'), ('tiny', 'bczhou/TinyLLaVA-3.1B')]):
        part = pd.DataFrame(records).query('model == @label')
        axis.bar(part.layer, part.mask_sensitivity_increment, color='#3377aa')
        axis.axhline(0, color='black', linewidth=.7)
        axis.set_title(label)
        axis.set_ylabel('Mức tăng độ nhạy')
        axis.set_xlabel('Layer hiện tại (so với layer trước)')
    figure.savefig(OUT / 'layer_mask_increment.png', dpi=170)
    plt.close(figure)


if __name__ == '__main__':
    main()
