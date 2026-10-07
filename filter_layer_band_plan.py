"""Remove TinyLLaVA bands touching 29–31 only after the five-image adapter check."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'eda_results/hgai'
check = json.loads((OUT / 'tiny_last3_adapter5_validation.json').read_text(encoding='utf-8'))
assert check['images'] == 5 and check['layers_inclusive'] == [29, 31] and check['alpha'] == .02
assert check['nonfinite_images'] > 0
path = OUT / 'layer_band_scan_plan.csv'
frame = pd.read_csv(path)
assert len(frame) in (71, 65)
filtered = frame[~((frame.model == 'tiny') & (frame.end_inclusive >= 29))]
assert filtered.groupby(['model', 'width_kind']).size().to_dict() == {
    ('smol', 'quarter_depth'): 19, ('smol', 'v2_width'): 11,
    ('tiny', 'quarter_depth'): 22, ('tiny', 'v2_width'): 13}
filtered.to_csv(path, index=False, encoding='utf-8-sig')
print('BANDS', len(filtered))
