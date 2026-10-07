"""Freeze the observed first-to-last CSV write windows for 100-image curves."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'eda_results/hgai'
payload = {'scope': 'first CSV write to last CSV write; excludes model loading, setup and bootstrap',
           'models': {}}
for model, filename in [('smol', 'layer_curves100_smol_per_image.csv'),
                        ('tiny', 'layer_curves100_tiny_sdpa_per_image.csv')]:
    path = OUT / filename
    stat = path.stat()
    payload['models'][model] = {
        'source': filename,
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'first_write_utc': datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(),
        'last_write_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        'minutes': (stat.st_mtime - stat.st_ctime) / 60}
target = OUT / 'curve_measure_windows.json'
if target.exists():
    raise FileExistsError(target)
target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(target)
