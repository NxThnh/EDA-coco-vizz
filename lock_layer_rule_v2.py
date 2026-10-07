"""Create the user-specified v2 rule once; refuse to overwrite it."""
import hashlib
import json
from pathlib import Path

path=Path(__file__).resolve().parent/'eda_results/hgai/layer_rule_v2_lock.json'
if path.exists():
    raise FileExistsError(path)
payload={
    'version':2,
    'status':'locked_after_viewing_smolvlm2_20_image_curves',
    'input':'Mean VAR and mean image-mask sensitivity indexed by decoder layer; dev images only.',
    'normalization':'For each curve independently, clip((x-P10)/(P90-P10),0,1), where P10 and P90 are the 10th and 90th percentiles across all layers.',
    'threshold':0.5,
    'minimum_width':'max(2, ceil(0.10 * L)) layers',
    'candidate':'Maximal contiguous bands where both normalized curves are >= 0.50.',
    'selection':'Choose the qualifying band with the largest sum of min(normalized VAR, normalized sensitivity); tie goes to earliest start.',
    'failure':'Return no boundary if either curve is non-finite, P90<=P10, L<8, or no qualifying band.',
    'indexing':'Zero-based decoder layers; both endpoints inclusive.',
    'selection_data':'No CHAIR or eval used; parameters fixed by user after viewing SmolVLM2 20-image curves.'
}
digest=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
path.write_text(json.dumps({**payload,'rule_sha256':digest},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(path,digest)
