"""Fetch public annotation JSONs (no images/weights) for model-specific leakage checks."""
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import requests

ROOT=Path(__file__).resolve().parent


def main():
    info=json.loads((ROOT/'scratch/ShareGPT4V_dataset_api.json').read_text())
    names=[r['rfilename'] for r in info['siblings'] if r['rfilename'].endswith('.json')]
    records=[]
    for name in names:
        url=f"https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/{info['sha']}/{name}"
        path=ROOT/'scratch'/name
        if not path.exists():
            # Persist only a completed HTTP response; retries cannot leave a false complete file.
            temp=path.with_suffix('.download')
            for attempt in range(4):
                try:
                    with requests.get(url,stream=True,timeout=180) as response:
                        response.raise_for_status()
                        with temp.open('wb') as f:
                            for chunk in response.iter_content(1024*1024):
                                f.write(chunk)
                    temp.replace(path)
                    break
                except requests.RequestException:
                    if attempt==3:
                        raise
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        records.append({'repo':'Lin-Chen/ShareGPT4V','revision':info['sha'],'url':url,
                        'path':str(path.relative_to(ROOT)).replace('\\','/'),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
        print('Downloaded annotation:',name,path.stat().st_size,flush=True)
        (ROOT/'eda_results/hgai/training_metadata_provenance.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    # Obtain filenames/IDs without downloading the embedded images in parquet.
    one=json.loads((ROOT/'scratch/LLaVA-OneVision-Data_dataset_api.json').read_text())
    url=f"https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data/raw/{one['sha']}/README.md"
    r=requests.get(url,timeout=90);r.raise_for_status()
    (ROOT/'scratch/OneVision_README.md').write_text(r.text,encoding='utf-8')


if __name__=='__main__':main()
