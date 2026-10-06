"""Fetch official pinned SmolVLM2 weights, resuming partial files and verifying LFS SHA256."""
import json
import time

import requests

from fetch_hgai_assets import get, REVISIONS, ROOT
from hgai_cpu_audit import sha, OUT


def main():
    repo = 'HuggingFaceTB/SmolVLM2-2.2B-Instruct'
    revision = REVISIONS[repo]
    folder = ROOT / 'model_assets/SmolVLM2-2.2B-Instruct'
    info = get(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true').json()
    (folder / 'weights_api.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    records = []
    for item in info['siblings']:
        name = item['rfilename']
        if name in {'model.safetensors.index.json', 'generation_config.json'}:
            data = get(f'https://huggingface.co/{repo}/raw/{revision}/{name}').content
            (folder / name).write_bytes(data)
        if not name.endswith('.safetensors'):
            continue
        expected = item['lfs']['sha256']; size = item['lfs']['size']
        target = folder / name
        if target.exists() and target.stat().st_size == size and sha(target) == expected:
            print('Verified cached weights:', name, flush=True)
        else:
            partial = folder / (name + '.part')
            url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true'
            for attempt in range(12):
                start = partial.stat().st_size if partial.exists() else 0
                if start == size:
                    break
                headers = {'Range': f'bytes={start}-'} if start else {}
                try:
                    with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as response:
                        response.raise_for_status()
                        if start and (response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-')):
                            raise ValueError('Server did not honor resume range; refusing to append.')
                        with partial.open('ab' if start else 'wb') as f:
                            last = start
                            for block in response.iter_content(1024 * 1024):
                                if block:
                                    f.write(block)
                                if f.tell() - last >= 128 * 1024 * 1024:
                                    print(name, round(100 * f.tell() / size, 1), '%', flush=True)
                                    last = f.tell()
                    break
                except requests.RequestException as error:
                    print('Retry weight download:', attempt + 1, type(error).__name__, flush=True)
                    time.sleep(2)
            if partial.stat().st_size != size or sha(partial) != expected:
                raise ValueError('Weight download incomplete or hash mismatch: ' + name)
            partial.replace(target)
        records.append({'repo': repo, 'revision': revision, 'file': name, 'bytes': size, 'sha256': expected})
    if not records:
        raise ValueError('No safetensors files found in official checkpoint.')
    (OUT / 'smol_weights_provenance.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    tree = get('https://api.github.com/repos/huggingface/smollm/git/trees/main?recursive=1').json()
    (ROOT / 'scratch/smollm_github_tree.json').write_text(json.dumps(tree), encoding='utf-8')
    print('Official weights verified:', sum(r['bytes'] for r in records), 'bytes', flush=True)


if __name__ == '__main__':
    main()
