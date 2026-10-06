"""Fetch pinned public TinyLLaVA/LLaVA checkpoints and verify every LFS SHA-256."""
import argparse
import json
import time
from pathlib import Path

import requests
from fetch_hgai_assets import ROOT, get
from hgai_cpu_audit import OUT, sha

MODELS = {
    'tiny': ('bczhou/TinyLLaVA-3.1B', '71e73b5bef64b9e9b10028750d8c9494e904adb2'),
    'llava': ('llava-hf/llava-1.5-7b-hf', 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9'),
}


def fetch(key):
    repo, revision = MODELS[key]
    folder = ROOT / 'model_assets' / repo.split('/')[-1]
    folder.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    info = get(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true').json()
    assert info['sha'] == revision
    records = []
    for item in info['siblings']:
        name = item['rfilename']
        if '/' in name or name in {'training_args.bin', '.gitattributes'}:
            continue
        if not name.endswith(('.json', '.txt', '.model', '.jinja', '.safetensors')):
            continue
        target = folder / name
        if not name.endswith('.safetensors'):
            data = get(f'https://huggingface.co/{repo}/resolve/{revision}/{name}').content
            target.write_bytes(data)
            continue
        expected, size = item['lfs']['sha256'], item['lfs']['size']
        if not (target.exists() and target.stat().st_size == size and sha(target) == expected):
            partial = target.with_suffix(target.suffix + '.part')
            url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true'
            failures = 0
            while True:
                start = partial.stat().st_size if partial.exists() else 0
                if start == size:
                    break
                if start > size:
                    raise ValueError('Oversized partial download: ' + str(partial))
                end = min(size-1, start+32*1024**2-1)
                try:
                    with requests.get(url+f'&range_start={start}', headers={'Range': f'bytes={start}-{end}'},
                                      stream=True, timeout=(30, 90)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise ValueError('Server ignored byte range; refusing to append.')
                        with partial.open('ab' if start else 'wb') as stream:
                            for block in response.iter_content(1024 * 1024):
                                if block:
                                    stream.write(block)
                    failures = 0
                    if start//(256*1024**2) != (end+1)//(256*1024**2) or end+1==size:
                        print(key, name, round(100 * (end+1) / size, 1), '%', flush=True)
                except requests.RequestException as error:
                    failures += 1
                    if failures >= 20:raise
                    print('Retry', key, name, type(error).__name__, flush=True)
                    time.sleep(2)
            if not partial.exists() or partial.stat().st_size != size or sha(partial) != expected:
                raise ValueError('Incomplete/corrupt official weight file: ' + name)
            partial.replace(target)
        records.append({'repo': repo, 'revision': revision, 'file': name, 'bytes': size, 'sha256': expected})
        print('Verified', key, name, flush=True)
    if not records:
        raise ValueError('No official weights found')
    (OUT / (key + '_weights_provenance.json')).write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('Complete', key, sum(r['bytes'] for r in records), 'bytes', flush=True)


if __name__ == '__main__':
    from filelock import FileLock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=[*MODELS, 'all'], default='all')
    args = parser.parse_args()
    runtime=ROOT/'scratch/hgai_runtime'
    runtime.mkdir(parents=True,exist_ok=True)
    with FileLock(str(runtime/'checkpoint_download.lock'),timeout=0):
        for key in MODELS if args.model == 'all' else [args.model]:
            fetch(key)
