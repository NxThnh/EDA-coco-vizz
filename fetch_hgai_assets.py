"""Download public CPU audit assets only; no model weights or remote code execution."""
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
LOG = ROOT / 'eda_results/hgai/assets_provenance.json'
REVISIONS = {
    'HuggingFaceTB/SmolVLM2-2.2B-Instruct': '482adb537c021c86670beed01cd58990d01e72e4',
    'bczhou/TinyLLaVA-3.1B': '71e73b5bef64b9e9b10028750d8c9494e904adb2',
    'bczhou/TinyLLaVA-3.1B-SigLIP': '83a94bb0142ad7b333839646cba8fbb8ea78a487',
    'google/siglip-so400m-patch14-384': '9fdffc58afc957d1a03a25b10dba0329ab15c2a3',
}
ALLOWED = {'config.json', 'preprocessor_config.json', 'processor_config.json',
           'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
           'added_tokens.json', 'vocab.json', 'merges.txt', 'chat_template.json',
           'chat_template.jinja', 'README.md'}


def get(url):
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2)


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / 'scratch').mkdir(parents=True, exist_ok=True)
    records = []
    repos = ['HuggingFaceTB/SmolVLM2-2.2B-Instruct', 'bczhou/TinyLLaVA-3.1B',
             'bczhou/TinyLLaVA-3.1B-SigLIP', 'google/siglip-so400m-patch14-384']
    for repo in repos:
        info = get('https://huggingface.co/api/models/' + repo + '/revision/' + REVISIONS[repo]).json()
        revision = info['sha']
        folder = ROOT / 'model_assets' / repo.split('/')[-1]
        folder.mkdir(parents=True, exist_ok=True)
        for item in info['siblings']:
            name = item['rfilename']
            if name not in ALLOWED:
                continue
            url = f'https://huggingface.co/{repo}/raw/{revision}/{name}'
            data = get(url).content
            (folder / name).write_bytes(data)
            records.append({'repo': repo, 'revision': revision, 'url': url,
                            'path': str((folder / name).relative_to(ROOT)).replace('\\', '/'),
                            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        print('Fetched CPU assets:', repo, revision, flush=True)
    # Source snapshots allow later source inspection without relying on a mutable web page.
    LOG.write_text(json.dumps(records, indent=2), encoding='utf-8')
    for repo in ['Lin-Chen/ShareGPT4V', 'lmms-lab/LLaVA-OneVision-Data']:
        path = ROOT / 'scratch' / (repo.split('/')[-1] + '_dataset_api.json')
        if path.exists():
            info=json.loads(path.read_text(encoding='utf-8'))
        else:
            revisions={'Lin-Chen/ShareGPT4V':'55d02b0bc53a2754095a14110dda6daedd95671d',
                       'lmms-lab/LLaVA-OneVision-Data':'7ca5e5bf8b2006d5dfa0549756198474f0897f63'}
            info = get('https://huggingface.co/api/datasets/' + repo + '/revision/' + revisions[repo]).json()
        path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Dataset metadata:', repo, info.get('sha'), flush=True)
    LOG.write_text(json.dumps(records, indent=2), encoding='utf-8')
    github_repo = 'ZhangqiJiang07/middle_layers_indicating_hallucinations'
    revision_file=ROOT/'eda_resources/hgai_original/revision.json'
    if revision_file.exists():
        revision=json.loads(revision_file.read_text())['revision']
    else:
        info = get(f'https://api.github.com/repos/{github_repo}/commits/main').json()
        revision = info['sha']
    for name in ['chair_eval.py', 'eval_data_loader.py', 'model_manager.py',
                 'constants.py', 'utils.py', 'modify_attention.py', 'README.md']:
        data = get(f'https://raw.githubusercontent.com/{github_repo}/{revision}/{name}').content
        folder = ROOT / 'eda_resources/hgai_original'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
    (folder / 'revision.json').write_text(json.dumps({'repo': github_repo, 'revision': revision}), encoding='utf-8')


if __name__ == '__main__':
    main()
