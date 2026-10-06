"""Pin CHAIR evaluator and official POPE COCO question files; no benchmark images."""
import json

from fetch_hgai_assets import ROOT, get
from hgai_cpu_audit import OUT, sha


def main():
    records = []
    previous = json.loads((OUT / 'metrics_provenance.json').read_text()) if (OUT / 'metrics_provenance.json').exists() else []
    for repo, branch in [('LisaAnne/Hallucination', 'master'), ('RUCAIBox/POPE', 'main')]:
        commit = next((r['revision'] for r in previous if r['repo'] == repo), None)
        if commit is None:
            commit = get(f'https://api.github.com/repos/{repo}/commits/{branch}').json()['sha']
        tree = get(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1').json()['tree']
        names = [r['path'] for r in tree if r['type'] == 'blob' and
                 (r['path'] in {'utils/chair.py', 'utils/misc.py', 'data/synonyms.txt', 'LICENSE', 'LICENSE.md', 'evaluate.py'}
                  or ('coco' in r['path'].lower() and r['path'].endswith('.json') and 'pope' in r['path'].lower()))]
        folder = ROOT / 'eda_resources' / ('chair_official' if repo.startswith('LisaAnne') else 'pope_official')
        folder.mkdir(exist_ok=True)
        for name in names:
            path = folder / name.split('/')[-1]
            path.write_bytes(get(f'https://raw.githubusercontent.com/{repo}/{commit}/{name}').content)
            records.append({'repo': repo, 'revision': commit, 'source_path': name,
                            'local_path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)})
            print('Metric asset:', repo, name, flush=True)
    (OUT / 'metrics_provenance.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    old_normalizer = OUT / 'chair_normalizer_provenance.json'
    revision = json.loads(old_normalizer.read_text())['revision'] if old_normalizer.exists() else get('https://api.github.com/repos/clips/pattern/commits/master').json()['sha']
    path = ROOT / 'eda_resources/chair_official/pattern_inflect.py'
    path.write_bytes(get(f'https://raw.githubusercontent.com/clips/pattern/{revision}/pattern/text/en/inflect.py').content)
    normalizer = {'repo': 'clips/pattern', 'revision': revision, 'path': 'pattern/text/en/inflect.py',
                  'sha256': sha(path), 'synonyms_sha256': sha(path.with_name('synonyms.txt'))}
    old_normalizer.write_text(json.dumps(normalizer, indent=2), encoding='utf-8')
    import nltk
    nltk.download('punkt_tab', download_dir=str(ROOT / 'model_assets/nltk_data'), quiet=True, raise_on_error=True)


if __name__ == '__main__':
    main()
