"""Verify pinned TinyLLaVA bytes, local loading, and one dev attention intervention."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv-eda/Scripts/python.exe'
OUT = ROOT / 'eda_results/hgai'
RUNS = ROOT / 'experiment_outputs/tiny_checkpoint_validation'
RUNS.mkdir(parents=True, exist_ok=True)


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    provenance = OUT / 'tiny_weights_provenance.json'
    assert provenance.is_file(), 'Official checkpoint weights must finish downloading first.'
    weights = json.loads(provenance.read_text(encoding='utf-8'))
    for item in weights:
        path = ROOT / 'model_assets/TinyLLaVA-3.1B' / item['file']
        assert path.stat().st_size == item['bytes']
        assert sha256(path) == item['sha256']
    outputs = []
    for alpha in (0, 0.02):
        output = RUNS / f'tiny_dev1_alpha_{alpha:.3f}.jsonl'
        cmd = [str(PYTHON), 'vlm_experiment.py', '--model', 'tiny', '--partition',
               'dev_common_unmatched500', '--condition', 'detailed64', '--limit', '1',
               '--layers', '5', '18', '--alpha', str(alpha), '--output', str(output), '--resume']
        subprocess.run(cmd, cwd=ROOT, check=True)
        single = records(output)
        assert len(single) == 1
        outputs.append(single[0])
    assert outputs[0]['image_id'] == outputs[1]['image_id']
    assert outputs[0]['hgai_modified_calls'] == 0
    assert outputs[1]['hgai_modified_calls'] > 0
    result = {'checkpoint': 'bczhou/TinyLLaVA-3.1B',
              'revision': weights[0]['revision'], 'weight_shards_verified_sha256': len(weights),
              'partition': 'dev_common_unmatched500', 'images': 1,
              'condition': 'detailed64', 'alpha_tested': 0.02,
              'loaded': True, 'attention_intervention': True,
              'modified_calls': outputs[1]['hgai_modified_calls'],
              'caption_identical': outputs[0]['caption'] == outputs[1]['caption'],
              'baseline_path': str((RUNS / 'tiny_dev1_alpha_0.000.jsonl').relative_to(ROOT)),
              'intervention_path': str((RUNS / 'tiny_dev1_alpha_0.020.jsonl').relative_to(ROOT))}
    (OUT / 'tiny_checkpoint_validation.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('TINY_VALIDATED', result['modified_calls'], 'modified calls')


if __name__ == '__main__':
    main()
