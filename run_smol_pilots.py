"""Run and score reproducible SmolVLM2 NF4 feasibility pilots, sequentially on one GPU."""
import argparse
import subprocess
import sys

from hgai_cpu_audit import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caption-images', type=int, default=20)
    parser.add_argument('--pope-questions', type=int, default=60)
    args = parser.parse_args()
    def run(script, *arguments):
        subprocess.run([sys.executable, script, *arguments], cwd=ROOT, check=True)
    output_dir = ROOT / 'experiment_outputs/smol_baseline'
    path = output_dir / f'chair_dev_common_unmatched500_nf4_split_on_n{args.caption_images}.jsonl'
    run('smol_baseline.py', '--limit', str(args.caption_images), *(['--resume'] if path.exists() else []))
    run('score_hgai_baseline.py', str(path))
    for variant in ['random', 'popular', 'adversarial']:
        path = output_dir / f'pope_{variant}_nf4_split_on_n{args.pope_questions}.jsonl'
        run('smol_baseline.py', '--task', 'pope', '--pope-variant', variant, '--limit', str(args.pope_questions),
            *(['--resume'] if path.exists() else []))
        run('score_hgai_baseline.py', str(path))


if __name__ == '__main__':
    main()
