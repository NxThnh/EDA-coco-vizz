"""Paired FP16 versus NF4 CHAIR control on the same dev images.

Run on a GPU with enough memory for the chosen FP16 model.
"""
import argparse
import subprocess

from hgai_cpu_audit import ROOT
from score_vlm_runs import compare, score


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['smol','tiny','llava'],required=True)
    parser.add_argument('--images',type=int,default=30)
    parser.add_argument('--bootstrap',type=int,default=10000)
    args=parser.parse_args()
    if not 10<=args.images<=100:raise ValueError('Use 10–100 paired dev images')
    folder=ROOT/'experiment_outputs/precision_control'
    folder.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for precision in ('nf4','fp16'):
        output=folder/f'{args.model}_dev{args.images}_{precision}.jsonl'
        subprocess.run([str(ROOT/'.venv-eda/Scripts/python.exe'),'vlm_experiment.py',
                        '--model',args.model,'--partition','dev_common_unmatched500',
                        '--limit',str(args.images),'--precision',precision,'--alpha','0',
                        '--output',str(output),'--resume'],cwd=ROOT,check=True)
        score(output)
        outputs.append(output)
    compare(*outputs,folder/f'{args.model}_dev{args.images}_nf4_vs_fp16_ci.json',args.bootstrap)


if __name__=='__main__':main()
