#!/usr/bin/env python3
"""Figures for the measured CLIP and VizWiz quality findings in Section 8."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "eda_results/vlm_extensions"
FIG = ROOT / "eda_results/figures"


def main():
    clip = json.loads((SOURCE / "clip_embedding_audit.json").read_text(encoding="utf-8"))
    quality = json.loads((SOURCE / "vizwiz_quality_issue_summary.json").read_text(encoding="utf-8"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.7))
    labels = ["COCO within", "VizWiz within", "Cross domain"]
    pad = clip["image_domain"]
    raw = clip["image_domain_no_pad"]
    keys = ["mean_within_coco_cosine", "mean_within_vizwiz_cosine", "mean_cross_domain_cosine"]
    x = np.arange(3)
    ax1.bar(x - .18, [pad[k] for k in keys], .35, label="Square pad", color="#e08b3f")
    ax1.bar(x + .18, [raw[k] for k in keys], .35, label="Default resize/crop", color="#3377b7")
    ax1.set_xticks(x, labels, rotation=15)
    ax1.set_ylabel("Mean CLIP image cosine")
    ax1.set_title(f"Image similarity (n={pad['balanced_n_per_domain']} per domain)")
    ax1.legend(fontsize=8)
    for x_pos, dataset in enumerate(("COCO", "VizWiz")):
        block = clip[dataset]
        ax2.bar(x_pos - .18, block["paired_cosine_mean"], .35, label="Matched caption" if x_pos == 0 else None, color="#4a9b71")
        ax2.bar(x_pos + .18, block["shuffled_cosine_mean"], .35, label="Shuffled caption" if x_pos == 0 else None, color="#a3a3a3")
    ax2.set_xticks([0, 1], ["COCO", "VizWiz"])
    ax2.set_ylabel("Mean CLIP image–text cosine")
    ax2.set_title("Matched vs shuffled caption")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "08_clip_domain_alignment.png", dpi=220)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    names = ["Poor framing", "Blur", "Rotated", "Too dark", "Too bright", "Obstructed"]
    fields = ["poor_framing", "blur", "rotated", "too_dark", "too_bright", "obstructed"]
    ax1.barh(list(reversed(names)), [quality["flaws"][field]["majority_pct"] for field in reversed(fields)], color="#c46b45")
    ax1.set_xlabel("VizWiz val images with ≥3/5 flaw votes (%)")
    ax1.set_title("Official image quality labels")
    groups = quality["by_refusal"]
    ax2.plot([row["refusal_captions"] for row in groups], [row["blur_majority_pct"] for row in groups], "o-", label="Blur", color="#ad4f59")
    ax2.plot([row["refusal_captions"] for row in groups], [row["poor_framing_majority_pct"] for row in groups], "o-", label="Poor framing", color="#4d77ac")
    ax2.set_xlabel("Number of refusal captions (out of 5)")
    ax2.set_ylabel("Images with ≥3/5 flaw votes (%)")
    ax2.set_title("Quality labels by refusal count")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(axis="x" if ax is ax1 else "y", alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "09_vizwiz_quality_labels.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
