"""Recompute the report's Tables 2 and 4 on a reproducible COCO-500 sample.

The script deliberately does not run CHAIR/POPE: those evaluations require model
outputs (and, for POPE, image-level positive/negative object queries), which are
not present in this repository.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
COCO_ANN = ROOT / "coco" / "annotations" / "instances_val2014.json"
OUT = ROOT / "eda_results" / "coco500"
SEED = 42
N = 500


def save_table_png(df: pd.DataFrame, title: str, path: Path, widths=None) -> None:
    fig_w = 15
    fig_h = max(3.2, 0.43 * (len(df) + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=16, loc="left")
    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        colWidths=widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dbe3")
        if row == 0:
            cell.set_facecolor("#173f5f")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f2f6fa")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with COCO_ANN.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    rng = random.Random(SEED)
    sampled_images = rng.sample(coco["images"], N)
    sampled_ids = {img["id"] for img in sampled_images}

    widths = np.asarray([img["width"] for img in sampled_images])
    heights = np.asarray([img["height"] for img in sampled_images])
    ar = widths / heights
    mp = widths * heights / 1_000_000

    table2 = pd.DataFrame(
        [
            ["N ảnh", N],
            ["Chiều rộng TB ± SD (px)", f"{widths.mean():.1f} ± {widths.std():.1f}"],
            ["Chiều cao TB ± SD (px)", f"{heights.mean():.1f} ± {heights.std():.1f}"],
            ["Trung vị W × H (px)", f"{np.median(widths):.0f} × {np.median(heights):.0f}"],
            ["Khoảng W (px)", f"{widths.min()}–{widths.max()}"],
            ["Khoảng H (px)", f"{heights.min()}–{heights.max()}"],
            ["AR trung bình / trung vị", f"{ar.mean():.2f} / {np.median(ar):.2f}"],
            ["Ảnh ngang (AR > 1.05)", f"{(ar > 1.05).mean() * 100:.2f}%"],
            ["Ảnh dọc (AR < 0.95)", f"{(ar < 0.95).mean() * 100:.2f}%"],
            ["Ảnh vuông", f"{((ar >= 0.95) & (ar <= 1.05)).mean() * 100:.2f}%"],
            ["Megapixel TB", f"{mp.mean():.3f}"],
        ],
        columns=["Chỉ số", "COCO-500 (seed=42)"],
    )

    category_by_id = {c["id"]: c for c in coco["categories"]}
    sampled_annotations = [
        a for a in coco["annotations"] if a["image_id"] in sampled_ids
    ]
    super_counts = Counter(
        category_by_id[a["category_id"]]["supercategory"]
        for a in sampled_annotations
    )
    total_objects = sum(super_counts.values())
    table4 = pd.DataFrame(
        [
            [rank, name, count, f"{count / total_objects * 100:.2f}%"]
            for rank, (name, count) in enumerate(super_counts.most_common(), 1)
        ],
        columns=["Hạng", "Siêu danh mục", "Số object", "Tỷ lệ"],
    )

    OUT.mkdir(parents=True, exist_ok=True)
    table2.to_csv(OUT / "table2_coco500_resolution.csv", index=False, encoding="utf-8-sig")
    table4.to_csv(OUT / "table4_coco500_supercategories.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        sorted(sampled_ids), columns=["image_id"]
    ).to_csv(OUT / "coco500_image_ids_seed42.csv", index=False)
    save_table_png(
        table2,
        "Bảng 2 — Thống kê ảnh trên mẫu ngẫu nhiên 500 ảnh COCO",
        OUT / "table2_coco500_resolution.png",
        widths=[0.55, 0.35],
    )
    save_table_png(
        table4,
        f"Bảng 4 — Phân bố siêu danh mục trên COCO-500 ({total_objects:,} object instances)",
        OUT / "table4_coco500_supercategories.png",
        widths=[0.09, 0.36, 0.22, 0.18],
    )
    print(table2.to_string(index=False))
    print()
    print(table4.to_string(index=False))
    print(f"\nSaved outputs to: {OUT}")


if __name__ == "__main__":
    main()
