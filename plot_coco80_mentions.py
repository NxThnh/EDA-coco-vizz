#!/usr/bin/env python3
"""Plot COCO-80 lexical mention rates using one caption per val image."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def build_figure(coco=None, viz=None, categories=None):
    if coco is None:
        coco = json.loads((ROOT / "coco/annotations/captions_val2014.json").read_text(encoding="utf-8"))
    if viz is None:
        viz = json.loads((ROOT / "vizwiz/annotations/val.json").read_text(encoding="utf-8"))
    if categories is None:
        instances = json.loads((ROOT / "coco/annotations/instances_val2014.json").read_text(encoding="utf-8"))
        categories = [x["name"] for x in instances["categories"]]
    synonyms = {}
    for line in (ROOT / "eda_resources/chair_synonyms.txt").read_text(encoding="utf-8").splitlines():
        terms = [x.strip().casefold() for x in line.split(",") if x.strip()]
        synonyms.update((term, terms[0]) for term in terms)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(term) for term in sorted(synonyms, key=len, reverse=True)) + r")\b", re.I)
    counts = {}
    denominators = {}
    for label, data in (("COCO", coco), ("VizWiz", viz)):
        grouped = defaultdict(list)
        for ann in data["annotations"]:
            if label == "VizWiz" and (ann["is_precanned"] or ann["is_rejected"]):
                continue
            if ann["caption"].strip():
                grouped[ann["image_id"]].append(ann)
        chosen = [min(anns, key=lambda x: x["id"])["caption"] for anns in grouped.values()]
        denominators[label] = len(chosen)
        matches = [set(synonyms[m.group().casefold()] for m in pattern.finditer(caption)) for caption in chosen]
        counts[label] = {category: sum(category in found for found in matches) for category in categories}
    rows = [{"category": category, "coco_images_with_mention": counts["COCO"][category],
             "coco_pct": round(100 * counts["COCO"][category] / denominators["COCO"], 3),
             "vizwiz_images_with_mention": counts["VizWiz"][category],
             "vizwiz_pct": round(100 * counts["VizWiz"][category] / denominators["VizWiz"], 3)}
            for category in categories]
    rows.sort(key=lambda row: row["vizwiz_pct"], reverse=True)
    table = ROOT / "eda_results/tables/coco80_caption_mentions.csv"
    with table.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    chart = list(reversed(rows))
    y = np.arange(len(chart))
    fig, ax = plt.subplots(figsize=(12, 23))
    ax.barh(y - .19, [r["coco_pct"] for r in chart], height=.36, label=f"COCO (n={denominators['COCO']:,})", color="#3377b7")
    ax.barh(y + .19, [r["vizwiz_pct"] for r in chart], height=.36, label=f"VizWiz (n={denominators['VizWiz']:,})", color="#e08b3f")
    ax.set_yticks(y, [r["category"] for r in chart], fontsize=8)
    ax.set_xlabel("Images whose selected caption mentions class or CHAIR synonym (%)")
    ax.set_title("COCO-80 lexical mentions in one valid caption per image")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(ROOT / "eda_results/figures/04_coco_category_distribution.png", dpi=200)
    plt.close(fig)
    return rows


if __name__ == "__main__":
    print(f"Plotted {len(build_figure())} COCO-80 classes")
