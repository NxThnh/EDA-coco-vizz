#!/usr/bin/env python3
"""Recompute VizWiz refusal and resize sensitivity from the full val image CSV."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "eda_results/tables"


def interval(values, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = [np.median(values[rng.integers(0, len(values), len(values))]) for _ in range(1000)]
    return [round(float(x), 2) for x in np.quantile(estimates, [0.025, 0.975])]


def write(name, rows):
    with (TABLES / name).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    measurements = list(csv.DictReader((TABLES / "laplacian_full_val.csv").open(encoding="utf-8-sig")))
    for row in measurements:
        row["image_id"] = int(row["image_id"])
        row["original_short_edge"] = int(row["original_short_edge"])
        row["laplacian_variance_480px"] = float(row["laplacian_variance_480px"])
        row["mean_luminance_480px"] = float(row["mean_luminance_480px"])
    viz = json.loads((ROOT / "vizwiz/annotations/val.json").read_text(encoding="utf-8"))
    refusal = Counter(a["image_id"] for a in viz["annotations"] if a.get("is_precanned"))
    viz_rows = [row for row in measurements if row["dataset"] == "VizWiz"]
    assert len(viz_rows) == 7750 and len(measurements) == 48254
    groups = []
    for count in range(6):
        values = [row["laplacian_variance_480px"] for row in viz_rows if refusal[row["image_id"]] == count]
        lo, hi = interval(values, 42 + count)
        groups.append({"refusal_captions": count, "images": len(values), "median_laplacian": round(float(np.median(values)), 2),
                       "ci95_low": lo, "ci95_high": hi})
    write("laplacian_by_precanned_full.csv", groups)
    resize = []
    for dataset in ("COCO", "VizWiz"):
        domain = [row for row in measurements if row["dataset"] == dataset]
        for rule, selected in (("all", domain), ("original_short_edge_ge_480", [row for row in domain if row["original_short_edge"] >= 480])):
            values = [row["laplacian_variance_480px"] for row in selected]
            lo, hi = interval(values, 52 if dataset == "COCO" else 53)
            resize.append({"dataset": dataset, "rule": rule, "images": len(values),
                           "median_laplacian": round(float(np.median(values)), 2), "ci95_low": lo, "ci95_high": hi})
    write("laplacian_resize_sensitivity_full.csv", resize)
    summary_path = ROOT / "eda_results/eda_summary_metrics.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["blur_luminance_full_val"] = {
            "scope": "all validation images; EXIF transpose; short edge resized to 480 px",
            "coco_images": 40504, "vizwiz_images": 7750,
            "coco_median_laplacian": resize[0]["median_laplacian"],
            "vizwiz_median_laplacian": resize[2]["median_laplacian"],
            "coco_median_luminance": round(float(np.median([r["mean_luminance_480px"] for r in measurements if r["dataset"] == "COCO"])), 2),
            "vizwiz_median_luminance": round(float(np.median([r["mean_luminance_480px"] for r in measurements if r["dataset"] == "VizWiz"])), 2),
            "vizwiz_by_refusal_file": "eda_results/tables/laplacian_by_precanned_full.csv",
            "resize_sensitivity_file": "eda_results/tables/laplacian_resize_sensitivity_full.csv"}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"by_refusal": groups, "resize": resize}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
