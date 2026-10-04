#!/usr/bin/env python3
"""Laplacian variance at 480 px for every COCO/VizWiz validation image."""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eda_results/tables"


def measure(item):
    dataset, image_id, path = item
    with Image.open(path) as raw:
        im = ImageOps.exif_transpose(raw).convert("L")
        edge = min(im.size)
        size = tuple(max(1, round(d * 480 / edge)) for d in im.size)
        pixels = np.asarray(im.resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    lap = pixels[:-2, 1:-1] + pixels[2:, 1:-1] + pixels[1:-1, :-2] + pixels[1:-1, 2:] - 4 * pixels[1:-1, 1:-1]
    return {"dataset": dataset, "image_id": image_id, "original_short_edge": edge,
            "laplacian_variance_480px": round(float(lap.var()), 6),
            "mean_luminance_480px": round(float(pixels.mean()), 6)}


def main():
    items = []
    for dataset, ann, folder in (("COCO", "coco/annotations/captions_val2014.json", "coco/val2014"),
                                 ("VizWiz", "vizwiz/annotations/val.json", "vizwiz/val")):
        data = json.loads((ROOT / ann).read_text(encoding="utf-8"))
        items.extend((dataset, image["id"], ROOT / folder / image["file_name"]) for image in data["images"])
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(measure, items))
    with (OUT / "laplacian_full_val.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {}
    for dataset in ("COCO", "VizWiz"):
        selected = [row for row in rows if row["dataset"] == dataset]
        values = np.array([row["laplacian_variance_480px"] for row in selected])
        rng = np.random.default_rng(42)
        medians = np.array([np.median(values[rng.integers(0, len(values), len(values))]) for _ in range(1000)])
        result[dataset] = {"images": len(selected), "median": round(float(np.median(values)), 4),
                           "bootstrap_ci95": [round(float(x), 4) for x in np.quantile(medians, [.025, .975])],
                           "upsampled_images": sum(row["original_short_edge"] < 480 for row in selected)}
    (OUT / "laplacian_full_val_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
