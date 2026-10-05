#!/usr/bin/env python3
"""Geometry for CLIP ViT-L/14-336 center crop and LLaVA square-pad modes."""
import csv
import json
import statistics
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eda_results/vlm_extensions"


def quantiles(values):
    values = sorted(values)
    return {"median": round(statistics.median(values), 4),
            "p90": round(values[int(.9 * (len(values) - 1))], 4),
            "p95": round(values[int(.95 * (len(values) - 1))], 4),
            "max": round(values[-1], 4)}


def dimensions(dataset):
    if dataset == "COCO":
        data = json.loads((ROOT / "coco/annotations/captions_val2014.json").read_text(encoding="utf-8"))
        for image in data["images"]:
            yield image["id"], image["width"], image["height"]
    else:
        data = json.loads((ROOT / "vizwiz/annotations/val.json").read_text(encoding="utf-8"))
        for image in data["images"]:
            with Image.open(ROOT / "vizwiz/val" / image["file_name"]) as im:
                # EXIF transpose returns a new image but does not decode pixels.
                w, h = ImageOps.exif_transpose(im).size
            yield image["id"], w, h


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    with (OUT / "llava15_preprocessing_geometry.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "image_id", "width_exif", "height_exif", "long_edge_px", "long_edge_to_336_ratio", "actual_pad_mode_crop_area_lost_pct", "actual_pad_mode_distortion_factor", "square_pad_area_pct", "pad_tokens_estimate", "patch_tokens", "center_crop_area_lost_pct_if_used", "direct_square_distortion_factor_if_used"])
        writer.writeheader()
        for dataset in ("COCO", "VizWiz"):
            crop, distortion, long_edges, shrink = [], [], [], []
            for image_id, w, h in dimensions(dataset):
                # Resize shortest edge to 336, then 336x336 center crop.
                lost = 100 * (1 - min(w, h) / max(w, h))
                factor = max(w / h, h / w)
                crop.append(lost)
                distortion.append(factor)
                long_edges.append(max(w, h))
                shrink.append(max(w, h) / 336)
                writer.writerow({"dataset": dataset, "image_id": image_id, "width_exif": w, "height_exif": h,
                                 "long_edge_px": max(w, h), "long_edge_to_336_ratio": round(max(w, h) / 336, 4),
                                 "actual_pad_mode_crop_area_lost_pct": 0,
                                 "actual_pad_mode_distortion_factor": 1,
                                 "square_pad_area_pct": round(lost, 4),
                                 "center_crop_area_lost_pct_if_used": round(lost, 4),
                                 "direct_square_distortion_factor_if_used": round(factor, 4),
                                 "patch_tokens": 576, "pad_tokens_estimate": round(576 * lost / 100, 2)})
            summary[dataset] = {"images": len(crop), "image_aspect_ratio": "pad", "actual_crop_area_lost_pct": 0,
                                "actual_distortion_factor": 1,
                                "long_edge_px": quantiles(long_edges),
                                "long_edge_to_336_ratio": quantiles(shrink),
                                "center_crop_area_lost_pct_if_used": quantiles(crop),
                                "square_pad_area_pct": quantiles(crop),
                                "direct_square_distortion_factor_if_used": quantiles(distortion),
                                "patch_tokens": 576, "pad_tokens_estimate_median": round(statistics.median(crop) * 5.76, 2)}
    (OUT / "llava15_preprocessing_geometry.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
