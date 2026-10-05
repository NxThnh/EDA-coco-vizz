#!/usr/bin/env python3
"""Audit COCO val2014 against the official LLaVA-1.5 instruction-tuning JSON.

Download the 665K file from the URL in the report to scratch/ first. The
streaming parser reads only top-level image lines in the official pretty JSON.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "scratch/llava_v1_5_mix665k.json"
OUT = ROOT / "eda_results/vlm_extensions"
EXPECTED_SHA256 = "ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8"


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Download official llava_v1_5_mix665k.json to {SOURCE}")
    digest = hashlib.sha256()
    paths = Counter()
    with SOURCE.open("rb") as source:
        for line in source:
            digest.update(line)
            if line.startswith(b'    "image": '):
                value = json.loads(line.split(b":", 1)[1].strip().rstrip(b","))
                if isinstance(value, str):
                    paths[value] += 1
    if digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError("Official source SHA256 mismatch; results not written")
    coco_ids = {int(Path(path).stem) for path in paths if path.startswith("coco/") and Path(path).stem.isdigit()}
    coco_train2017_ids = {int(Path(path).stem) for path in paths if path.startswith("coco/train2017/") and Path(path).stem.isdigit()}
    val = json.loads((ROOT / "coco/annotations/captions_val2014.json").read_text(encoding="utf-8"))
    manifest = list(csv.DictReader((OUT / "evaluation_manifest_seed42.csv").open(encoding="utf-8-sig")))
    val_ids = {image["id"] for image in val["images"]}
    manifest_ids = {int(row["image_id"]) for row in manifest if row["dataset"] == "COCO"}
    rows = [{"image_id": image["id"], "file_name": image["file_name"],
             "in_llava_mix_coco_path": int(image["id"] in coco_ids),
             "in_llava_mix_train2017_path": int(image["id"] in coco_train2017_ids),
             "in_evaluation_manifest": int(image["id"] in manifest_ids)} for image in val["images"]]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "llava15_instruction_mix_overlap.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    prefixes = Counter(path.split("/", 1)[0] for path in paths)
    result = {"source": "liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json",
              "source_sha256": digest.hexdigest(), "entries_with_string_image": sum(paths.values()),
              "unique_image_paths": len(paths), "image_path_prefix_counts": dict(prefixes),
              "vizwiz_named_image_paths": sum("vizwiz" in path.casefold() for path in paths),
              "coco_val2014_images": len(val_ids), "coco_val2014_in_mix": len(val_ids & coco_ids),
              "coco_val2014_not_in_mix": len(val_ids - coco_ids),
              "manifest_coco_images": len(manifest_ids), "manifest_coco_in_mix": len(manifest_ids & coco_ids),
              "manifest_coco_not_in_mix": len(manifest_ids - coco_ids),
              "scope": "Image paths in official LLaVA-1.5 instruction-tuning mix only; does not audit pretraining or aliases."}
    (OUT / "llava15_instruction_mix_overlap.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
