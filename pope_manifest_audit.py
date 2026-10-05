#!/usr/bin/env python3
"""Compare published POPE COCO image list with the local evaluation manifest."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "scratch/coco_pope_random.json"
OUT = ROOT / "eda_results/vlm_extensions"


def main():
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    pope = {int(Path(row["image"]).stem.split("_")[-1]) for row in rows}
    manifest = {int(row["image_id"]) for row in csv.DictReader((OUT / "evaluation_manifest_seed42.csv").open(encoding="utf-8-sig")) if row["dataset"] == "COCO"}
    result = {"source": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json",
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "questions": len(rows), "unique_pope_images": len(pope),
              "local_coco_manifest_images": len(manifest), "overlap_images": len(pope & manifest),
              "interpretation": "Published POPE results require the official questions and image list, not the local random COCO sample."}
    (OUT / "pope_manifest_overlap.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
