#!/usr/bin/env python3
"""Audit LLaVA VG/GQA aliases and VizWiz EXIF versus Rotated votes."""
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eda_results/vlm_extensions"
VG_SHA256 = "b87a94918cb2ff4d952cf1dfeca0b9cf6cd6fd204c2f8704645653be1163681a"
MIX_SHA256 = "ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_vg():
    metadata_zip = ROOT / "scratch/image_data.json.zip"
    mix_file = ROOT / "scratch/llava_v1_5_mix665k.json"
    if sha256(metadata_zip) != VG_SHA256 or sha256(mix_file) != MIX_SHA256:
        raise ValueError("Visual Genome metadata or LLaVA mix SHA-256 differs from audited source")
    with zipfile.ZipFile(metadata_zip) as archive:
        metadata = json.loads(archive.read("image_data.json"))
    vg_to_coco = {int(row["image_id"]): int(row["coco_id"])
                  for row in metadata if row.get("coco_id") is not None}
    paths = defaultdict(set)
    with mix_file.open("rb") as stream:
        for line in stream:
            if line.startswith(b'    "image": '):
                value = json.loads(line.split(b":", 1)[1].strip().rstrip(b","))
                if not isinstance(value, str):
                    continue
                match = re.search(r"(?:^|/)(\d+)\.(?:jpe?g|png)$", value, re.I)
                if match and value.split("/", 1)[0] in {"vg", "gqa"}:
                    paths[int(match.group(1))].add(value.split("/", 1)[0])
    overlap = list(csv.DictReader((OUT / "llava15_instruction_mix_overlap.csv").open(encoding="utf-8-sig")))
    coco_to_vg = defaultdict(list)
    for vg_id in paths:
        if vg_id in vg_to_coco:
            coco_to_vg[vg_to_coco[vg_id]].append(vg_id)
    rows = []
    for row in overlap:
        coco_id = int(row["image_id"])
        vg_ids = sorted(coco_to_vg[coco_id])
        rows.append({"image_id": coco_id, "in_evaluation_manifest": row["in_evaluation_manifest"],
                     "direct_coco_path": row["in_llava_mix_coco_path"],
                     "vg_gqa_alias": int(bool(vg_ids)),
                     "vg_ids": "|".join(map(str, vg_ids)),
                     "source_prefixes": "|".join(sorted(set().union(*(paths[i] for i in vg_ids)))) if vg_ids else ""})
    write_csv(OUT / "llava15_vg_gqa_alias_overlap.csv", rows)
    eval_rows = [r for r in rows if r["in_evaluation_manifest"] == "1"]
    result = {"vg_metadata_source": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
              "vg_metadata_sha256": VG_SHA256, "mix_sha256": MIX_SHA256,
              "vg_metadata_images": len(metadata), "vg_images_with_coco_id": len(vg_to_coco),
              "mix_vg_gqa_unique_image_ids": len(paths),
              "val_coco_direct": sum(r["direct_coco_path"] == "1" for r in rows),
              "val_vg_gqa_alias": sum(r["vg_gqa_alias"] for r in rows),
              "val_alias_only": sum(r["vg_gqa_alias"] and r["direct_coco_path"] == "0" for r in rows),
              "eval_coco_direct": sum(r["direct_coco_path"] == "1" for r in eval_rows),
              "eval_vg_gqa_alias": sum(r["vg_gqa_alias"] for r in eval_rows),
              "eval_alias_only": sum(r["vg_gqa_alias"] and r["direct_coco_path"] == "0" for r in eval_rows),
              "eval_neither": sum(not r["vg_gqa_alias"] and r["direct_coco_path"] == "0" for r in eval_rows),
              "scope": "Exact COCO ID alias through Visual Genome image_data.json; no pixel matching or pretraining audit."}
    (OUT / "llava15_vg_gqa_alias_overlap.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def audit_exif():
    votes = {row["file_name"]: row for row in csv.DictReader(
        (OUT / "vizwiz_quality_issue_votes_val.csv").open(encoding="utf-8-sig"))}
    rows = []
    for filename, row in votes.items():
        with Image.open(ROOT / "vizwiz/val" / filename) as image:
            orientation = int(image.getexif().get(274, 1))
        rows.append({"file_name": filename, "orientation": orientation,
                     "exif_rotation": int(orientation in {3, 5, 6, 7, 8}),
                     "exif_dimension_swap": int(orientation in {5, 6, 7, 8}),
                     "rotated_majority": int(int(row["rotated_votes"]) >= 3),
                     "rotated_votes": row["rotated_votes"]})
    write_csv(OUT / "vizwiz_exif_rotated_crosscheck.csv", rows)
    cross = Counter((r["exif_rotation"], r["rotated_majority"]) for r in rows)
    swap = Counter((r["exif_dimension_swap"], r["rotated_majority"]) for r in rows)
    result = {"images": len(rows), "exif_orientation_counts": dict(Counter(str(r["orientation"]) for r in rows)),
              "exif_rotation_by_rotated_majority": {f"exif_{a}_label_{b}": cross[a, b]
                                                    for a in (0, 1) for b in (0, 1)},
              "exif_dimension_swap_by_rotated_majority": {f"swap_{a}_label_{b}": swap[a, b]
                                                          for a in (0, 1) for b in (0, 1)},
              "note": "EXIF tag signals requested display transform; Rotated is a human quality issue label."}
    (OUT / "vizwiz_exif_rotated_crosscheck.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps({"vg": audit_vg(), "exif": audit_exif()}, indent=2))
