#!/usr/bin/env python3
"""Build context EDA image-caption manifests from local COCO/VizWiz data.

The 1,000-image evaluation set is fixed by the existing EDA manifest. Raw
validation images outside that set form a separate development partition.
Training images are currently absent; training records are metadata only.
HGAI tuning/evaluation uses data_prepared/hgai from hgai_cpu_audit.py instead.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data_prepared"
EVAL = ROOT / "eda_results/vlm_extensions/evaluation_manifest_seed42.csv"
HASHES = ROOT / "eda_results/vlm_extensions/val_image_hashes.csv"
SOURCES = (
    ("COCO", "train", "coco/annotations/captions_train2014.json", "coco/train2014"),
    ("COCO", "val", "coco/annotations/captions_val2014.json", "coco/val2014"),
    ("VizWiz", "train", "vizwiz/annotations/train.json", "vizwiz/train"),
    ("VizWiz", "val", "vizwiz/annotations/val.json", "vizwiz/val"),
)


def read_json(relative):
    with (ROOT / relative).open(encoding="utf-8") as stream:
        return json.load(stream)


def safe_image_path(folder, filename):
    # Filenames are untrusted metadata, even when supplied by an annotation file.
    if Path(filename).name != filename or filename in ("", ".", ".."):
        raise ValueError(f"Unsafe image filename: {filename!r}")
    return f"{folder}/{filename}"


def clean_caption(text):
    return " ".join(text.split())


def load_eval():
    selected = {}
    with EVAL.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = (row["dataset"], int(row["image_id"]))
            if key in selected:
                raise ValueError(f"Duplicate evaluation image: {key}")
            selected[key] = row
    if Counter(dataset for dataset, _ in selected) != {"COCO": 500, "VizWiz": 500}:
        raise ValueError("Evaluation manifest must contain 500 images per dataset")
    return selected


def duplicate_dev_exclusions(selected):
    """Keep exact-byte siblings of eval images out of the development set."""
    if not HASHES.is_file():
        raise FileNotFoundError(f"Exact-byte split check requires {HASHES}")
    rows = list(csv.DictReader(HASHES.open(newline="", encoding="utf-8-sig")))
    by_hash = defaultdict(list)
    for row in rows:
        if row["decode_ok"] == "1" and row["md5"]:
            by_hash[(row["dataset"], row["md5"])].append(row)
    excluded = []
    for members in by_hash.values():
        eval_members = [m for m in members if (m["dataset"], int(m["image_id"])) in selected]
        if eval_members:
            for member in members:
                key = (member["dataset"], int(member["image_id"]))
                if key not in selected:
                    excluded.append({"dataset": member["dataset"], "image_id": member["image_id"],
                                     "file_name": member["file_name"], "md5": member["md5"],
                                     "eval_sibling_ids": "|".join(m["image_id"] for m in eval_members)})
    return sorted(excluded, key=lambda r: (r["dataset"], int(r["image_id"])))


def make_record(dataset, split, image, annotations, folder, eval_row):
    filename = image["file_name"]
    path = safe_image_path(folder, filename)
    caption_rows = []
    for ann in sorted(annotations, key=lambda item: item["id"]):
        caption = clean_caption(ann.get("caption", ""))
        precanned = bool(ann.get("is_precanned", False))
        rejected = bool(ann.get("is_rejected", False))
        caption_rows.append({
            "annotation_id": ann["id"], "text": caption,
            "is_precanned": precanned, "is_rejected": rejected,
            "usable_for_captioning": bool(caption) and not (precanned or rejected),
        })
    usable = [item["text"] for item in caption_rows if item["usable_for_captioning"]]
    unavailable = sum(item["is_precanned"] or item["is_rejected"] for item in caption_rows)
    partition = "train" if split == "train" else ("eval" if eval_row else "dev")
    return {
        "dataset": dataset, "source_split": split, "partition": partition,
        "image_id": image["id"], "file_name": filename, "image_path": path,
        "image_available": (ROOT / path).is_file(),
        "width_annotation": image.get("width"), "height_annotation": image.get("height"),
        "text_detected": image.get("text_detected"),
        "eval_stratum": eval_row["stratum"] if eval_row else None,
        "captions": caption_rows, "valid_captions": usable,
        "valid_caption_count": len(usable), "refusal_or_rejected_count": unavailable,
        # This is an annotation-derived proxy, not a gold unanswerability label.
        "majority_refusal_proxy": unavailable >= 3 if dataset == "VizWiz" else None,
    }


def build(verify_images=False):
    OUT.mkdir(exist_ok=True)
    selected = load_eval()
    duplicate_exclusions = duplicate_dev_exclusions(selected)
    excluded_keys = {(r["dataset"], int(r["image_id"])) for r in duplicate_exclusions}
    encountered = set()
    counts = Counter()
    missing = Counter()
    zero_byte = Counter()
    failed_verify = []
    object_worklist = []
    if verify_images:
        from PIL import Image, ImageOps
    streams = {p: (OUT / f"{p}.jsonl").open("w", encoding="utf-8", newline="\n")
               for p in ("train", "dev", "eval")}
    try:
        for dataset, split, annotation_file, image_dir in SOURCES:
            data = read_json(annotation_file)
            grouped = defaultdict(list)
            for ann in data["annotations"]:
                grouped[ann["image_id"]].append(ann)
            for image in sorted(data["images"], key=lambda item: item["id"]):
                key = (dataset, image["id"])
                if split == "val" and key in excluded_keys:
                    continue
                eval_row = selected.get(key) if split == "val" else None
                if eval_row:
                    if eval_row["file_name"] != image["file_name"]:
                        raise ValueError(f"Evaluation filename mismatch: {key}")
                    encountered.add(key)
                row = make_record(dataset, split, image, grouped[image["id"]], image_dir, eval_row)
                if dataset == "VizWiz" and eval_row:
                    for annotator in ("A", "B"):
                        object_worklist.append({
                            "image_id": image["id"], "file_name": image["file_name"],
                            "image_path": row["image_path"], "annotator_id": annotator,
                            "objects_visible": "", "objects_uncertain": "",
                            "readable_scene_text": "", "image_unanswerable": "",
                            "review_status": "unlabeled",
                        })
                partition = row["partition"]
                counts[(dataset, partition, "images")] += 1
                counts[(dataset, partition, "valid_captions")] += row["valid_caption_count"]
                if not row["valid_caption_count"]:
                    counts[(dataset, partition, "images_without_valid_caption")] += 1
                if not row["image_available"]:
                    missing[(dataset, partition)] += 1
                else:
                    image_path = ROOT / row["image_path"]
                    if image_path.stat().st_size == 0:
                        zero_byte[(dataset, partition)] += 1
                    elif verify_images:
                        try:
                            with Image.open(image_path) as picture:
                                # Force full pixel decode and exercise the same EXIF
                                # orientation handling required by model loaders.
                                picture.load()
                                ImageOps.exif_transpose(picture)
                        except Exception as exc:
                            failed_verify.append({"image_path": row["image_path"], "error": str(exc)})
                streams[partition].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for stream in streams.values():
            stream.close()
    if encountered != set(selected):
        raise ValueError(f"Evaluation images missing from annotation: {len(set(selected) - encountered)}")
    with (OUT / "exact_duplicate_dev_exclusions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ("dataset", "image_id", "file_name", "md5", "eval_sibling_ids")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(duplicate_exclusions)
    worklist_path = OUT / "vizwiz_object_annotation_worklist.csv"
    preserve_worklist = False
    if worklist_path.exists():
        with worklist_path.open(newline="", encoding="utf-8-sig") as stream:
            preserve_worklist = any(
                row.get("review_status") != "unlabeled" or
                any(row.get(field, "").strip() for field in
                    ("objects_visible", "objects_uncertain", "readable_scene_text", "image_unanswerable"))
                for row in csv.DictReader(stream)
            )
    if not preserve_worklist:
        with worklist_path.open("w", newline="", encoding="utf-8-sig") as stream:
            fields = ("image_id", "file_name", "image_path", "annotator_id", "objects_visible",
                      "objects_uncertain", "readable_scene_text", "image_unanswerable", "review_status")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(object_worklist)
    summary = {
        "schema_version": 1,
        "evaluation_source": str(EVAL.relative_to(ROOT)).replace("\\", "/"),
        "partitions": [
            {"dataset": d, "partition": p, "images": counts[(d, p, "images")],
             "valid_captions": counts[(d, p, "valid_captions")],
             "images_without_valid_caption": counts[(d, p, "images_without_valid_caption")],
             "missing_images": missing[(d, p)], "zero_byte_images": zero_byte[(d, p)]}
            for d in ("COCO", "VizWiz") for p in ("train", "dev", "eval")
        ],
        "image_decode_verify_enabled": verify_images,
        "exact_duplicate_dev_exclusions": len(duplicate_exclusions),
        "image_decode_verify_failures": failed_verify,
        "notes": [
            "Training images are not present locally; train.jsonl contains annotation metadata only.",
            "Eval images and their exact-byte siblings are excluded from dev.jsonl; val is never used for training.",
            "VizWiz refusal/rejected captions are retained with flags but excluded from valid_captions.",
            "majority_refusal_proxy is not a gold label of unanswerability.",
            "Apply EXIF transpose before model preprocessing; no images are rewritten here.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-images", action="store_true", help="Fully decode all local images and apply EXIF orientation with Pillow")
    args = parser.parse_args()
    result = build(args.verify_images)
    for row in result["partitions"]:
        print(f"{row['dataset']:6} {row['partition']:5} images={row['images']:6} valid_captions={row['valid_captions']:6} missing={row['missing_images']}")
    print(f"Image decode failures: {len(result['image_decode_verify_failures'])}")
