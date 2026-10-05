#!/usr/bin/env python3
"""Measure how many COCO instance classes are omitted by COCO captions."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eda_results/vlm_extensions"


def main():
    caps = json.loads((ROOT / "coco/annotations/captions_val2014.json").read_text(encoding="utf-8"))
    inst = json.loads((ROOT / "coco/annotations/instances_val2014.json").read_text(encoding="utf-8"))
    classes = {c["id"]: c["name"] for c in inst["categories"]}
    synonyms = {}
    for line in (ROOT / "eda_resources/chair_synonyms.txt").read_text(encoding="utf-8").splitlines():
        terms = [term.strip().casefold() for term in line.split(",") if term.strip()]
        if terms:
            synonyms.update((term, terms[0]) for term in terms)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(term) for term in sorted(synonyms, key=len, reverse=True)) + r")\b", re.I)
    by_image = defaultdict(list)
    for ann in caps["annotations"]:
        by_image[ann["image_id"]].append(ann)
    objects = defaultdict(set)
    for ann in inst["annotations"]:
        objects[ann["image_id"]].add(classes[ann["category_id"]])
    rows = []
    for image in caps["images"]:
        image_id = image["id"]
        truth = objects[image_id]
        caption_sets = [set(synonyms[m.group().casefold()] for m in pattern.finditer(a["caption"]))
                        for a in sorted(by_image[image_id], key=lambda a: a["id"])]
        selected = caption_sets[0]
        union = set().union(*caption_sets)
        rows.append({"image_id": image_id, "instance_classes": len(truth),
                     "captions": len(caption_sets), "captions_with_coco80_word": sum(bool(s) for s in caption_sets),
                     "selected_caption_classes": len(selected), "all_caption_union_classes": len(union),
                     "instance_classes_found_selected": len(truth & selected),
                     "instance_classes_found_union": len(truth & union),
                     "instance_classes_missed_selected": len(truth - selected),
                     "instance_classes_missed_union": len(truth - union),
                     "caption_classes_absent_from_instances_union": len(union - truth)})
    with (OUT / "coco_caption_label_completeness.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    total = sum(row["instance_classes"] for row in rows)
    summary = {"images": len(rows), "images_with_instance_class": sum(row["instance_classes"] > 0 for row in rows),
               "captions": sum(row["captions"] for row in rows),
               "caption_coco80_word_coverage_pct": round(100 * sum(row["captions_with_coco80_word"] for row in rows) / sum(row["captions"] for row in rows), 2),
               "annotated_instance_class_occurrences": total,
               "selected_caption_class_recall_pct": round(100 * sum(row["instance_classes_found_selected"] for row in rows) / total, 2),
               "all_five_caption_union_class_recall_pct": round(100 * sum(row["instance_classes_found_union"] for row in rows) / total, 2),
               "images_with_any_coco80_word_selected_pct": round(100 * sum(row["selected_caption_classes"] > 0 for row in rows) / len(rows), 2),
               "images_with_any_coco80_word_union_pct": round(100 * sum(row["all_caption_union_classes"] > 0 for row in rows) / len(rows), 2),
               "images_with_all_instance_classes_in_union_pct": round(100 * sum(row["instance_classes"] > 0 and row["instance_classes_missed_union"] == 0 for row in rows) / sum(row["instance_classes"] > 0 for row in rows), 2),
               "scope": "Recall of annotated COCO-80 class sets using the repo's lexical CHAIR synonym matcher; instance annotations are not exhaustive visual ground truth."}
    (OUT / "coco_caption_label_completeness.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
