#!/usr/bin/env python3
"""Reproducible, annotation-only VLM EDA for local COCO/VizWiz data.

Run: python vlm_eda_extensions.py
The output is deliberately limited to claims supported by local annotations.
"""

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eda_results" / "vlm_extensions"
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?", re.I)
STOP = set("a an the is are was were of on in at to for with and or this that there it its as by from near next one two three some many has have be been being".split())


def load(relative):
    with (ROOT / relative).open(encoding="utf-8") as stream:
        return json.load(stream)


def words(caption):
    return TOKEN.findall(caption.casefold())


def write_csv(name, rows, fields):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_audit(dataset, split, data, image_dir=None):
    images = data["images"]
    annotations = data.get("annotations", [])
    ids = [image["id"] for image in images]
    names = [image["file_name"] for image in images]
    annotation_ids = [a["id"] for a in annotations]
    image_ids = set(ids)
    refs = Counter(a["image_id"] for a in annotations)
    bad_flags = sum(
        not isinstance(a.get("is_precanned"), bool) or not isinstance(a.get("is_rejected"), bool)
        for a in annotations
    ) if dataset == "VizWiz" else None
    missing = empty = None
    if image_dir is not None:
        missing = sum(not (image_dir / name).is_file() for name in names)
        empty = sum((image_dir / name).is_file() and (image_dir / name).stat().st_size == 0 for name in names)
    return {
        "dataset": dataset, "split": split, "images": len(images), "annotations": len(annotations),
        "duplicate_image_ids": len(ids) - len(image_ids),
        "duplicate_filenames": len(names) - len(set(names)),
        "duplicate_annotation_ids": len(annotation_ids) - len(set(annotation_ids)),
        "orphan_annotations": sum(a["image_id"] not in image_ids for a in annotations),
        "images_without_annotations": sum(refs[image_id] == 0 for image_id in ids) if annotations else None,
        "images_not_five_captions": sum(refs[image_id] != 5 for image_id in ids) if annotations else None,
        "empty_captions": sum(not str(a.get("caption", "")).strip() for a in annotations),
        "invalid_vizwiz_flags": bad_flags,
        "missing_local_images": missing, "zero_byte_local_images": empty,
    }


def overlap_audit(dataset, train, val):
    train_ids = {i["id"] for i in train["images"]}
    val_ids = {i["id"] for i in val["images"]}
    train_names = {i["file_name"] for i in train["images"]}
    val_names = {i["file_name"] for i in val["images"]}
    def eligible(a):
        return (not a.get("is_precanned", False) and not a.get("is_rejected", False)
                and bool(words(a["caption"])))
    train_caps = {" ".join(words(a["caption"])) for a in train["annotations"] if eligible(a)}
    val_caps = [" ".join(words(a["caption"])) for a in val["annotations"] if eligible(a)]
    return {
        "dataset": dataset, "overlap_image_ids": len(train_ids & val_ids),
        "overlap_filenames": len(train_names & val_names),
        "eligible_val_captions": len(val_caps),
        "val_captions_matching_train_count": sum(c in train_caps for c in val_caps),
        "val_captions_matching_train_pct": round(100 * sum(c in train_caps for c in val_caps) / len(val_caps), 3),
        "interpretation": "Valid captions only; matches can be generic wording, not proof of image or model pretraining leakage."
    }


def valid_captions(data, vizwiz=False):
    by_image = defaultdict(list)
    for a in data["annotations"]:
        if vizwiz and (a["is_precanned"] or a["is_rejected"]):
            continue
        if a["caption"].strip():
            by_image[a["image_id"]].append(a)
    return by_image


def js_divergence(p, q):
    total_p, total_q = sum(p.values()), sum(q.values())
    if not total_p or not total_q:
        return None
    value = 0.0
    for term in p.keys() | q.keys():
        a, b = p[term] / total_p, q[term] / total_q
        m = (a + b) / 2
        if a:
            value += a * math.log2(a / m) / 2
        if b:
            value += b * math.log2(b / m) / 2
    return round(value, 6)


def vocabulary(coco_by_image, viz_by_image):
    counts = []
    for by_image in (coco_by_image, viz_by_image):
        counter = Counter()
        # One deterministic annotation per image avoids giving images with more
        # valid captions a larger weight. Keep all eligible images in each domain.
        for image_id in sorted(by_image):
            caption = min(by_image[image_id], key=lambda a: a["id"])["caption"]
            counter.update(words(caption))
        counts.append(counter)
    c, v = counts
    rows = []
    for label, counter in (("COCO", c), ("VizWiz", v)):
        total = sum(counter.values())
        rows.append({"dataset": label, "images_with_valid_caption": len(coco_by_image if label == "COCO" else viz_by_image),
                     "word_tokens": total, "word_types": len(counter),
                     "type_token_ratio": round(len(counter) / total, 6),
                     "singleton_types": sum(n == 1 for n in counter.values()),
                     "top_20_words": ", ".join(f"{w}:{n}" for w, n in counter.most_common(20))})
    return rows, {"jensen_shannon_divergence_bits": js_divergence(c, v),
                  "tokenization": "lowercase ASCII word regex; one lowest-ID valid caption per image",
                  "warning": "Lexical divergence, not a VLM embedding distance."}


def caption_agreement(viz):
    grouped = defaultdict(list)
    for a in viz["annotations"]:
        grouped[a["image_id"]].append(a)
    rows = []
    for image in viz["images"]:
        anns = grouped[image["id"]]
        valid = [a for a in anns if not a["is_precanned"] and not a["is_rejected"] and a["caption"].strip()]
        sets = [set(words(a["caption"])) - STOP for a in valid]
        scores = [len(sets[i] & sets[j]) / len(sets[i] | sets[j])
                  for i in range(len(sets)) for j in range(i + 1, len(sets)) if sets[i] | sets[j]]
        rows.append({"image_id": image["id"], "file_name": image["file_name"],
                     "valid_captions": len(valid), "precanned_count": sum(a["is_precanned"] for a in anns),
                     "rejected_count": sum(a["is_rejected"] for a in anns),
                     "pairwise_content_jaccard_mean": round(sum(scores) / len(scores), 4) if scores else "",
                     "pair_count": len(scores)})
    return rows


def evaluation_manifest(coco, viz, agreement, n=500, seed=42):
    rng = random.Random(seed)
    # Stratify VizWiz on annotation-backed conditions, then sample reproducibly.
    by_id = {r["image_id"]: r for r in agreement}
    strata = defaultdict(list)
    for image in viz["images"]:
        row = by_id[image["id"]]
        refusal = min(3, row["precanned_count"] + row["rejected_count"])
        strata[(refusal, bool(image["text_detected"]))].append(image)
    total = sum(map(len, strata.values()))
    quotas = {key: len(items) * n / total for key, items in strata.items()}
    allocation = {key: int(quotas[key]) for key in strata}
    for key in sorted(strata, key=lambda k: (-(quotas[k] - allocation[k]), k))[:n - sum(allocation.values())]:
        allocation[key] += 1
    rows = []
    for key in sorted(strata):
        sampled = rng.sample(sorted(strata[key], key=lambda i: i["id"]), allocation[key])
        for image in sorted(sampled, key=lambda i: i["id"]):
            rows.append({"dataset": "VizWiz", "image_id": image["id"], "file_name": image["file_name"],
                         "stratum": f"refusal_count_capped_{key[0]}__text_{int(key[1])}", "seed": seed})
    selected = rng.sample(sorted(coco["images"], key=lambda i: i["id"]), n)
    for image in sorted(selected, key=lambda i: i["id"]):
        rows.append({"dataset": "COCO", "image_id": image["id"], "file_name": image["file_name"],
                     "stratum": "random_val2014", "seed": seed})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ct = load("coco/annotations/captions_train2014.json")
    cv = load("coco/annotations/captions_val2014.json")
    vt = load("vizwiz/annotations/train.json")
    vv = load("vizwiz/annotations/val.json")
    vx = load("vizwiz/annotations/test.json")
    audits = [split_audit("COCO", "train", ct), split_audit("COCO", "val", cv, ROOT / "coco/val2014"),
              split_audit("VizWiz", "train", vt), split_audit("VizWiz", "val", vv, ROOT / "vizwiz/val"),
              split_audit("VizWiz", "test", vx)]
    write_csv("integrity.csv", audits, list(audits[0]))
    overlaps = [overlap_audit("COCO", ct, cv), overlap_audit("VizWiz", vt, vv)]
    write_csv("train_val_overlap.csv", overlaps, list(overlaps[0]))
    vocab, divergence = vocabulary(valid_captions(cv), valid_captions(vv, True))
    write_csv("vocabulary.csv", vocab, list(vocab[0]))
    agreement = caption_agreement(vv)
    write_csv("vizwiz_caption_agreement.csv", agreement, list(agreement[0]))
    eligible = [r["pairwise_content_jaccard_mean"] for r in agreement if r["pairwise_content_jaccard_mean"] != ""]
    summary = {"lexical": divergence,
               "vizwiz_caption_agreement": {"images_with_two_or_more_valid_captions": len(eligible),
                                             "mean_image_level_pairwise_content_jaccard": round(sum(eligible) / len(eligible), 4),
                                             "images_with_zero_valid_captions": sum(r["valid_captions"] == 0 for r in agreement),
                                             "warning": "Word overlap is not semantic agreement or image-caption alignment."},
               "scope": "Only COCO/VizWiz val images are local. No VLM weights, image encoder, model tokenizer or VizWiz object ground truth supplied."}
    with (OUT / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    manifest = evaluation_manifest(cv, vv, agreement)
    write_csv("evaluation_manifest_seed42.csv", manifest, list(manifest[0]))
    annotation_template = [
        {"image_id": row["image_id"], "file_name": row["file_name"],
         "objects_visible": "", "objects_uncertain": "", "readable_scene_text": "",
         "image_unanswerable": "", "annotator_id": "", "review_status": "unlabeled"}
        for row in manifest if row["dataset"] == "VizWiz"
    ]
    write_csv("vizwiz_object_annotation_template.csv", annotation_template, list(annotation_template[0]))
    print(json.dumps({"integrity": audits, "overlap": overlaps, "summary": summary,
                      "evaluation_manifest_rows": len(manifest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
