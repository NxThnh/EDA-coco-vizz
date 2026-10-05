#!/usr/bin/env python3
"""Reproducible VLM EDA for local COCO/VizWiz annotations.

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


def coco2017_leakage_audit(val2014):
    """Compare image IDs with local train2017 annotation; train images are unnecessary."""
    path = ROOT / "coco/annotations/instances_train2017.json"
    if not path.is_file():
        return {"comparison": "COCO val2014 vs train2017", "status": "annotation_missing",
                "annotation_file": str(path.relative_to(ROOT)), "val2014_images": len(val2014["images"]),
                "overlap_images": "", "overlap_pct": ""}
    train2017 = load("coco/annotations/instances_train2017.json")
    val_ids = {image["id"] for image in val2014["images"]}
    overlap = len(val_ids & {image["id"] for image in train2017["images"]})
    return {"comparison": "COCO val2014 vs train2017", "status": "measured",
            "annotation_file": str(path.relative_to(ROOT)), "val2014_images": len(val_ids),
            "overlap_images": overlap, "overlap_pct": round(100 * overlap / len(val_ids), 3)}


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


def selected_captions(by_image):
    """One valid caption per image, chosen by the smallest annotation ID."""
    return {image_id: min(anns, key=lambda a: a["id"])["caption"]
            for image_id, anns in by_image.items()}


def vocabulary(coco_by_image, viz_by_image, coco_train_by_image):
    counts = []
    for by_image in (coco_by_image, viz_by_image):
        counter = Counter()
        # One deterministic annotation per image avoids giving images with more
        # valid captions a larger weight. Keep all eligible images in each domain.
        for caption in selected_captions(by_image).values():
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
    train_counter = Counter(w for caption in selected_captions(coco_train_by_image).values() for w in words(caption))
    train_types = set(train_counter)
    viz_oov_tokens = sum(n for word, n in v.items() if word not in train_types)
    rng = random.Random(42)
    coco_caps = list(selected_captions(coco_by_image).values())
    noise = []
    for _ in range(100):
        rng.shuffle(coco_caps)
        midpoint = len(coco_caps) // 2
        left = Counter(w for cap in coco_caps[:midpoint] for w in words(cap))
        right = Counter(w for cap in coco_caps[midpoint:] for w in words(cap))
        noise.append(js_divergence(left, right))
    log_odds = []
    # Dirichlet-smoothed unigram log odds; report frequency as well as score.
    total_c, total_v = sum(c.values()), sum(v.values())
    vocabulary_size = len(c.keys() | v.keys())
    for word in c.keys() | v.keys():
        if c[word] + v[word] < 20:
            continue
        score = math.log((v[word] + 0.5) / (total_v + 0.5 * vocabulary_size)) - math.log((c[word] + 0.5) / (total_c + 0.5 * vocabulary_size))
        log_odds.append({"word": word, "coco_count": c[word], "vizwiz_count": v[word], "log_odds_vizwiz_vs_coco": round(score, 5)})
    write_csv("distinctive_words_log_odds.csv",
              sorted(log_odds, key=lambda x: (-x["log_odds_vizwiz_vs_coco"], x["word"]))[:50]
              + sorted(log_odds, key=lambda x: (x["log_odds_vizwiz_vs_coco"], x["word"]))[:50],
              list(log_odds[0]))
    return rows, {"jensen_shannon_divergence_bits": js_divergence(c, v),
                  "coco_half_split_js_bits_mean": round(sum(noise) / len(noise), 6),
                  "coco_half_split_js_bits_p95": sorted(noise)[94],
                  "coco_half_split_seed": 42, "coco_half_split_repeats": 100,
                  "vizwiz_oov_vs_coco_train_token_pct": round(100 * viz_oov_tokens / total_v, 4),
                  "vizwiz_oov_vs_coco_train_type_pct": round(100 * len(set(v) - train_types) / len(v), 4),
                  "tokenization": "lowercase ASCII word regex; one lowest-ID valid caption per image",
                  "warning": "Lexical divergence, not a VLM embedding distance."}


def caption_agreement(data, vizwiz=False):
    grouped = defaultdict(list)
    for a in data["annotations"]:
        grouped[a["image_id"]].append(a)
    rows = []
    for image in data["images"]:
        anns = grouped[image["id"]]
        valid = [a for a in anns if (not vizwiz or (not a["is_precanned"] and not a["is_rejected"])) and a["caption"].strip()]
        sets = [set(words(a["caption"])) - STOP for a in valid]
        scores = [len(sets[i] & sets[j]) / len(sets[i] | sets[j])
                  for i in range(len(sets)) for j in range(i + 1, len(sets)) if sets[i] | sets[j]]
        rows.append({"image_id": image["id"], "file_name": image["file_name"],
                     "valid_captions": len(valid), "precanned_count": sum(a.get("is_precanned", False) for a in anns),
                     "rejected_count": sum(a.get("is_rejected", False) for a in anns),
                     "pairwise_content_jaccard_mean": round(sum(scores) / len(scores), 4) if scores else "",
                     "pair_count": len(scores)})
    return rows


def chair_ontology(viz):
    lines = (ROOT / "eda_resources/chair_synonyms.txt").read_text(encoding="utf-8").splitlines()
    synonyms = {}
    for line in lines:
        terms = [term.strip().casefold() for term in line.split(",") if term.strip()]
        for term in terms:
            synonyms[term] = terms[0]
    # Longest phrase first prevents counting 'phone' within 'mobile phone'.
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(term) for term in sorted(synonyms, key=len, reverse=True)) + r")\b", re.I)
    grouped = defaultdict(list)
    for ann in viz["annotations"]:
        if not ann["is_precanned"] and not ann["is_rejected"] and ann["caption"].strip():
            grouped[ann["image_id"]].append(ann)
    rows = []
    for image in viz["images"]:
        anns = grouped[image["id"]]
        mentions = [set(synonyms[m.group().casefold()] for m in pattern.finditer(a["caption"])) for a in anns]
        counts = Counter(term for terms in mentions for term in terms)
        rows.append({"image_id": image["id"], "file_name": image["file_name"],
                     "valid_captions": len(anns), "captions_with_coco80_word": sum(bool(x) for x in mentions),
                     "image_has_coco80_word": int(any(mentions)),
                     "objects_mentioned_by_at_least_two_captions": ";".join(sorted(term for term, count in counts.items() if count >= 2))})
    eligible = [r for r in rows if r["valid_captions"]]
    return rows, {"valid_caption_images": len(eligible),
                  "images_with_coco80_word": sum(r["image_has_coco80_word"] for r in eligible),
                  "image_coverage_pct": round(100 * sum(r["image_has_coco80_word"] for r in eligible) / len(eligible), 3),
                  "valid_captions": sum(r["valid_captions"] for r in eligible),
                  "captions_with_coco80_word": sum(r["captions_with_coco80_word"] for r in eligible),
                  "caption_coverage_pct": round(100 * sum(r["captions_with_coco80_word"] for r in eligible) / sum(r["valid_captions"] for r in eligible), 3),
                  "method": "CHAIR synonyms; lexical mention only, not visual ground truth"}


def reference_token_lengths(coco, viz):
    model_path = ROOT / "model_assets/llava-v1.5-7b/tokenizer.model"
    if not model_path.is_file():
        return {"status": "tokenizer_missing", "path": str(model_path.relative_to(ROOT))}
    import sentencepiece as spm
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    result = {"status": "measured", "tokenizer": "LLaVA-1.5-7B tokenizer.model (SentencePiece)",
              "scope": "one lowest-ID valid reference caption per image; excludes prompt, image tokens and model output"}
    for name, data, filtered in (("COCO", coco, False), ("VizWiz", viz, True)):
        caps = selected_captions(valid_captions(data, filtered))
        lengths = sorted(len(processor.encode(caption)) for caption in caps.values())
        result[name] = {"images": len(lengths), "p50": lengths[len(lengths) // 2],
                        "p90": lengths[int(.9 * (len(lengths) - 1))],
                        "p95": lengths[int(.95 * (len(lengths) - 1))],
                        "p99": lengths[int(.99 * (len(lengths) - 1))], "max": lengths[-1]}
    return result


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
    stratum_rows = []
    for key in sorted(strata):
        stratum_rows.append({"refusal_count_capped": key[0], "text_detected": int(key[1]),
                             "population_images": len(strata[key]), "sample_images": allocation[key]})
    return rows, stratum_rows


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
    leakage = coco2017_leakage_audit(cv)
    write_csv("coco2017_leakage.csv", [leakage], list(leakage))
    vocab, divergence = vocabulary(valid_captions(cv), valid_captions(vv, True), valid_captions(ct))
    write_csv("vocabulary.csv", vocab, list(vocab[0]))
    agreement = caption_agreement(vv, True)
    coco_agreement = caption_agreement(cv)
    write_csv("vizwiz_caption_agreement.csv", agreement, list(agreement[0]))
    write_csv("coco_caption_agreement.csv", coco_agreement, list(coco_agreement[0]))
    eligible = [r["pairwise_content_jaccard_mean"] for r in agreement if r["pairwise_content_jaccard_mean"] != ""]
    coco_eligible = [r["pairwise_content_jaccard_mean"] for r in coco_agreement if r["pairwise_content_jaccard_mean"] != ""]
    ontology_rows, ontology_summary = chair_ontology(vv)
    write_csv("vizwiz_chair_ontology_mentions.csv", ontology_rows, list(ontology_rows[0]))
    summary = {"lexical": divergence, "vizwiz_chair_ontology": ontology_summary,
               "llava_reference_token_lengths": reference_token_lengths(cv, vv),
               "vizwiz_caption_agreement": {"images_with_two_or_more_valid_captions": len(eligible),
                                             "mean_image_level_pairwise_content_jaccard": round(sum(eligible) / len(eligible), 4),
                                             "images_with_zero_valid_captions": sum(r["valid_captions"] == 0 for r in agreement),
                                             "warning": "Word overlap is not semantic agreement or image-caption alignment."},
               "coco_caption_agreement": {"images_with_two_or_more_valid_captions": len(coco_eligible),
                                          "mean_image_level_pairwise_content_jaccard": round(sum(coco_eligible) / len(coco_eligible), 4)},
               "scope": "COCO/VizWiz val images, LLaVA tokenizer and CLIP vision/text encoder are local. LLaVA generation weights and VizWiz object ground truth are not supplied."}
    with (OUT / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    manifest, strata = evaluation_manifest(cv, vv, agreement)
    write_csv("evaluation_manifest_seed42.csv", manifest, list(manifest[0]))
    write_csv("evaluation_manifest_strata_seed42.csv", strata, list(strata[0]))
    ontology_by_id = {str(row["image_id"]): row for row in ontology_rows}
    review_pool = [ontology_by_id[str(row["image_id"])] for row in manifest
                   if row["dataset"] == "VizWiz" and ontology_by_id[str(row["image_id"])]["objects_mentioned_by_at_least_two_captions"]]
    review = [dict(row, confirmed_objects="", rejected_suggestions="", additional_visible_objects="",
                   reviewer_id="", review_status="unreviewed") for row in random.Random(42).sample(review_pool, min(100, len(review_pool)))]
    write_csv("vizwiz_object_consensus_review100_seed42.csv", review, list(review[0]))
    annotation_template = [
        {"image_id": row["image_id"], "file_name": row["file_name"],
         "objects_visible": "", "objects_uncertain": "", "readable_scene_text": "",
         "image_unanswerable": "", "annotator_id": "", "review_status": "unlabeled"}
        for row in manifest if row["dataset"] == "VizWiz"
    ]
    write_csv("vizwiz_object_annotation_template.csv", annotation_template, list(annotation_template[0]))
    print(json.dumps({"integrity": audits, "overlap": overlaps, "coco2017_leakage": leakage, "summary": summary,
                      "evaluation_manifest_rows": len(manifest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
