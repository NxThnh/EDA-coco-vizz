#!/usr/bin/env python3
"""CLIP ViT-L/14-336 measurements on the fixed 500+500 validation manifest."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eda_results/vlm_extensions"
MODEL = ROOT / "model_assets/clip-vit-large-patch14-336"


def main():
    if not (MODEL / "pytorch_model.bin").is_file():
        raise FileNotFoundError(MODEL / "pytorch_model.bin")
    processor = CLIPProcessor.from_pretrained(MODEL, local_files_only=True)
    model = CLIPModel.from_pretrained(MODEL, local_files_only=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    manifests = list(csv.DictReader((OUT / "evaluation_manifest_seed42.csv").open(encoding="utf-8-sig", newline="")))
    captions = {}
    for dataset, path in (("COCO", "coco/annotations/captions_val2014.json"), ("VizWiz", "vizwiz/annotations/val.json")):
        data = json.loads((ROOT / path).read_text(encoding="utf-8"))
        grouped = defaultdict(list)
        for ann in data["annotations"]:
            if dataset == "VizWiz" and (ann["is_precanned"] or ann["is_rejected"]):
                continue
            if ann["caption"].strip():
                grouped[str(ann["image_id"])].append(ann)
        for image_id, anns in grouped.items():
            captions[dataset, image_id] = min(anns, key=lambda a: a["id"])["caption"]
    rows = [(row, captions.get((row["dataset"], row["image_id"]))) for row in manifests]
    viz_data = json.loads((ROOT / "vizwiz/annotations/val.json").read_text(encoding="utf-8"))
    viz_flags = {str(image["id"]): bool(image["text_detected"]) for image in viz_data["images"]}
    viz_refusals = defaultdict(int)
    for ann in viz_data["annotations"]:
        if ann["is_precanned"] or ann["is_rejected"]:
            viz_refusals[str(ann["image_id"])] += 1
    image_vecs, unpadded_vecs, text_vecs = [], [], []
    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            images, originals = [], []
            for row, _ in batch:
                folder = "coco/val2014" if row["dataset"] == "COCO" else "vizwiz/val"
                with Image.open(ROOT / folder / row["file_name"]) as im:
                    image = ImageOps.exif_transpose(im).convert("RGB")
                    originals.append(image)
                    side = max(image.size)
                    mean = tuple(int(x * 255) for x in processor.image_processor.image_mean)
                    padded = Image.new("RGB", (side, side), mean)
                    padded.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
                    images.append(padded)
            inputs = processor(text=[caption or "" for _, caption in batch], images=images,
                               return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            unpadded_pixels = processor.image_processor(images=originals, return_tensors="pt")["pixel_values"].to(device)
            unpadded_features = model.get_image_features(pixel_values=unpadded_pixels)
            text_features = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            unpadded_features = torch.nn.functional.normalize(unpadded_features, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            image_vecs.append(image_features.cpu().numpy())
            unpadded_vecs.append(unpadded_features.cpu().numpy())
            text_vecs.append(text_features.cpu().numpy())
            if (start // 8) % 25 == 0:
                print(f"Encoded {min(start + 8, len(rows))}/{len(rows)}", flush=True)
    images = np.concatenate(image_vecs)
    unpadded_images = np.concatenate(unpadded_vecs)
    texts = np.concatenate(text_vecs)
    rng = np.random.default_rng(42)
    result = {"checkpoint": "openai/clip-vit-large-patch14-336", "sample": "evaluation_manifest_seed42.csv",
              "image_preprocessing": "LLaVA-1.5-7B image_aspect_ratio=pad (square pad with CLIP mean), then CLIP 336 resize",
              "caption_rule": "one lowest-ID valid caption per image", "device": device,
              "warning": "CLIP cosine is a proxy, not visual grounding or LLaVA answer accuracy"}
    by_domain = {}
    detailed = []
    for dataset in ("COCO", "VizWiz"):
        ix = np.array([i for i, (row, caption) in enumerate(rows) if row["dataset"] == dataset and caption is not None])
        by_domain[dataset] = ix
        if len(ix) < 2:
            continue
        pair = (images[ix] * texts[ix]).sum(axis=1)
        shuffle = rng.permutation(len(ix))
        while np.any(shuffle == np.arange(len(ix))):
            shuffle = rng.permutation(len(ix))
        wrong = (images[ix] * texts[ix[shuffle]]).sum(axis=1)
        unpadded_pair = (unpadded_images[ix] * texts[ix]).sum(axis=1)
        for pos, index in enumerate(ix):
            item = rows[index][0]
            detailed.append({"dataset": dataset, "image_id": item["image_id"],
                             "text_detected": int(viz_flags[item["image_id"]]) if dataset == "VizWiz" else "",
                             "refusal_count": viz_refusals[item["image_id"]] if dataset == "VizWiz" else "",
                             "paired_cosine_pad": round(float(pair[pos]), 6),
                             "shuffled_cosine_pad": round(float(wrong[pos]), 6),
                             "paired_cosine_no_pad": round(float(unpadded_pair[pos]), 6)})
        result[dataset] = {"n": len(ix), "paired_cosine_mean": round(float(pair.mean()), 5),
                           "shuffled_cosine_mean": round(float(wrong.mean()), 5),
                           "paired_minus_shuffled": round(float((pair - wrong).mean()), 5),
                           "paired_cosine_no_pad_mean": round(float(unpadded_pair.mean()), 5)}
    all_coco = np.array([i for i, (row, _) in enumerate(rows) if row["dataset"] == "COCO"])
    all_viz = np.array([i for i, (row, _) in enumerate(rows) if row["dataset"] == "VizWiz"])
    c, v = images[all_coco], images[all_viz]
    n = min(len(c), len(v))
    c, v = c[:n], v[:n]
    result["image_domain"] = {"balanced_n_per_domain": n,
        "mean_cross_domain_cosine": round(float((c @ v.T).mean()), 5),
        "mean_within_coco_cosine": round(float(((c @ c.T).sum() - n) / (n * (n - 1))), 5),
        "mean_within_vizwiz_cosine": round(float(((v @ v.T).sum() - n) / (n * (n - 1))), 5),
        "centroid_cosine": round(float(np.dot(c.mean(0), v.mean(0)) / (np.linalg.norm(c.mean(0)) * np.linalg.norm(v.mean(0)))), 5)}
    unpad_c = unpadded_images[all_coco][:n]
    unpad_v = unpadded_images[all_viz][:n]
    result["image_domain_no_pad"] = {
        "balanced_n_per_domain": n,
        "mean_cross_domain_cosine": round(float((unpad_c @ unpad_v.T).mean()), 5),
        "mean_within_coco_cosine": round(float(((unpad_c @ unpad_c.T).sum() - n) / (n * (n - 1))), 5),
        "mean_within_vizwiz_cosine": round(float(((unpad_v @ unpad_v.T).sum() - n) / (n * (n - 1))), 5)}
    with (OUT / "clip_pair_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed[0])); writer.writeheader(); writer.writerows(detailed)
    strata = defaultdict(list)
    for item in detailed:
        if item["dataset"] == "VizWiz":
            strata[(min(3, item["refusal_count"]), item["text_detected"])].append(item)
    stratified = []
    for (refusal, text_detected), items in sorted(strata.items()):
        stratified.append({"refusal_count_capped": refusal, "text_detected": text_detected,
                           "images": len(items),
                           "paired_cosine_pad_mean": round(float(np.mean([r["paired_cosine_pad"] for r in items])), 5),
                           "shuffled_cosine_pad_mean": round(float(np.mean([r["shuffled_cosine_pad"] for r in items])), 5),
                           "paired_minus_shuffled_mean": round(float(np.mean([r["paired_cosine_pad"] - r["shuffled_cosine_pad"] for r in items])), 5)})
    with (OUT / "clip_alignment_by_vizwiz_stratum.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(stratified[0])); writer.writeheader(); writer.writerows(stratified)
    result["vizwiz_strata"] = stratified
    (OUT / "clip_embedding_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
