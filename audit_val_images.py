#!/usr/bin/env python3
"""Decode every local val image and find byte/visual hash duplicates."""
import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eda_results/vlm_extensions"


def inspect(item):
    dataset, image_id, path = item
    try:
        md5 = hashlib.md5()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                md5.update(block)
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.load()
            gray = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
        dct = cv2.dct(gray)[:8, :8]
        bits = (dct > np.median(dct[1:, :])).reshape(-1)
        phash = f"{sum(int(bit) << i for i, bit in enumerate(bits)):016x}"
        return {"dataset": dataset, "image_id": image_id, "file_name": path.name, "decode_ok": 1,
                "md5": md5.hexdigest(), "phash": phash, "error": ""}
    except Exception as exc:
        return {"dataset": dataset, "image_id": image_id, "file_name": path.name, "decode_ok": 0,
                "md5": "", "phash": "", "error": str(exc)[:160]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for dataset, ann, folder in (("COCO", "coco/annotations/captions_val2014.json", "coco/val2014"),
                                 ("VizWiz", "vizwiz/annotations/val.json", "vizwiz/val")):
        data = json.loads((ROOT / ann).read_text(encoding="utf-8"))
        items.extend((dataset, image["id"], ROOT / folder / image["file_name"]) for image in data["images"])
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(inspect, items))
    with (OUT / "val_image_hashes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    duplicates = {}
    for method in ("md5", "phash"):
        groups = defaultdict(list)
        for row in rows:
            if row[method]:
                groups[row[method]].append(f"{row['dataset']}:{row['image_id']}")
        duplicates[method] = [{"hash": key, "images": value} for key, value in groups.items() if len(value) > 1]
    summary = {"images": len(rows), "decode_failures": sum(not row["decode_ok"] for row in rows),
               "md5_duplicate_groups": len(duplicates["md5"]), "phash_exact_duplicate_groups": len(duplicates["phash"]),
               "note": "Exact pHash equality only; near matches require a Hamming-distance threshold and manual review."}
    (OUT / "val_image_audit.json").write_text(json.dumps({"summary": summary, "duplicates": duplicates}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
