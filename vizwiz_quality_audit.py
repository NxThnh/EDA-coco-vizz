#!/usr/bin/env python3
"""Join official VizWiz-QualityIssues val votes to caption and Laplacian EDA."""
import csv
import hashlib
import json
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "scratch/vizwiz_quality_annotations.zip"
OUT = ROOT / "eda_results/vlm_extensions"
FLAWS = {"BLR": "blur", "BRT": "too_bright", "DRK": "too_dark", "FRM": "poor_framing",
         "OBS": "obstructed", "ROT": "rotated", "OTH": "other", "NON": "none"}


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Download VizWiz-QualityIssues annotations.zip to {SOURCE}")
    with zipfile.ZipFile(SOURCE) as source:
        quality = json.loads(source.read("val.json"))
    captions = json.loads((ROOT / "vizwiz/annotations/val.json").read_text(encoding="utf-8"))
    by_id = defaultdict(list)
    for ann in captions["annotations"]:
        by_id[ann["image_id"]].append(ann)
    by_file = {image["file_name"]: image["id"] for image in captions["images"]}
    laps = {int(row["image_id"]): float(row["laplacian_variance_480px"])
            for row in csv.DictReader((ROOT / "eda_results/tables/laplacian_full_val.csv").open(encoding="utf-8-sig"))
            if row["dataset"] == "VizWiz"}
    rows = []
    for item in quality:
        file_name = item["image"]
        image_id = by_file[file_name]
        refusal = sum(a.get("is_precanned", False) for a in by_id[image_id])
        row = {"image_id": image_id, "file_name": file_name, "refusal_captions": refusal,
               "unrecognizable_votes": item["unrecognizable"], "laplacian_variance_480px": laps[image_id]}
        row.update({f"{name}_votes": item["flaws"][code] for code, name in FLAWS.items()})
        rows.append(row)
    assert len(rows) == len(by_file) == len(laps) == 7750
    rows.sort(key=lambda row: row["image_id"])
    with (OUT / "vizwiz_quality_issue_votes_val.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    vote_summary = {}
    for name in FLAWS.values():
        selected = [row for row in rows if row[f"{name}_votes"] >= 3]
        vote_summary[name] = {"majority_images": len(selected), "majority_pct": round(100 * len(selected) / len(rows), 2),
                              "median_laplacian": round(statistics.median(row["laplacian_variance_480px"] for row in selected), 2) if selected else None}
    by_refusal = []
    for group in range(6):
        selected = [row for row in rows if row["refusal_captions"] == group]
        by_refusal.append({"refusal_captions": group, "images": len(selected),
                           "blur_majority_pct": round(100 * sum(row["blur_votes"] >= 3 for row in selected) / len(selected), 2),
                           "poor_framing_majority_pct": round(100 * sum(row["poor_framing_votes"] >= 3 for row in selected) / len(selected), 2),
                           "dark_majority_pct": round(100 * sum(row["too_dark_votes"] >= 3 for row in selected) / len(selected), 2),
                           "unrecognizable_majority_pct": round(100 * sum(row["unrecognizable_votes"] >= 3 for row in selected) / len(selected), 2)})
    with (OUT / "vizwiz_quality_by_refusal.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_refusal[0])); writer.writeheader(); writer.writerows(by_refusal)
    result = {"source": "https://vizwiz.cs.colorado.edu/VizWiz_final/image_quality/annotations.zip",
              "source_zip_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "matched_val_images": len(rows), "majority_threshold": "at least 3 of 5 votes",
              "flaws": vote_summary, "by_refusal": by_refusal}
    (OUT / "vizwiz_quality_issue_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
