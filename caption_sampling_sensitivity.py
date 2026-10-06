"""Compare person references using the lowest-ID valid caption per image."""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from person_synonyms import person_mentions


ROOT = Path(__file__).resolve().parent


def select_one_per_image(annotations, eligible):
    grouped = defaultdict(list)
    for annotation in annotations:
        if eligible(annotation):
            grouped[annotation["image_id"]].append(annotation)
    return [min(grouped[image_id], key=lambda a: a["id"]) for image_id in sorted(grouped)]


def person_reference_rate(selected):
    mentions = 0
    matched = 0
    for ann in selected:
        count = person_mentions(ann['caption'])
        mentions += count
        matched += count > 0
    return mentions, matched


def main():
    datasets = [
        ("MS-COCO val2014", ROOT / "coco/annotations/captions_val2014.json", lambda a: True),
        ("VizWiz val", ROOT / "vizwiz/annotations/val.json",
         lambda a: not a.get("is_precanned") and not a.get("is_rejected")),
    ]
    rows = []
    for name, path, eligible in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        selected = select_one_per_image(data["annotations"], eligible)
        mentions, matched = person_reference_rate(selected)
        rows.append({
            "Dataset": name,
            "All Images": len(data["images"]),
            "Images with Eligible Caption": len(selected),
            "Images with No Eligible Caption": len(data["images"]) - len(selected),
            "Selected Captions": len(selected),
            "Person-family Mentions": mentions,
            "Selected Captions with Person Family": matched,
            "Mentions per 1,000 Selected Captions": round(mentions / len(selected) * 1000, 2),
            "Selected Captions with Person Family (%)": round(matched / len(selected) * 100, 2),
            "Selection Rule": "lowest valid annotation ID",
            "Matching Rule": "CHAIR person synonyms; longest phrase first; lexical, no POS filter",
        })
    output = ROOT / "eda_results/tables/person_reference_one_caption_per_image.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(output)
    import pandas as pd
    summary=[]
    for r in rows:
        summary.append({'Dataset':r['Dataset'],'Caption Count':r['Selected Captions'],'Image Count':r['All Images'],
                        'Noun Family':'CHAIR person synonyms (same as Figure 4)', 'Mentions':r['Person-family Mentions'],
                        'Mentions per 1,000 captions':r['Mentions per 1,000 Selected Captions'],
                        'Captions with family (%)':r['Selected Captions with Person Family (%)'],
                        'Images whose selected caption mentions family (%)':round(100*r['Selected Captions with Person Family']/r['All Images'],2)})
    pd.DataFrame(summary).to_csv(ROOT/'eda_results/tables/person_reference_summary.csv',index=False,encoding='utf-8-sig')
    metrics_path=ROOT/'eda_results/eda_summary_metrics.json'
    metrics=json.loads(metrics_path.read_text(encoding='utf-8'))
    metrics['person_reference_summary']=summary
    metrics_path.write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == "__main__":
    main()
