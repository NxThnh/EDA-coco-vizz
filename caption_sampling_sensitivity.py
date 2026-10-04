"""Compare person references using the lowest-ID valid caption per image."""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import nltk
from nltk import pos_tag_sents
from nltk.stem import WordNetLemmatizer


ROOT = Path(__file__).resolve().parent
nltk.data.path.insert(0, str(ROOT / "eda_resources" / "nltk_data"))
FAMILY = {"man", "men", "woman", "people", "person"}


def select_one_per_image(annotations, eligible):
    grouped = defaultdict(list)
    for annotation in annotations:
        if eligible(annotation):
            grouped[annotation["image_id"]].append(annotation)
    return [min(grouped[image_id], key=lambda a: a["id"]) for image_id in sorted(grouped)]


def person_reference_rate(selected):
    lemmatizer = WordNetLemmatizer()
    tokens = [re.findall(r"\b[a-z]+\b", a["caption"].lower()) for a in selected]
    mentions = 0
    matched = 0
    for tagged in pos_tag_sents(tokens, lang="eng"):
        count = sum(
            lemmatizer.lemmatize(word, pos="n") in FAMILY
            for word, tag in tagged
            if tag in {"NN", "NNS", "NNP", "NNPS"}
        )
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
            "Mentions per 1,000 Selected Captions": round(mentions / len(selected) * 1000, 2),
            "Selected Captions with Person Family (%)": round(matched / len(selected) * 100, 2),
            "Selection Rule": "lowest valid annotation ID",
        })
    output = ROOT / "eda_results/tables/person_reference_one_caption_per_image.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(output)


if __name__ == "__main__":
    main()
