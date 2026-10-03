#!/usr/bin/env python3
"""
EDA: COCO 2014 vs VizWiz Captions Comparison
Graduation Thesis Data Analysis & Domain Gap Investigation

Authors: Graduation Thesis Project
Workspace: c:/Users/Admin/Downloads/HGAI_data
"""

import os
import json
import re
import time
import textwrap
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import nltk
from nltk import pos_tag_sents
from nltk.stem import WordNetLemmatizer
from PIL import Image, ImageOps

# Set random seed for reproducibility
np.random.seed(42)

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COCO_DIR = os.path.join(BASE_DIR, "coco")
VIZWIZ_DIR = os.path.join(BASE_DIR, "vizwiz")
OUT_DIR = os.path.join(BASE_DIR, "eda_results")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# Vendored NLTK resources make POS-based entity extraction reproducible offline.
NLTK_DATA_DIR = os.path.join(BASE_DIR, "eda_resources", "nltk_data")
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

print("="*70)
print("STARTING COMPREHENSIVE EDA: COCO vs VIZWIZ")
print("="*70)

# ==============================================================================
# 1. LOAD DATA & DATASET SIZES
# ==============================================================================
print("\n[1/7] Loading JSON annotations & computing split statistics...")

# VizWiz annotations
with open(os.path.join(VIZWIZ_DIR, "annotations", "train.json"), "r", encoding="utf-8") as f:
    viz_train = json.load(f)
with open(os.path.join(VIZWIZ_DIR, "annotations", "val.json"), "r", encoding="utf-8") as f:
    viz_val = json.load(f)
with open(os.path.join(VIZWIZ_DIR, "annotations", "test.json"), "r", encoding="utf-8") as f:
    viz_test = json.load(f)

# COCO annotations
with open(os.path.join(COCO_DIR, "annotations", "captions_train2014.json"), "r", encoding="utf-8") as f:
    coco_train_cap = json.load(f)
with open(os.path.join(COCO_DIR, "annotations", "captions_val2014.json"), "r", encoding="utf-8") as f:
    coco_val_cap = json.load(f)
with open(os.path.join(COCO_DIR, "annotations", "instances_val2014.json"), "r", encoding="utf-8") as f:
    coco_val_inst = json.load(f)

split_stats = [
    {
        "Dataset": "COCO 2014",
        "Split": "Train",
        "Images": len(coco_train_cap["images"]),
        "Captions": len(coco_train_cap["annotations"]),
        "Captions/Img": round(len(coco_train_cap["annotations"]) / len(coco_train_cap["images"]), 2)
    },
    {
        "Dataset": "COCO 2014",
        "Split": "Val",
        "Images": len(coco_val_cap["images"]),
        "Captions": len(coco_val_cap["annotations"]),
        "Captions/Img": round(len(coco_val_cap["annotations"]) / len(coco_val_cap["images"]), 2)
    },
    {
        "Dataset": "COCO 2014",
        "Split": "Total",
        "Images": len(coco_train_cap["images"]) + len(coco_val_cap["images"]),
        "Captions": len(coco_train_cap["annotations"]) + len(coco_val_cap["annotations"]),
        "Captions/Img": round((len(coco_train_cap["annotations"]) + len(coco_val_cap["annotations"])) / (len(coco_train_cap["images"]) + len(coco_val_cap["images"])), 2)
    },
    {
        "Dataset": "VizWiz",
        "Split": "Train",
        "Images": len(viz_train["images"]),
        "Captions": len(viz_train["annotations"]),
        "Captions/Img": round(len(viz_train["annotations"]) / len(viz_train["images"]), 2)
    },
    {
        "Dataset": "VizWiz",
        "Split": "Val",
        "Images": len(viz_val["images"]),
        "Captions": len(viz_val["annotations"]),
        "Captions/Img": round(len(viz_val["annotations"]) / len(viz_val["images"]), 2)
    },
    {
        "Dataset": "VizWiz",
        "Split": "Test",
        "Images": len(viz_test["images"]),
        "Captions": 0,  # Blind test split
        "Captions/Img": 0.0
    },
    {
        "Dataset": "VizWiz",
        "Split": "Total",
        "Images": len(viz_train["images"]) + len(viz_val["images"]) + len(viz_test["images"]),
        "Captions": len(viz_train["annotations"]) + len(viz_val["annotations"]),
        "Captions/Img": round((len(viz_train["annotations"]) + len(viz_val["annotations"])) / (len(viz_train["images"]) + len(viz_val["images"])), 2)
    }
]

df_splits = pd.DataFrame(split_stats)
df_splits.to_csv(os.path.join(TAB_DIR, "dataset_overview.csv"), index=False)
print("Splits table:")
print(df_splits)

# FIGURE 1: Dataset Splits Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
colors_coco = ['#3498db', '#2980b9']
colors_viz = ['#e74c3c', '#c0392b', '#962d22']

splits_labels = ['Train', 'Val', 'Test']
coco_img_counts = [len(coco_train_cap["images"]), len(coco_val_cap["images"]), 0]
viz_img_counts = [len(viz_train["images"]), len(viz_val["images"]), len(viz_test["images"])]

x = np.arange(len(splits_labels))
width = 0.35

bars1 = ax1.bar(x - width/2, coco_img_counts, width, label='MS-COCO 2014', color='#3498db', edgecolor='black', linewidth=0.8)
bars2 = ax1.bar(x + width/2, viz_img_counts, width, label='VizWiz-Captions', color='#e67e22', edgecolor='black', linewidth=0.8)

ax1.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
ax1.set_title('(A) Image Count by Split', fontsize=13, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(splits_labels, fontsize=11)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

for b in bars1:
    h = b.get_height()
    if h > 0:
        ax1.annotate(f'{h:,}', xy=(b.get_x() + b.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for b in bars2:
    h = b.get_height()
    if h > 0:
        ax1.annotate(f'{h:,}', xy=(b.get_x() + b.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Captions comparison
splits_caps = ['Train', 'Val']
coco_cap_counts = [len(coco_train_cap["annotations"]), len(coco_val_cap["annotations"])]
viz_cap_counts = [len(viz_train["annotations"]), len(viz_val["annotations"])]

x_cap = np.arange(len(splits_caps))
bars3 = ax2.bar(x_cap - width/2, coco_cap_counts, width, label='MS-COCO 2014', color='#2980b9', edgecolor='black', linewidth=0.8)
bars4 = ax2.bar(x_cap + width/2, viz_cap_counts, width, label='VizWiz-Captions', color='#d35400', edgecolor='black', linewidth=0.8)

ax2.set_ylabel('Number of Captions', fontsize=12, fontweight='bold')
ax2.set_title('(B) Caption Count by Split', fontsize=13, fontweight='bold', pad=10)
ax2.set_xticks(x_cap)
ax2.set_xticklabels(splits_caps, fontsize=11)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

for b in bars3:
    h = b.get_height()
    ax2.annotate(f'{h:,}', xy=(b.get_x() + b.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for b in bars4:
    h = b.get_height()
    ax2.annotate(f'{h:,}', xy=(b.get_x() + b.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig_path1 = os.path.join(FIG_DIR, "01_dataset_splits_comparison.png")
plt.savefig(fig_path1, dpi=300)
plt.close()
print(f"Saved: {fig_path1}")

# ==============================================================================
# 2. RESOLUTION & ASPECT RATIO ANALYSIS
# ==============================================================================
print("\n[2/7] Analyzing Image Resolution & Aspect Ratio...")

# COCO Val dimensions (from annotations)
coco_w = np.array([img["width"] for img in coco_val_cap["images"]])
coco_h = np.array([img["height"] for img in coco_val_cap["images"]])
coco_ar = coco_w / coco_h
coco_mp = (coco_w * coco_h) / 1e6

# VizWiz Val dimensions (from physical files)
viz_val_dir = os.path.join(VIZWIZ_DIR, "val")
viz_files = [img["file_name"] for img in viz_val["images"]]
print(f"Scanning image headers for {len(viz_files)} VizWiz val images...")

viz_w_list, viz_h_list = [], []
for fname in viz_files:
    fpath = os.path.join(viz_val_dir, fname)
    with Image.open(fpath) as img:
        w, h = ImageOps.exif_transpose(img).size
        viz_w_list.append(w)
        viz_h_list.append(h)

viz_w = np.array(viz_w_list)
viz_h = np.array(viz_h_list)
viz_ar = viz_w / viz_h
viz_mp = (viz_w * viz_h) / 1e6

def classify_orientation(ars):
    land = np.sum(ars > 1.05)
    port = np.sum(ars < 0.95)
    sq = len(ars) - land - port
    return land, port, sq

coco_land, coco_port, coco_sq = classify_orientation(coco_ar)
viz_land, viz_port, viz_sq = classify_orientation(viz_ar)

res_stats = [
    {
        "Dataset": "MS-COCO val2014",
        "N": len(coco_w),
        "Width Mean": round(float(np.mean(coco_w)), 1),
        "Width Median": float(np.median(coco_w)),
        "Width Std": round(float(np.std(coco_w)), 1),
        "Width Min": int(np.min(coco_w)),
        "Width Max": int(np.max(coco_w)),
        "Height Mean": round(float(np.mean(coco_h)), 1),
        "Height Median": float(np.median(coco_h)),
        "Height Std": round(float(np.std(coco_h)), 1),
        "Height Min": int(np.min(coco_h)),
        "Height Max": int(np.max(coco_h)),
        "AR Mean": round(float(np.mean(coco_ar)), 2),
        "AR Median": round(float(np.median(coco_ar)), 2),
        "Landscape (%)": round(coco_land / len(coco_ar) * 100, 2),
        "Portrait (%)": round(coco_port / len(coco_ar) * 100, 2),
        "Square (%)": round(coco_sq / len(coco_ar) * 100, 2),
        "MP Mean": round(float(np.mean(coco_mp)), 3)
    },
    {
        "Dataset": "VizWiz-Captions",
        "N": len(viz_w),
        "Width Mean": round(float(np.mean(viz_w)), 1),
        "Width Median": float(np.median(viz_w)),
        "Width Std": round(float(np.std(viz_w)), 1),
        "Width Min": int(np.min(viz_w)),
        "Width Max": int(np.max(viz_w)),
        "Height Mean": round(float(np.mean(viz_h)), 1),
        "Height Median": float(np.median(viz_h)),
        "Height Std": round(float(np.std(viz_h)), 1),
        "Height Min": int(np.min(viz_h)),
        "Height Max": int(np.max(viz_h)),
        "AR Mean": round(float(np.mean(viz_ar)), 2),
        "AR Median": round(float(np.median(viz_ar)), 2),
        "Landscape (%)": round(viz_land / len(viz_ar) * 100, 2),
        "Portrait (%)": round(viz_port / len(viz_ar) * 100, 2),
        "Square (%)": round(viz_sq / len(viz_ar) * 100, 2),
        "MP Mean": round(float(np.mean(viz_mp)), 3)
    }
]

df_res = pd.DataFrame(res_stats)
df_res.to_csv(os.path.join(TAB_DIR, "resolution_aspect_ratio.csv"), index=False)
print("Resolution & Aspect Ratio stats:")
print(df_res[["Dataset", "Width Mean", "Height Mean", "AR Mean", "Landscape (%)", "Portrait (%)", "MP Mean"]])

# FIGURE 2: Resolution & Aspect Ratio Comparison
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.28, wspace=0.22)

# (A) Aspect Ratio KDE Distribution
ax_ar = fig.add_subplot(gs[0, 0])
sns.kdeplot(coco_ar, ax=ax_ar, label=f'COCO Val (Median={np.median(coco_ar):.2f})', color='#2980b9', fill=True, alpha=0.35, linewidth=2)
sns.kdeplot(viz_ar, ax=ax_ar, label=f'VizWiz Val (Median={np.median(viz_ar):.2f})', color='#e67e22', fill=True, alpha=0.35, linewidth=2)
ax_ar.axvline(1.0, color='gray', linestyle=':', linewidth=1.5, label='Square (1.0)')
ax_ar.set_xlim(0.2, 2.5)
ax_ar.set_xlabel('Aspect Ratio (Width / Height)', fontsize=11, fontweight='bold')
ax_ar.set_ylabel('Density', fontsize=11, fontweight='bold')
ax_ar.set_title('(A) Aspect Ratio Distribution (Landscape vs Portrait Shift)', fontsize=12, fontweight='bold')
ax_ar.legend(frameon=True, facecolor='white', framealpha=0.9)

# (B) Orientation Breakdown Pie Charts
ax_pie = fig.add_subplot(gs[0, 1])
pie_labels = ['Landscape\n(W > H)', 'Portrait\n(H > W)', 'Square']
pie_colors = ['#3498db', '#e67e22', '#95a5a6']

coco_pcts = [coco_land/len(coco_ar), coco_port/len(coco_ar), coco_sq/len(coco_ar)]
viz_pcts = [viz_land/len(viz_ar), viz_port/len(viz_ar), viz_sq/len(viz_ar)]

# Nested donut chart: label the large sectors in a legend to avoid collisions.
ax_pie.pie(coco_pcts, radius=1.0, colors=pie_colors,
           wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1.5))
ax_pie.pie(viz_pcts, radius=0.68, colors=pie_colors,
           wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1.5))
ax_pie.set_title('(B) Orientation: Outer Ring = COCO | Inner Ring = VizWiz', fontsize=12, fontweight='bold')
legend_patches = [mpatches.Patch(color=c, label=f'{l}: COCO {cp:.1%}; VizWiz {vp:.1%}')
                  for c, l, cp, vp in zip(pie_colors, ['Landscape', 'Portrait', 'Square'], coco_pcts, viz_pcts)]
ax_pie.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.22), frameon=True, fontsize=8)

# (C) Width vs Height Scatter / Hexbin
ax_sc1 = fig.add_subplot(gs[1, 0])
idx_coco = np.random.choice(len(coco_w), size=min(4000, len(coco_w)), replace=False)
ax_sc1.scatter(coco_w[idx_coco], coco_h[idx_coco], alpha=0.25, s=12, color='#2980b9', edgecolors='none')
ax_sc1.set_xlabel('Width (pixels)', fontsize=11, fontweight='bold')
ax_sc1.set_ylabel('Height (pixels)', fontsize=11, fontweight='bold')
ax_sc1.set_title(f'(C1) COCO Val Resolutions (Bounded ~640px Max)', fontsize=12, fontweight='bold')
ax_sc1.set_xlim(0, 700)
ax_sc1.set_ylim(0, 700)

ax_sc2 = fig.add_subplot(gs[1, 1])
unique_res, res_counts = np.unique(np.column_stack((viz_w, viz_h)), axis=0, return_counts=True)
points = ax_sc2.scatter(unique_res[:, 0], unique_res[:, 1],
                        s=12 + 9 * np.sqrt(res_counts), c=np.log10(res_counts),
                        cmap='YlOrRd', alpha=0.8, edgecolors='#a04000', linewidths=0.3)
ax_sc2.set_xlabel('Width (pixels)', fontsize=11, fontweight='bold')
ax_sc2.set_ylabel('Height (pixels)', fontsize=11, fontweight='bold')
ax_sc2.set_title('(C2) VizWiz Val Resolutions (EXIF Corrected)', fontsize=12, fontweight='bold')
ax_sc2.set_xlim(0, 3500)
ax_sc2.set_ylim(0, 3500)
ax_sc2.text(0.02, 0.98, f'n={len(viz_w):,}; {len(unique_res):,} distinct sizes; bubble area ∝ count', transform=ax_sc2.transAxes,
            va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

plt.tight_layout()
fig_path2 = os.path.join(FIG_DIR, "02_resolution_aspect_ratio.png")
plt.savefig(fig_path2, dpi=300)
plt.close()
print(f"Saved: {fig_path2}")

# ==============================================================================
# 3. CAPTION LENGTH, VOCABULARY & LINGUISTIC ANALYSIS
# ==============================================================================
print("\n[3/7] Analyzing Captions, Vocabulary & Lexical Diversity...")

coco_captions = [a["caption"].strip() for a in coco_val_cap["annotations"]]
# Placeholder captions are quality labels, not natural-language descriptions.
# Exclude them from every linguistic/token/entity statistic below.
viz_captions_norm = [a["caption"].strip() for a in viz_val["annotations"] if not a.get("is_precanned")]

def tokenize_caption(caption):
    return re.findall(r'\b[a-z0-9]+\b', caption.lower())


def moving_average_ttr(tokens, window=1000):
    """MATTR with a fixed token window, comparable across corpus sizes."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    counts = Counter(tokens[:window])
    unique = len(counts)
    total = unique / window
    n_windows = 1
    for old, new in zip(tokens, tokens[window:]):
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
            unique -= 1
        if counts[new] == 0:
            unique += 1
        counts[new] += 1
        total += unique / window
        n_windows += 1
    return total / n_windows


def compute_caption_stats(name, caps):
    word_lens = [len(c.split()) for c in caps]
    char_lens = [len(c) for c in caps]
    tokens = [t for c in caps for t in tokenize_caption(c)]
    vocab = set(tokens)
    ttr = len(vocab) / len(tokens) if len(tokens) > 0 else 0
    return {
        "Subset": name,
        "Total Captions": len(caps),
        "Word Mean": round(float(np.mean(word_lens)), 2),
        "Word Median": float(np.median(word_lens)),
        "Word Std": round(float(np.std(word_lens)), 2),
        "Word Min": int(np.min(word_lens)),
        "Word Max": int(np.max(word_lens)),
        "Char Mean": round(float(np.mean(char_lens)), 2),
        "Char Median": float(np.median(char_lens)),
        "Total Tokens": len(tokens),
        "Vocab Size": len(vocab),
        "TTR (corpus-size dependent)": round(ttr, 4),
        "MATTR (window=1000)": round(moving_average_ttr(tokens, window=1000), 4)
    }

cap_stats = [
    compute_caption_stats("MS-COCO val2014", coco_captions),
    compute_caption_stats("VizWiz-Captions (placeholder removed)", viz_captions_norm),
]

df_caps = pd.DataFrame(cap_stats)
df_caps.to_csv(os.path.join(TAB_DIR, "caption_stats.csv"), index=False)
print("Caption stats:")
print(df_caps[["Subset", "Total Captions", "Word Mean", "Word Median", "Word Max", "Vocab Size", "MATTR (window=1000)"]])

# Word length arrays for plotting
coco_wlens = [len(c.split()) for c in coco_captions]
viz_wlens_norm = [len(c.split()) for c in viz_captions_norm]

# POS-based noun extraction. Generic image-reference nouns are excluded explicitly;
# counts, rates and the exact filtering rule are exported for auditability.
generic_image_nouns = {
    'image', 'photo', 'photograph', 'picture',
    'top', 'front', 'side', 'background', 'foreground', 'center', 'middle'
}
lemmatizer = WordNetLemmatizer()

def get_noun_entities(caps, batch_size=2000):
    counts = Counter()
    for start in range(0, len(caps), batch_size):
        tokenized = [re.findall(r'\b[a-z]+\b', c.lower()) for c in caps[start:start + batch_size]]
        for tagged in pos_tag_sents(tokenized, lang='eng'):
            for token, tag in tagged:
                if tag in {'NN', 'NNS', 'NNP', 'NNPS'}:
                    lemma = lemmatizer.lemmatize(token, pos='n')
                    if len(lemma) >= 2 and lemma not in generic_image_nouns:
                        counts[lemma] += 1
    return counts

print("Extracting caption nouns with NLTK POS tagging...")
coco_noun_counts = get_noun_entities(coco_captions)
viz_noun_counts = get_noun_entities(viz_captions_norm)
top_coco = coco_noun_counts.most_common(15)
top_viz_norm = viz_noun_counts.most_common(15)

entity_rows = []
for dataset, counts, n_caps in [
    ('COCO val2014 captions', coco_noun_counts, len(coco_captions)),
    ('VizWiz val descriptive captions', viz_noun_counts, len(viz_captions_norm)),
]:
    for rank, (entity, count) in enumerate(counts.most_common(15), start=1):
        entity_rows.append({
            'Dataset': dataset,
            'Rank': rank,
            'Entity': entity,
            'Count': count,
            'Mentions per 1,000 captions': round(count / n_caps * 1000, 2),
            'Extraction': 'NLTK noun POS + WordNet lemma; generic image/location nouns excluded'
        })
df_caption_entities = pd.DataFrame(entity_rows)
df_caption_entities.to_csv(os.path.join(TAB_DIR, 'caption_noun_entities_top15.csv'), index=False)

# FIGURE 3: Caption Length & Vocabulary Comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# (A) Caption Word Count Histogram/KDE
sns.histplot(coco_wlens, ax=ax1, color='#2980b9', stat='density', binwidth=1, label=f'COCO Val (Mean={np.mean(coco_wlens):.1f})', alpha=0.4, kde=True)
sns.histplot(viz_wlens_norm, ax=ax1, color='#e67e22', stat='density', binwidth=1, label=f'VizWiz Descriptive (Mean={np.mean(viz_wlens_norm):.1f})', alpha=0.4, kde=True)
ax1.set_xlim(3, 30)
ax1.set_xlabel('Caption Word Count', fontsize=11, fontweight='bold')
ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
ax1.set_title('(A) Caption Word-count Distribution', fontsize=12, fontweight='bold')
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

# (B) Boxplot Comparison
data_box = [coco_wlens, viz_wlens_norm]
labels_box = ['COCO Val', 'VizWiz (placeholder removed)']
colors_box = ['#3498db', '#d35400']
bplot = ax2.boxplot(data_box, patch_artist=True, labels=labels_box, showfliers=False, medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bplot['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax2.set_ylabel('Caption Word Count (outlier markers hidden)', fontsize=11, fontweight='bold')
ax2.set_title('(B) Word Count Summary Quartiles', fontsize=12, fontweight='bold')

# (C) Top Keywords COCO
words_c, counts_c = zip(*top_coco[::-1])
rates_c = [count / len(coco_captions) * 1000 for count in counts_c]
ax3.barh(words_c, rates_c, color='#2980b9', edgecolor='black', linewidth=0.7)
ax3.set_xlabel('Noun mentions per 1,000 captions', fontsize=11, fontweight='bold')
ax3.set_title('(C) Top 15 POS-tagged Nouns — COCO Val Captions', fontsize=12, fontweight='bold')

# (D) Top Keywords VizWiz Descriptive
words_v, counts_v = zip(*top_viz_norm[::-1])
rates_v = [count / len(viz_captions_norm) * 1000 for count in counts_v]
ax4.barh(words_v, rates_v, color='#e67e22', edgecolor='black', linewidth=0.7)
ax4.set_xlabel('Noun mentions per 1,000 descriptive captions', fontsize=11, fontweight='bold')
ax4.set_title('(D) Top 15 POS-tagged Nouns — VizWiz Val Captions', fontsize=12, fontweight='bold')

plt.tight_layout()
fig_path3 = os.path.join(FIG_DIR, "03_caption_length_and_tokens.png")
plt.savefig(fig_path3, dpi=300)
plt.close()
print(f"Saved: {fig_path3}")

# ==============================================================================
# 4. CAPTIONS PER IMAGE & COCO CATEGORY DISTRIBUTION
# ==============================================================================
print("\n[4/7] Analyzing Captions per Image & COCO Object Categories...")

coco_img_caps = Counter([a["image_id"] for a in coco_val_cap["annotations"]])
viz_img_caps = Counter([a["image_id"] for a in viz_val["annotations"]])

coco_caps_per_img = Counter(coco_img_caps.values())
viz_caps_per_img = Counter(viz_img_caps.values())
print(f"COCO captions per image: {coco_caps_per_img}")
print(f"VizWiz captions per image: {viz_caps_per_img}")

# COCO Categories
cats = {c["id"]: c["name"] for c in coco_val_inst["categories"]}
supercats = {c["id"]: c["supercategory"] for c in coco_val_inst["categories"]}

cat_counts = Counter([cats[a["category_id"]] for a in coco_val_inst["annotations"]])
supercat_counts = Counter([supercats[a["category_id"]] for a in coco_val_inst["annotations"]])

total_inst = len(coco_val_inst["annotations"])
df_coco_cats = pd.DataFrame([
    {"Category": name, "Count": count, "Percentage": round(count / total_inst * 100, 2)}
    for name, count in cat_counts.most_common()
])
df_coco_cats.to_csv(os.path.join(TAB_DIR, "coco_categories.csv"), index=False)

df_coco_super = pd.DataFrame([
    {"Supercategory": name, "Count": count, "Percentage": round(count / total_inst * 100, 2)}
    for name, count in supercat_counts.most_common()
])
df_coco_super.to_csv(os.path.join(TAB_DIR, "coco_supercategories.csv"), index=False)

# FIGURE 4: COCO Category Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

# Top 15 categories
top_15 = cat_counts.most_common(15)
cat_names = [c[0] for c in top_15][::-1]
cat_vals = [c[1] for c in top_15][::-1]
cat_pcts = [c[1] / total_inst * 100 for c in top_15][::-1]

colors_bar = ['#e74c3c' if name == 'person' else '#3498db' for name in cat_names]
bars = ax1.barh(cat_names, cat_vals, color=colors_bar, edgecolor='black', linewidth=0.7)
ax1.set_xlabel('Instance Count in Val (Total = 291,875)', fontsize=11, fontweight='bold')
ax1.set_title('(A) Top 15 COCO Object Categories (Person Dominance)', fontsize=12, fontweight='bold')
for b, pct in zip(bars, cat_pcts):
    w = b.get_width()
    ax1.annotate(f'{pct:.1f}% ({int(w):,})', xy=(w, b.get_y() + b.get_height()/2),
                 xytext=(5, 0), textcoords="offset points", va='center', fontsize=9, fontweight='bold')
ax1.set_xlim(0, max(cat_vals) * 1.25)

# Supercategories
super_sorted = supercat_counts.most_common()
s_names = [s[0] for s in super_sorted][::-1]
s_vals = [s[1] for s in super_sorted][::-1]
s_pcts = [s[1] / total_inst * 100 for s in super_sorted][::-1]

bars_s = ax2.barh(s_names, s_vals, color='#2ecc71', edgecolor='black', linewidth=0.7)
ax2.set_xlabel('Instance Count', fontsize=11, fontweight='bold')
ax2.set_title('(B) COCO 12 Supercategories Distribution', fontsize=12, fontweight='bold')
for b, pct in zip(bars_s, s_pcts):
    w = b.get_width()
    ax2.annotate(f'{pct:.1f}%', xy=(w, b.get_y() + b.get_height()/2),
                 xytext=(5, 0), textcoords="offset points", va='center', fontsize=9, fontweight='bold')
ax2.set_xlim(0, max(s_vals) * 1.2)

plt.tight_layout()
fig_path4 = os.path.join(FIG_DIR, "04_coco_category_distribution.png")
plt.savefig(fig_path4, dpi=300)
plt.close()
print(f"Saved: {fig_path4}")

# ==============================================================================
# 5. VIZWIZ QUALITY ISSUES & TEXT DETECTION ANALYSIS
# ==============================================================================
print("\n[5/7] Analyzing VizWiz Quality Issues & Low-Quality Rate...")

# Annotations level
n_viz_annos = len(viz_val["annotations"])
n_precanned = sum(1 for a in viz_val["annotations"] if a.get("is_precanned"))
n_rejected = sum(1 for a in viz_val["annotations"] if a.get("is_rejected"))
n_text_anno = sum(1 for a in viz_val["annotations"] if a.get("text_detected"))

# Image level
img_precanned = defaultdict(int)
img_rejected = defaultdict(int)
img_text = defaultdict(int)

for a in viz_val["annotations"]:
    iid = a["image_id"]
    if a.get("is_precanned"):
        img_precanned[iid] += 1
    if a.get("is_rejected"):
        img_rejected[iid] += 1
    if a.get("text_detected"):
        img_text[iid] += 1

all_img_ids = [img["id"] for img in viz_val["images"]]
total_viz_imgs = len(all_img_ids)

precanned_hist = Counter([img_precanned[i] for i in all_img_ids])
text_hist = Counter([img_text[i] for i in all_img_ids])

df_quality = pd.DataFrame([
    {
        "Precanned Count (/5)": k,
        "Num Images": precanned_hist[k],
        "Percentage": round(precanned_hist[k] / total_viz_imgs * 100, 2),
        "Interpretation": {
            0: "Five descriptive captions",
            1: "One refusal; four descriptive captions",
            2: "Two refusals; three descriptive captions",
            3: "Three refusals; two descriptive captions",
            4: "Four refusals; one descriptive caption",
            5: "Five refusal captions"
        }.get(k, "")
    }
    for k in range(6)
])
df_quality.to_csv(os.path.join(TAB_DIR, "vizwiz_quality_breakdown.csv"), index=False)
print("VizWiz Quality breakdown:")
print(df_quality)

# Measure physical blur and brightness on VizWiz Val vs COCO Val sample
print("Computing Laplacian variance (blur) and luminance on representative samples...")
def get_image_metrics(img_path, short_edge=480):
    with Image.open(img_path) as raw_img:
        img = ImageOps.exif_transpose(raw_img).convert('L')
        # Preserve aspect ratio and normalize spatial scale before comparison.
        scale = short_edge / min(img.size)
        new_size = tuple(max(1, int(round(v * scale))) for v in img.size)
        img_norm = img.resize(new_size, Image.Resampling.LANCZOS)
        arr = np.array(img_norm, dtype=np.float32)
        # Discrete Laplacian [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        lap = arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:] - 4*arr[1:-1, 1:-1]
        blur = float(lap.var())
        brightness = float(arr.mean())
        contrast = float(arr.std())
        return blur, brightness, contrast

# Deterministic random sample of 1000 images from each validation split.
coco_val_dir = os.path.join(COCO_DIR, "val2014")
rng = np.random.default_rng(42)
coco_files_all = [img["file_name"] for img in coco_val_cap["images"]]
viz_files_all = [img["file_name"] for img in viz_val["images"]]
coco_sample_files = list(rng.choice(coco_files_all, size=1000, replace=False))
viz_sample_files = list(rng.choice(viz_files_all, size=1000, replace=False))

coco_blurs, coco_brights = [], []
for fn in coco_sample_files:
    fp = os.path.join(coco_val_dir, fn)
    b, br, _ = get_image_metrics(fp)
    coco_blurs.append(b)
    coco_brights.append(br)

viz_blurs, viz_brights = [], []
for fn in viz_sample_files:
    fp = os.path.join(viz_val_dir, fn)
    b, br, _ = get_image_metrics(fp)
    viz_blurs.append(b)
    viz_brights.append(br)

def bootstrap_median_ci(values, seed=42, repeats=3000):
    values = np.asarray(values, dtype=float)
    rng_ci = np.random.default_rng(seed)
    medians = np.array([np.median(values[rng_ci.integers(0, len(values), len(values))])
                        for _ in range(repeats)])
    return np.quantile(medians, [0.025, 0.975]).tolist()

coco_lap_ci = bootstrap_median_ci(coco_blurs)
viz_lap_ci = bootstrap_median_ci(viz_blurs)
viz_file_ids = {img['file_name']: img['id'] for img in viz_val['images']}
sample_group_values = {k: [] for k in range(6)}
for filename, lap_value in zip(viz_sample_files, viz_blurs):
    sample_group_values[img_precanned[viz_file_ids[filename]]].append(lap_value)
lap_by_precanned = [
    {'Precanned Count (/5)': k, 'Sample Images': len(sample_group_values[k]),
     'Median Laplacian Variance': float(np.median(sample_group_values[k])) if sample_group_values[k] else None,
     'CI 95% Low': bootstrap_median_ci(sample_group_values[k], seed=42+k)[0] if sample_group_values[k] else None,
     'CI 95% High': bootstrap_median_ci(sample_group_values[k], seed=42+k)[1] if sample_group_values[k] else None}
    for k in range(6)
]
pd.DataFrame(lap_by_precanned).to_csv(os.path.join(TAB_DIR, 'laplacian_by_precanned.csv'), index=False)
print('Laplacian by precanned consensus (descriptive validation):')
print(pd.DataFrame(lap_by_precanned).to_string(index=False))

# FIGURE 5: VizWiz Quality Issues & Text Detection
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# (A) Quality Degradation Bar
prec_vals = [precanned_hist[k] for k in range(6)]
prec_pcts = [precanned_hist[k] / total_viz_imgs * 100 for k in range(6)]
colors_q = ['#27ae60', '#f39c12', '#e67e22', '#d35400', '#c0392b', '#7f1d1d']
bars_q = ax1.bar([f'{k}/5' for k in range(6)], prec_vals, color=colors_q, edgecolor='black', linewidth=0.8)
ax1.set_xlabel('Number of Precanned / Severe Quality Captions per Image', fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
quality_any_pct = sum(precanned_hist[k] for k in range(1, 6)) / total_viz_imgs * 100
quality_majority_pct = sum(precanned_hist[k] for k in range(3, 6)) / total_viz_imgs * 100
quality_unanimous_pct = precanned_hist[5] / total_viz_imgs * 100
ax1.set_title(f'(A) VizWiz Recognition Failure Signals (≥1/5: {quality_any_pct:.1f}%; ≥3/5: {quality_majority_pct:.1f}%)', fontsize=11, fontweight='bold')
for b, pct in zip(bars_q, prec_pcts):
    h = b.get_height()
    ax1.annotate(f'{pct:.1f}%\n({h:,})', xy=(b.get_x() + b.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax1.set_ylim(0, max(prec_vals) * 1.18)

# (B) Text Detection Annotator Consensus
text_vals = [text_hist[k] for k in range(6)]
text_pcts = [text_hist[k] / total_viz_imgs * 100 for k in range(6)]
colors_t = sns.color_palette("Blues", 6)
bars_t = ax2.bar([f'{k}/5' for k in range(6)], text_vals, color=colors_t, edgecolor='black', linewidth=0.8)
ax2.set_xlabel('Number of Annotators Confirming Text in Image', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
text_any_pct = sum(text_hist[k] for k in range(1, 6)) / total_viz_imgs * 100
text_majority_pct = sum(text_hist[k] for k in range(3, 6)) / total_viz_imgs * 100
text_unanimous_pct = text_hist[5] / total_viz_imgs * 100
ax2.set_title(f'(B) Text Detection Consensus (≥1/5: {text_any_pct:.1f}%; ≥3/5: {text_majority_pct:.1f}%)', fontsize=11, fontweight='bold')
for b, pct in zip(bars_t, text_pcts):
    h = b.get_height()
    ax2.annotate(f'{pct:.1f}%\n({h:,})', xy=(b.get_x() + b.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax2.set_ylim(0, max(text_vals) * 1.18)

# (C) Physical Blur (Laplacian Variance Log-Scale)
log_coco_blurs = np.log10(np.array(coco_blurs) + 1e-3)
log_viz_blurs = np.log10(np.array(viz_blurs) + 1e-3)

sns.kdeplot(log_coco_blurs, ax=ax3, color='#2980b9', label=f'COCO Val (median={np.median(coco_blurs):.1f})', fill=True, alpha=0.35, linewidth=2)
sns.kdeplot(log_viz_blurs, ax=ax3, color='#e74c3c', label=f'VizWiz Val (median={np.median(viz_blurs):.1f})', fill=True, alpha=0.35, linewidth=2)
ax3.set_xlabel('Log10(Laplacian variance): edge/texture signal', fontsize=11, fontweight='bold')
ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
ax3.set_title('(C) High-frequency detail indicator', fontsize=12, fontweight='bold')
ax3.legend(frameon=True, facecolor='white', framealpha=0.9)

# (D) Brightness / Luminance Distribution
sns.kdeplot(coco_brights, ax=ax4, color='#2980b9', label=f'COCO Val (median={np.median(coco_brights):.1f})', fill=True, alpha=0.35, linewidth=2)
sns.kdeplot(viz_brights, ax=ax4, color='#e67e22', label=f'VizWiz Val (median={np.median(viz_brights):.1f})', fill=True, alpha=0.35, linewidth=2)
ax4.set_xlabel('Mean Pixel Luminance (0=Black, 255=White)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Density', fontsize=11, fontweight='bold')
ax4.set_title('(D) Mean image luminance', fontsize=12, fontweight='bold')
ax4.legend(frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
fig_path5 = os.path.join(FIG_DIR, "05_vizwiz_quality_and_text_issues.png")
plt.savefig(fig_path5, dpi=300)
plt.close()
print(f"Saved: {fig_path5}")

# ==============================================================================
# 6. DOMAIN GAP: measured features are summarized in the report tables.
# ==============================================================================
print("\n[6/7] Synthesizing Multi-dimensional Domain Gap Analysis...")

print('Domain gap summary uses measured tables; no illustrative scores are generated.')

# ==============================================================================
# 7. QUALITATIVE IMAGE SAMPLES COMPARISON GRID
# ==============================================================================
print("\n[7/7] Generating Qualitative Sample Comparison Grid...")

# Curated Representative Samples
coco_candidates = [
    ("COCO_val2014_000000391895.jpg", "Curated Outdoor Scene", "A man with a red helmet on a moped on a dirt road."),
    ("COCO_val2014_000000522418.jpg", "Clear Object / Action", "A woman wearing a hair net cutting a large sheet cake."),
    ("COCO_val2014_000000184613.jpg", "Animals & Composition", "A young boy with an umbrella touching the horn of a yak."),
    ("COCO_val2014_000000161222.jpg", "Everyday Food / Table", "A table topped with plates of food and wine glasses.")
]

viz_candidates = [
    ("VizWiz_val_00000001.jpg", "Handheld / Medicine OCR", "A person is holding a small black bottle of NIGHT TIME medicine."),
    ("VizWiz_val_00000000.jpg", "Screen / Digital Display", "A partial computer screen showing a repair prompt popup."),
    ("VizWiz_val_00000013.jpg", "Severe Blur / Movement", "A blurry photo of an unrecognizable object on a table."),
    ("VizWiz_val_00000006.jpg", "Severe Glare / Obscured Scene", "Quality issues are too severe to recognize visual content.")
]

def get_valid_path(base_folder, fname, default_idx=0):
    p = os.path.join(base_folder, fname)
    if os.path.exists(p):
        return p
    all_f = [f for f in os.listdir(base_folder) if f.endswith('.jpg')]
    return os.path.join(base_folder, all_f[default_idx])

fig, axes = plt.subplots(2, 4, figsize=(16, 10))

# Top row: COCO
for col, (fn, title, cap) in enumerate(coco_candidates):
    ax = axes[0, col]
    img_p = get_valid_path(coco_val_dir, fn, col)
    img = Image.open(img_p)
    ax.imshow(img)
    ax.set_title(f"[COCO] {title}", fontsize=10, fontweight='bold', color='#1a5276', pad=6)
    wrapped_cap = textwrap.fill(cap, width=38)
    ax.set_xlabel(f'"{wrapped_cap}"', fontsize=9, style='italic', color='#2c3e50')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#2980b9')
        spine.set_linewidth(2)

# Bottom row: VizWiz
for col, (fn, title, cap) in enumerate(viz_candidates):
    ax = axes[1, col]
    img_p = get_valid_path(viz_val_dir, fn, col)
    img = Image.open(img_p)
    ax.imshow(img)
    is_failed = "Glare" in title or "Blur" in title
    border_color = '#c0392b' if is_failed else '#d35400'
    ax.set_title(f"[VizWiz] {title}", fontsize=10, fontweight='bold', color=border_color, pad=6)
    wrapped_cap = textwrap.fill(cap, width=38)
    ax.set_xlabel(f'"{wrapped_cap}"', fontsize=9, style='italic', color='#7f1d1d' if is_failed else '#2c3e50')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(2)

fig.suptitle('Qualitative Comparison: Benchmark Paradigm (MS-COCO) vs. In-The-Wild Reality (VizWiz-Captions)\n'
             'Minh họa trực quan: Bố cục chuẩn mực/đủ sáng (COCO) đối lập với Rung mờ, Nhãn chữ, Tay cầm & Lóa sáng (VizWiz)',
             fontsize=13, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.02, 1, 0.92], h_pad=4.5, w_pad=1.0)
fig_path7 = os.path.join(FIG_DIR, "07_qualitative_samples_grid.png")
plt.savefig(fig_path7, dpi=300)
plt.close()
print(f"Saved: {fig_path7}")

# ==============================================================================
# SAVE FULL METRICS JSON
# ==============================================================================
summary_metrics = {
    "splits": split_stats,
    "resolution": res_stats,
    "caption_stats": cap_stats,
    "coco_top_supercategories": df_coco_super.to_dict(orient="records"),
    "vizwiz_quality_distribution": df_quality.to_dict(orient="records"),
    "vizwiz_thresholds": {
        "quality_any_ge_1_of_5_pct": quality_any_pct,
        "quality_majority_ge_3_of_5_pct": quality_majority_pct,
        "quality_unanimous_5_of_5_pct": quality_unanimous_pct,
        "text_any_ge_1_of_5_pct": text_any_pct,
        "text_majority_ge_3_of_5_pct": text_majority_pct,
        "text_unanimous_5_of_5_pct": text_unanimous_pct
    },
    "blur_luminance": {
        "normalization": "EXIF transpose; preserve aspect ratio; resize short edge to 480 px",
        "sample_size_per_dataset": 1000,
        "sample_seed": 42,
        "coco_mean_blur": float(np.mean(coco_blurs)),
        "coco_median_blur": float(np.median(coco_blurs)),
        "coco_median_blur_ci95": coco_lap_ci,
        "coco_mean_luminance": float(np.mean(coco_brights)),
        "coco_median_luminance": float(np.median(coco_brights)),
        "vizwiz_mean_blur": float(np.mean(viz_blurs)),
        "vizwiz_median_blur": float(np.median(viz_blurs)),
        "vizwiz_median_blur_ci95": viz_lap_ci,
        "vizwiz_laplacian_by_precanned": lap_by_precanned,
        "vizwiz_mean_luminance": float(np.mean(viz_brights)),
        "vizwiz_median_luminance": float(np.median(viz_brights))
    }
}

with open(os.path.join(OUT_DIR, "eda_summary_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(summary_metrics, f, indent=2, ensure_ascii=False)

print("\n" + "="*70)
print("COMPREHENSIVE EDA FINISHED SUCCESSFULLY!")
print(f"All figures saved to: {FIG_DIR}")
print(f"All tables saved to: {TAB_DIR}")
print(f"Metrics saved to: {os.path.join(OUT_DIR, 'eda_summary_metrics.json')}")
print("="*70)
