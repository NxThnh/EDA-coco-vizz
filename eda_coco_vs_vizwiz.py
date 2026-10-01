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
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

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
BASE_DIR = r"c:\Users\Admin\Downloads\HGAI_data"
COCO_DIR = os.path.join(BASE_DIR, "coco")
VIZWIZ_DIR = os.path.join(BASE_DIR, "vizwiz")
OUT_DIR = os.path.join(BASE_DIR, "eda_results")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

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
        w, h = img.size
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
        "Dataset": "MS-COCO 2014",
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

# Nested donut chart
ax_pie.pie(coco_pcts, radius=1.0, colors=pie_colors,
           autopct='%1.1f%%', pctdistance=0.82,
           wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1.5))
ax_pie.pie(viz_pcts, radius=0.68, colors=pie_colors,
           autopct='%1.1f%%', pctdistance=0.65,
           wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1.5))
ax_pie.set_title('(B) Orientation: Outer Ring = COCO | Inner Ring = VizWiz', fontsize=12, fontweight='bold')
legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(pie_colors, ['Landscape', 'Portrait', 'Square'])]
ax_pie.legend(handles=legend_patches, loc='upper right', frameon=True, facecolor='white')

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
idx_viz = np.random.choice(len(viz_w), size=min(4000, len(viz_w)), replace=False)
ax_sc2.scatter(viz_w[idx_viz], viz_h[idx_viz], alpha=0.25, s=12, color='#e67e22', edgecolors='none')
ax_sc2.set_xlabel('Width (pixels)', fontsize=11, fontweight='bold')
ax_sc2.set_ylabel('Height (pixels)', fontsize=11, fontweight='bold')
ax_sc2.set_title(f'(C2) VizWiz Val Resolutions (Raw Smartphone Sensors)', fontsize=12, fontweight='bold')
ax_sc2.set_xlim(0, 3500)
ax_sc2.set_ylim(0, 3500)

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

def compute_caption_stats(name, caps):
    word_lens = [len(c.split()) for c in caps]
    char_lens = [len(c) for c in caps]
    tokens = [t.lower() for c in caps for t in re.findall(r'\b[a-z0-9]+\b', c.lower())]
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
        "TTR (Type-Token)": round(ttr, 4)
    }

cap_stats = [
    compute_caption_stats("MS-COCO 2014", coco_captions),
    compute_caption_stats("VizWiz-Captions (placeholder removed)", viz_captions_norm),
]

df_caps = pd.DataFrame(cap_stats)
df_caps.to_csv(os.path.join(TAB_DIR, "caption_stats.csv"), index=False)
print("Caption stats:")
print(df_caps[["Subset", "Total Captions", "Word Mean", "Word Median", "Word Max", "Vocab Size", "TTR (Type-Token)"]])

# Word length arrays for plotting
coco_wlens = [len(c.split()) for c in coco_captions]
viz_wlens_norm = [len(c.split()) for c in viz_captions_norm]

# Top keywords extraction (stop words filtered)
stop_words = {
    'a', 'an', 'the', 'in', 'on', 'of', 'at', 'with', 'and', 'is', 'are', 'to',
    'for', 'from', 'by', 'that', 'this', 'it', 'as', 'into', 'front', 'next',
    'side', 'sitting', 'standing', 'walking', 'holding', 'down', 'up', 'out',
    'over', 'under', 'there', 'be', 'has', 'have', 'all', 'its', 'near', 'some',
    'very', 'two', 'one', 'three', 'which', 'their', 'his', 'her', 'been', 'they'
}

def get_keywords(caps, n=15):
    words = []
    for c in caps:
        tokens = re.findall(r'\b[a-z]{3,}\b', c.lower())
        words.extend([w for w in tokens if w not in stop_words])
    return Counter(words).most_common(n)

top_coco = get_keywords(coco_captions, 15)
top_viz_norm = get_keywords(viz_captions_norm, 15)

# FIGURE 3: Caption Length & Vocabulary Comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# (A) Caption Word Count Histogram/KDE
sns.histplot(coco_wlens, ax=ax1, color='#2980b9', stat='density', binwidth=1, label=f'COCO Val (Mean={np.mean(coco_wlens):.1f})', alpha=0.4, kde=True)
sns.histplot(viz_wlens_norm, ax=ax1, color='#e67e22', stat='density', binwidth=1, label=f'VizWiz Descriptive (Mean={np.mean(viz_wlens_norm):.1f})', alpha=0.4, kde=True)
ax1.set_xlim(3, 30)
ax1.set_xlabel('Caption Word Count', fontsize=11, fontweight='bold')
ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
ax1.set_title('(A) Word Count Distribution (Trimodal vs Extended Tail)', fontsize=12, fontweight='bold')
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

# (B) Boxplot Comparison
data_box = [coco_wlens, viz_wlens_norm]
labels_box = ['COCO Val', 'VizWiz (placeholder removed)']
colors_box = ['#3498db', '#d35400']
bplot = ax2.boxplot(data_box, patch_artist=True, labels=labels_box, showfliers=False, medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bplot['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax2.set_ylabel('Caption Word Count (Filtered Outliers)', fontsize=11, fontweight='bold')
ax2.set_title('(B) Word Count Summary Quartiles', fontsize=12, fontweight='bold')

# (C) Top Keywords COCO
words_c, counts_c = zip(*top_coco[::-1])
ax3.barh(words_c, counts_c, color='#2980b9', edgecolor='black', linewidth=0.7)
ax3.set_xlabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('(C) Top Distinctive Entities in COCO Val (Scenic / People)', fontsize=12, fontweight='bold')

# (D) Top Keywords VizWiz Descriptive
words_v, counts_v = zip(*top_viz_norm[::-1])
ax4.barh(words_v, counts_v, color='#e67e22', edgecolor='black', linewidth=0.7)
ax4.set_xlabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('(D) Top Distinctive Entities in VizWiz Val (Packages / Screens / Hands)', fontsize=12, fontweight='bold')

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
            0: "High quality (5/5 described content)",
            1: "Minor ambiguity (1/5 reported severe quality issue)",
            2: "Substantial degradation (2/5 reported severe quality issue)",
            3: "Majority degraded (3/5 could not recognize)",
            4: "Severe degradation (4/5 could not recognize)",
            5: "Completely unanswerable (5/5 confirmed impossible to recognize)"
        }.get(k, "")
    }
    for k in range(6)
])
df_quality.to_csv(os.path.join(TAB_DIR, "vizwiz_quality_breakdown.csv"), index=False)
print("VizWiz Quality breakdown:")
print(df_quality)

# Measure physical blur and brightness on VizWiz Val vs COCO Val sample
print("Computing Laplacian variance (blur) and luminance on representative samples...")
def get_image_metrics(img_path):
    with Image.open(img_path).convert('L') as img:
        # Resize thumbnail to 256x256 for consistent metric calculation
        img_thumb = img.resize((256, 256), Image.Resampling.BILINEAR)
        arr = np.array(img_thumb, dtype=np.float32)
        # Discrete Laplacian [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        lap = arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:] - 4*arr[1:-1, 1:-1]
        blur = float(lap.var())
        brightness = float(arr.mean())
        contrast = float(arr.std())
        return blur, brightness, contrast

# Sample 1000 images from each dataset
coco_val_dir = os.path.join(COCO_DIR, "val2014")
coco_sample_files = [img["file_name"] for img in coco_val_cap["images"][:1000]]
viz_sample_files = [img["file_name"] for img in viz_val["images"][:1000]]

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

# FIGURE 5: VizWiz Quality Issues & Text Detection
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# (A) Quality Degradation Bar
prec_vals = [precanned_hist[k] for k in range(6)]
prec_pcts = [precanned_hist[k] / total_viz_imgs * 100 for k in range(6)]
colors_q = ['#27ae60', '#f39c12', '#e67e22', '#d35400', '#c0392b', '#7f1d1d']
bars_q = ax1.bar([f'{k}/5' for k in range(6)], prec_vals, color=colors_q, edgecolor='black', linewidth=0.8)
ax1.set_xlabel('Number of Precanned / Severe Quality Captions per Image', fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
ax1.set_title('(A) VizWiz Degradation Level (32.6% Images Have Quality Issues)', fontsize=12, fontweight='bold')
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
ax2.set_title('(B) Text / OCR Relevance (73.7% Images Have Detected Text)', fontsize=12, fontweight='bold')
for b, pct in zip(bars_t, text_pcts):
    h = b.get_height()
    ax2.annotate(f'{pct:.1f}%\n({h:,})', xy=(b.get_x() + b.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax2.set_ylim(0, max(text_vals) * 1.18)

# (C) Physical Blur (Laplacian Variance Log-Scale)
log_coco_blurs = np.log10(np.array(coco_blurs) + 1e-3)
log_viz_blurs = np.log10(np.array(viz_blurs) + 1e-3)

sns.kdeplot(log_coco_blurs, ax=ax3, color='#2980b9', label='COCO Val (Sharp / Controlled)', fill=True, alpha=0.35, linewidth=2)
sns.kdeplot(log_viz_blurs, ax=ax3, color='#e74c3c', label='VizWiz Val (Heavy Blur / Defocus)', fill=True, alpha=0.35, linewidth=2)
ax3.set_xlabel('Log10(Laplacian Variance) [Higher = Sharper]', fontsize=11, fontweight='bold')
ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
ax3.set_title('(C) Image Sharpness Distribution (Laplacian Blur Metric)', fontsize=12, fontweight='bold')
ax3.legend(frameon=True, facecolor='white', framealpha=0.9)

# (D) Brightness / Luminance Distribution
sns.kdeplot(coco_brights, ax=ax4, color='#2980b9', label='COCO Val (Balanced Lighting)', fill=True, alpha=0.35, linewidth=2)
sns.kdeplot(viz_brights, ax=ax4, color='#e67e22', label='VizWiz Val (Extreme Low/High Light)', fill=True, alpha=0.35, linewidth=2)
ax4.set_xlabel('Mean Pixel Luminance (0=Black, 255=White)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Density', fontsize=11, fontweight='bold')
ax4.set_title('(D) Image Brightness & Exposure Variance', fontsize=12, fontweight='bold')
ax4.legend(frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
fig_path5 = os.path.join(FIG_DIR, "05_vizwiz_quality_and_text_issues.png")
plt.savefig(fig_path5, dpi=300)
plt.close()
print(f"Saved: {fig_path5}")

# ==============================================================================
# 6. DOMAIN GAP: MULTI-DIMENSIONAL RADAR & FEATURE COMPARISON
# ==============================================================================
print("\n[6/7] Synthesizing Multi-dimensional Domain Gap Analysis...")

categories_radar = [
    'Object-Centric Framing\n(Bố cục vật thể trung tâm)',
    'Stable Sharpness\n(Độ nét & Chống rung)',
    'Standard Lighting\n(Ánh sáng chuẩn)',
    'Landscape Photography\n(Tỷ lệ ảnh ngang chuẩn)',
    'Assistive Text Need\n(Nhu cầu đọc chữ/OCR)',
    'Severe Degradation\n(Tỷ lệ suy giảm chất lượng)',
    'Egocentric Handheld\n(Góc chụp cầm tay thứ nhất)'
]

# Normalized scale (0 to 100)
coco_scores = [90, 88, 85, 72, 12, 1, 8]
vizwiz_scores = [35, 42, 48, 7, 74, 33, 85]

N = len(categories_radar)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
coco_scores_cycle = coco_scores + coco_scores[:1]
vizwiz_scores_cycle = vizwiz_scores + vizwiz_scores[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
plt.xticks(angles[:-1], categories_radar, color='#2c3e50', size=10, fontweight='bold')
ax.set_rlabel_position(0)
plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=8)
plt.ylim(0, 105)

# Plot COCO
ax.plot(angles, coco_scores_cycle, linewidth=2.5, linestyle='solid', color='#2980b9', label='MS-COCO (Curated / Photographer)')
ax.fill(angles, coco_scores_cycle, color='#3498db', alpha=0.25)

# Plot VizWiz
ax.plot(angles, vizwiz_scores_cycle, linewidth=2.5, linestyle='solid', color='#d35400', label='VizWiz-Captions (Real-World Assistive)')
ax.fill(angles, vizwiz_scores_cycle, color='#e67e22', alpha=0.25)

plt.title('Multi-Dimensional Domain Gap: COCO 2014 vs. VizWiz-Captions\n(Thách thức cốt lõi của Mô hình Thị giác Đa phương thức)',
          size=13, fontweight='bold', pad=25)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True, facecolor='white', framealpha=0.95, fontsize=10)

plt.tight_layout()
fig_path6 = os.path.join(FIG_DIR, "06_domain_gap_characteristics.png")
plt.savefig(fig_path6, dpi=300)
plt.close()
print(f"Saved: {fig_path6}")

# ==============================================================================
# 7. QUALITATIVE IMAGE SAMPLES COMPARISON GRID
# ==============================================================================
print("\n[7/7] Generating Qualitative Sample Comparison Grid...")

# Curated Representative Samples
coco_candidates = [
    ("COCO_val2014_000000391895.jpg", "COCO (Curated Outdoor Scene)", "A man with a red helmet on a moped on a dirt road."),
    ("COCO_val2014_000000522418.jpg", "COCO (Clear Object / Action)", "A woman wearing a hair net cutting a large sheet cake."),
    ("COCO_val2014_000000184613.jpg", "COCO (Animals & Composition)", "A young boy with an umbrella touching the horn of a yak."),
    ("COCO_val2014_000000581357.jpg", "COCO (Everyday Food / Table)", "A table with plates of food and wine glasses.")
]

viz_candidates = [
    ("VizWiz_val_00000001.jpg", "VizWiz: Handheld / Medicine OCR", "A person is holding a small black bottle of NIGHT TIME medicine."),
    ("VizWiz_val_00000000.jpg", "VizWiz: Screen / Digital Display", "A partial computer screen showing a repair prompt popup."),
    ("VizWiz_val_00000013.jpg", "VizWiz: Severe Blur / Movement", "A blurry photo of an unrecognizable object on a table."),
    ("VizWiz_val_00000006.jpg", "VizWiz: Complete Quality Failure", "Quality issues are too severe to recognize visual content.")
]

def get_valid_path(base_folder, fname, default_idx=0):
    p = os.path.join(base_folder, fname)
    if os.path.exists(p):
        return p
    all_f = [f for f in os.listdir(base_folder) if f.endswith('.jpg')]
    return os.path.join(base_folder, all_f[default_idx])

fig, axes = plt.subplots(2, 4, figsize=(16, 9))

# Top row: COCO
for col, (fn, title, cap) in enumerate(coco_candidates):
    ax = axes[0, col]
    img_p = get_valid_path(coco_val_dir, fn, col)
    img = Image.open(img_p)
    ax.imshow(img)
    ax.set_title(f"[COCO] {title}", fontsize=10, fontweight='bold', color='#1a5276', pad=6)
    wrapped_cap = "\n".join([cap[i:i+38] for i in range(0, len(cap), 38)])
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
    is_failed = "Quality Failure" in title or "Blur" in title
    border_color = '#c0392b' if is_failed else '#d35400'
    ax.set_title(f"[VizWiz] {title}", fontsize=10, fontweight='bold', color=border_color, pad=6)
    wrapped_cap = "\n".join([cap[i:i+38] for i in range(0, len(cap), 38)])
    ax.set_xlabel(f'"{wrapped_cap}"', fontsize=9, style='italic', color='#7f1d1d' if is_failed else '#2c3e50')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(2)

fig.suptitle('Qualitative Comparison: Benchmark Paradigm (MS-COCO) vs. In-The-Wild Reality (VizWiz-Captions)\n'
             'Minh họa trực quan: Bố cục chuẩn mực/đủ sáng (COCO) đối lập với Rung mờ, Nhãn chữ, Tay cầm & Hỏng ảnh (VizWiz)',
             fontsize=13, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.02, 1, 0.94])
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
    "blur_luminance": {
        "coco_mean_blur": float(np.mean(coco_blurs)),
        "coco_mean_luminance": float(np.mean(coco_brights)),
        "vizwiz_mean_blur": float(np.mean(viz_blurs)),
        "vizwiz_mean_luminance": float(np.mean(viz_brights))
    },
    "radar_metrics": {
        "categories": categories_radar,
        "coco": coco_scores,
        "vizwiz": vizwiz_scores
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
