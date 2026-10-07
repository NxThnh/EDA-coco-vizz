"""CPU measurements requested for the HGAI report; never edits locked manifests."""
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'eda_results/hgai'


def main():
    overlap = pd.read_csv(OUT / 'small_vlm_training_overlap.csv')
    pool = set(overlap.loc[overlap.tiny_share_recipe_known.eq(0), 'image_id'].astype(int))
    data = json.loads((ROOT / 'coco/annotations/instances_val2014.json').read_text(encoding='utf-8'))
    all_ids = {int(r['id']) for r in data['images']}
    assert len(all_ids) == 40504 and len(pool) == 2977 and pool <= all_ids
    rest = all_ids - pool
    classes = defaultdict(set)
    instances = defaultdict(int)
    boxes = defaultdict(lambda: defaultdict(list))
    sizes = {int(r['id']): (r['width'], r['height']) for r in data['images']}
    for ann in data['annotations']:
        i, c = ann['image_id'], ann['category_id']
        classes[i].add(c)
        instances[i] += 1
        x, y, w, h = ann['bbox']
        if w > 0 and h > 0:
            boxes[i][c].append((x, y, x+w, y+h))
    rows = []
    for cat in data['categories']:
        c = cat['id']
        a = sum(c in classes[i] for i in pool)
        b = sum(c in classes[i] for i in rest)
        p, q = a/len(pool), b/len(rest)
        total = (a+b)/len(all_ids)
        difference = 100*(p-total)
        scale = len(rest)/len(all_ids)
        se = 100*scale*math.sqrt(p*(1-p)/len(pool)+q*(1-q)/len(rest))
        rows.append({'Lớp':cat['name'],'Ảnh nhóm chưa khớp':a,'Ảnh toàn val':a+b,
                     'Tỷ lệ nhóm (%)':100*p,'Tỷ lệ toàn val (%)':100*total,
                     'Hiệu (điểm %)':difference,'CI 95% dưới':difference-1.96*se,
                     'CI 95% trên':difference+1.96*se})
    pd.DataFrame(rows).to_csv(OUT/'object_shift_pool_vs_val.csv',index=False,encoding='utf-8-sig')
    summary = {'pool_images':len(pool),'val_images':len(all_ids),
               'pool_mean_classes_per_image':sum(len(classes[i]) for i in pool)/len(pool),
               'val_mean_classes_per_image':sum(len(classes[i]) for i in all_ids)/len(all_ids),
               'pool_mean_instances_per_image':sum(instances[i] for i in pool)/len(pool),
               'val_mean_instances_per_image':sum(instances[i] for i in all_ids)/len(all_ids),
               'ci_method':'Wald 95%, finite-pool overlap handled by comparison to complement'}
    (OUT/'object_shift_pool_vs_val_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    selected=[r for r in rows if r['CI 95% dưới']>0 or r['CI 95% trên']<0]
    fig, ax = plt.subplots(figsize=(8,max(3,len(selected)*0.28+1)))
    if selected:
        selected.sort(key=lambda r:r['Hiệu (điểm %)'])
        y=range(len(selected))
        ax.errorbar([r['Hiệu (điểm %)'] for r in selected],y,
                    xerr=[[r['Hiệu (điểm %)']-r['CI 95% dưới'] for r in selected],
                          [r['CI 95% trên']-r['Hiệu (điểm %)'] for r in selected]],fmt='o')
        ax.set_yticks(list(y),[r['Lớp'] for r in selected])
        ax.axvline(0,color='gray')
        ax.set_xlabel('Hiệu tỷ lệ ảnh chứa lớp (điểm %)')
    else:
        ax.text(.5,.5,'Không có lớp có CI 95% không chứa 0',ha='center',va='center',transform=ax.transAxes)
    fig.tight_layout();fig.savefig(OUT/'sampling_object_shift_pool.png',dpi=200);plt.close(fig)
    direct=ROOT/'coco/annotations/instances_val2017.json'
    archive=ROOT/'coco/annotations/annotations_trainval2017.zip'
    if direct.exists() or (archive.exists() and zipfile.is_zipfile(archive)):
        if direct.exists():
            ids={int(r['id']) for r in json.loads(direct.read_text(encoding='utf-8'))['images']}
            source='coco/annotations/instances_val2017.json'
        else:
            with zipfile.ZipFile(archive) as z:
                key=next((n for n in z.namelist() if n.endswith('/instances_val2017.json')),None)
                if key is None: raise RuntimeError('Missing instances_val2017.json in official archive')
                ids={int(r['id']) for r in json.load(z.open(key))['images']}
            source='coco/annotations/annotations_trainval2017.zip :: '+key
        membership={'pool_images':len(pool),'in_val2017':len(pool&ids),
                    'outside_val2017':len(pool-ids),'val2017_images':len(ids),
                    'source':source}
        (OUT/'unmatched_pool_val2017_membership.json').write_text(json.dumps(membership,ensure_ascii=False,indent=2),encoding='utf-8')

    crop_rows=[]
    random_ids=set(pd.read_csv(ROOT/'data_prepared/hgai/eval_random500.csv').image_id.astype(int))
    for scope, ids in [('Toàn val',all_ids),('eval_random500',random_ids)]:
        pairs=fully=partial=0
        for i in ids:
            width,height=sizes[i]
            side=min(width,height)
            left=(width-side)/2; top=(height-side)/2
            right=left+side; bottom=top+side
            for group in boxes[i].values():
                pairs+=1
                outside=[]; cut=[]
                for x1,y1,x2,y2 in group:
                    overlap_w=max(0,min(x2,right)-max(x1,left))
                    overlap_h=max(0,min(y2,bottom)-max(y1,top))
                    outside.append(overlap_w*overlap_h==0)
                    cut.append(x1<left or y1<top or x2>right or y2>bottom)
                fully+=int(all(outside))
                partial+=int(any(c and not o for c,o in zip(cut,outside)))
        crop_rows.append({'Phạm vi':scope,'Cặp ảnh–lớp':pairs,'Mọi instance ngoài':fully,
                          'Mọi instance ngoài (%)':100*fully/pairs,
                          'Cặp bị cắt một phần':partial,'Cặp bị cắt một phần (%)':100*partial/pairs})
    pd.DataFrame(crop_rows).to_csv(OUT/'center_crop_image_class_coverage.csv',index=False,encoding='utf-8-sig')
    print('Object classes with CI excluding zero:',len(selected),'crop scopes:',len(crop_rows))


if __name__ == '__main__': main()
