"""Measure COCO object bounding boxes outside a centered square crop."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'coco/annotations/instances_val2014.json').read_text())
images={im['id']:im for im in data['images']}
counts={'instances':0,'center_outside':0,'fully_outside':0,'partly_outside':0,'area_total':0.0,'area_outside':0.0}
for ann in data['annotations']:
    im=images[ann['image_id']]
    width,height=im['width'],im['height']
    side=min(width,height)
    left=(width-side)/2
    top=(height-side)/2
    x,y,w,h=ann['bbox']
    if w<=0 or h<=0:continue
    overlap_w=max(0,min(x+w,left+side)-max(x,left))
    overlap_h=max(0,min(y+h,top+side)-max(y,top))
    inside=overlap_w*overlap_h
    area=w*h
    counts['instances']+=1
    counts['center_outside']+=not(left<=x+w/2<=left+side and top<=y+h/2<=top+side)
    counts['fully_outside']+=inside==0
    counts['partly_outside']+=inside<area-1e-7
    counts['area_total']+=area
    counts['area_outside']+=area-inside
result={'crop':'center square before resize to CLIP 336px',**counts,
        'center_outside_pct':100*counts['center_outside']/counts['instances'],
        'fully_outside_pct':100*counts['fully_outside']/counts['instances'],
        'partly_outside_pct':100*counts['partly_outside']/counts['instances'],
        'bbox_area_outside_pct':100*counts['area_outside']/counts['area_total']}
dest=ROOT/'eda_results/hgai/coco_center_crop_object_coverage.json'
dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
