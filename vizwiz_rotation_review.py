"""Create a deterministic paired before/after EXIF audit (40 images, two strata)."""
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'eda_results/hgai'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cross = pd.read_csv(ROOT/'eda_results/vlm_extensions/vizwiz_exif_rotated_crosscheck.csv')
    chosen = []
    for label, exif in [('exif_and_rotated',1),('rotated_without_exif',0)]:
        pool=cross[(cross.rotated_majority==1)&(cross.exif_rotation==exif)].sort_values('file_name')
        indices=random.Random(42).sample(list(pool.index),20)
        for index in sorted(indices):
            r=pool.loc[index].to_dict()
            r.update(stratum=label,review_status='pending',after_exif_orientation='',notes='')
            chosen.append(r)
    worklist=OUT/'vizwiz_rotation_review40.csv'
    if not worklist.exists():
        pd.DataFrame(chosen).to_csv(worklist,index=False,encoding='utf-8-sig')
    for page in range(4):
        canvas=Image.new('RGB',(1100,1450),'white')
        draw=ImageDraw.Draw(canvas)
        for pos,row in enumerate(chosen[page*10:(page+1)*10]):
            # Two pairs per row; five rows, each pair contains raw and EXIF.
            x=(pos%2)*550;y=(pos//2)*290
            with Image.open(ROOT/'vizwiz/val'/row['file_name']) as raw:
                for col,picture in enumerate([raw.convert('RGB'),ImageOps.exif_transpose(raw).convert('RGB')]):
                    picture.thumbnail((255,240))
                    canvas.paste(picture,(x+col*270+(255-picture.width)//2,y+30+(240-picture.height)//2))
            draw.text((x+4,y+2),row['file_name']+' | '+str(row['orientation']),fill='black')
            draw.text((x+4,y+273),'raw                         after EXIF',fill='black')
        canvas.save(OUT/f'vizwiz_rotation_review_{page+1}.jpg',quality=90)
    (OUT/'vizwiz_rotation_review_sampling.json').write_text(json.dumps({
        'seed':42,'images_reviewed_target':40,'exif_and_rotated_population':199,
        'rotated_without_exif_population':646,'stratum_sample_size':20,
        'note':'EXIF does not remove the original human Rotated label. Orientation judgment requires image review.'},indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
