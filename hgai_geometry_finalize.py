"""Correct geometry from actual SmolVLM processor semantics; full-val fixed-mode audit."""
import math
import json
import pandas as pd

from hgai_cpu_audit import ROOT, OUT, EXT, sha


def smol_grid(w,h):
    # Exact integer size rules from pinned Transformers 4.53.3 SmolVLM processor.
    if w>=h:
        rw=1536;rh=int(rw/(w/h));rh+=rh%2
    else:
        rh=1536;rw=int(rh*(w/h));rw+=rw%2
    rw=max(rw,1);rh=max(rh,1)
    if rw>=rh:
        tw=math.ceil(rw/384)*384;th=math.ceil(int(tw/(rw/rh))/384)*384
    else:
        th=math.ceil(rh/384)*384;tw=math.ceil(int(th*(rw/rh))/384)*384
    return max(tw,384),max(th,384)


def main():
    path=OUT/'processor_per_image.csv'
    frame=pd.read_csv(path)
    frame['distortion_factor']=frame['distortion_factor'].astype(float)
    for index,r in frame.iterrows():
        aspect=r.width/r.height
        if r.model=='SmolVLM2-2.2B':
            tw,th=smol_grid(r.width,r.height)
            factor=(tw/r.width)/(th/r.height)
            frame.loc[index,'global_image_distortion_factor']=max(aspect,1/aspect)
            frame.loc[index,'distortion_factor']=max(factor,1/factor) if r['mode']=='split_on' else max(aspect,1/aspect)
            frame.loc[index,'local_grid_width']=tw//384 if r['mode']=='split_on' else 0
            frame.loc[index,'local_grid_height']=th//384 if r['mode']=='split_on' else 0
            frame.loc[index,'local_long_edge_source_per_model_pixel']=max(r.width/tw,r.height/th) if r['mode']=='split_on' else None
            if r['mode']=='split_on':
                assert r.tiles_including_global==(tw//384)*(th//384)+1,(r.image_id,r.tiles_including_global,tw,th)
        else:
            frame.loc[index,'global_image_distortion_factor']=1
    frame.to_csv(path,index=False,encoding='utf-8-sig')
    summaries=[]
    for (partition,model,mode),g in frame.groupby(['partition','model','mode']):
        summaries.append({'partition':partition,'model':model,'mode':mode,'n':len(g),
                          'mean_padding_pct':g.padding_area_pct.mean(),'crop_area_lost_pct':g.crop_area_lost_pct.max(),
                          'median_distortion':g.distortion_factor.median(),'max_distortion':g.distortion_factor.max(),
                          'median_global_distortion':g.global_image_distortion_factor.median(),
                          'median_global_long_edge_source_per_384':g.long_edge_to_384_ratio.median(),
                          'median_local_source_per_pixel':g.local_long_edge_source_per_model_pixel.dropna().median() if g.local_long_edge_source_per_model_pixel.notna().any() else None})
    pd.DataFrame(summaries).to_csv(OUT/'processor_geometry_summary.csv',index=False,encoding='utf-8-sig')
    # Analytical controls over full val, using audited EXIF dimensions, not a claim of model forward passes.
    dims=pd.read_csv(EXT/'llava15_preprocessing_geometry.csv')[['dataset','image_id','width_exif','height_exif']]
    rows=[]
    for _,r in dims.iterrows():
        w,h=r.width_exif,r.height_exif
        for model,side,mode in [('TinyLLaVA-3.1B',384,'pad'),('TinyLLaVA-3.1B',384,'center_crop_control'),
                                ('TinyLLaVA-3.1B',384,'direct_square_control'),
                                ('LLaVA-1.5-7B',336,'hgai_center_crop'),('LLaVA-1.5-7B',336,'pad_control')]:
            rows.append({'dataset':r.dataset,'image_id':r.image_id,'model':model,'mode':mode,
                         'padding_area_pct':100*(1-min(w,h)/max(w,h)) if 'pad' in mode else 0,
                         'crop_area_lost_pct':100*(1-min(w,h)/max(w,h)) if 'crop' in mode else 0,
                         'distortion_factor':max(w/h,h/w) if mode=='direct_square_control' else 1,
                         'long_edge_source_per_target':max(w,h)/side,'short_edge_source_per_target':min(w,h)/side})
    full=pd.DataFrame(rows);full.to_csv(OUT/'fixed_mode_geometry_full_val.csv',index=False,encoding='utf-8-sig')
    full.groupby(['dataset','model','mode']).agg(images=('image_id','size'),
           median_pad_pct=('padding_area_pct','median'),median_crop_pct=('crop_area_lost_pct','median'),
           median_distortion=('distortion_factor','median'),
           median_long_edge_source_per_target=('long_edge_source_per_target','median')).reset_index().to_csv(
           OUT/'fixed_mode_geometry_summary.csv',index=False,encoding='utf-8-sig')
    environment_path=OUT/'processor_environment.json'
    environment=json.loads(environment_path.read_text(encoding='utf-8'))
    environment['geometry_postprocessing']={'code_sha256':sha(ROOT/'hgai_geometry_finalize.py'),
        'smol_rule':'Pinned resize->round grid to multiples384->tiles plus square global; split_off direct square',
        'fields_corrected':['distortion_factor','global_image_distortion_factor','local_grid_width','local_grid_height','local_long_edge_source_per_model_pixel'],
        'not_model_forward':'Analytical geometry appended to already measured CPU processor token outputs.'}
    environment_path.write_text(json.dumps(environment,indent=2),encoding='utf-8')


if __name__=='__main__':main()
