"""Build both reader-facing HGAI reports from measured repository artifacts."""
import json
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from build_hgai_report import table as source_table

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'eda_results/hgai'
REPORT=ROOT/'EDA_COCO_vs_VizWiz_Report.md'
ACTIVE=['dev_common_unmatched500','eval_common_unmatched500','eval_random500']
TABLES={}
SOURCES={
 'report_active_manifests.csv':'eda_results/hgai/manifest_summary.csv',
 'report_training_overlap.csv':'eda_results/hgai/small_vlm_training_overlap_summary.csv',
 'report_sampling_geometry.csv':'eda_results/hgai/sampling_geometry_shift.csv',
 'report_object_shift.csv':'eda_results/hgai/object_shift_pool_vs_val.csv',
 'report_crop_coverage.csv':'eda_results/hgai/center_crop_image_class_coverage.csv',
 'report_smol_token_distribution.csv':'eda_results/hgai/smol_tokens_768_per_image.csv',
 'report_compute_budget.csv':'experiment_outputs/smol_baseline/chair_dev_common_unmatched500_nf4_split_on_n20.run.json',
 'report_tiny_checkpoint_comparison.csv':'eda_results/hgai/tiny_checkpoint_public_metadata.json',
 'report_alpha_probe20.csv':'eda_results/hgai/alpha_probe20_summary.csv',
 'report_alpha_probe20_extra.csv':'eda_results/hgai/alpha_probe20_extra.csv',
 'report_layer_sensitivity.csv':'eda_results/hgai/layer_rule_sensitivity.csv',
 'report_processor_geometry.csv':'eda_results/hgai/processor_geometry_summary.csv',
 'report_person_reference.csv':'eda_results/tables/person_reference_one_caption_per_image.csv',
 'report_supercategories.csv':'eda_results/hgai/coco_supercategories.csv',
}


def vi(number,digits=0):
    if isinstance(number,(int,float)) and abs(number)<.5*10**(-digits):number=0.0
    return f'{number:,.{digits}f}'.replace(',','@').replace('.',',').replace('@','.')


def ratio(count,total):
    return f'{vi(count)}/{vi(total)} ({vi(100*count/total,1)}%)'


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def save(name,rows):
    TABLES[name]=pd.DataFrame(rows)
    TABLES[name].to_csv(OUT/name,index=False,encoding='utf-8-sig')


def table(relative,title,columns=None):
    name=Path(relative).name
    if name in TABLES:
        return source_table(f'eda_results/hgai/{name}',title,columns,frame=TABLES[name])
    return source_table(relative,title,columns)


def main():
    summary=pd.read_csv(OUT/'manifest_summary.csv')
    summary=summary[summary.partition.isin(ACTIVE)]
    save('report_active_manifests.csv',[{'Bộ ảnh':r.partition,'Số ảnh':int(r.images),'Seed':int(r.seed),
        'Khớp nguồn đã kiểm tra':ratio(int(r.known_union),int(r.images)),
        'Chưa khớp nguồn đã kiểm tra':ratio(int(r.unmatched),int(r.images))}
        for r in summary.itertuples()])
    overlap=pd.read_csv(OUT/'small_vlm_training_overlap_summary.csv')
    overlap_main=overlap[overlap.scope.eq('val2014')]
    meaning={'exact_SFT_manifest_ID_join':'Chỉ xác nhận ID trong SFT đã kiểm tra.',
             'recipe_conditional_exact_SFT_ID_join':'Chỉ xác nhận ID nếu checkpoint dùng recipe này.',
             'recipe_conditional_pretrain_and_SFT_ID_join':'Chỉ xác nhận ID nếu checkpoint dùng recipe này.',
             'upstream_pool_candidate_not_verified_training_exposure':'Chỉ là ảnh ứng viên ở nguồn upstream.'}
    save('report_training_overlap.csv',[{'Bộ ảnh':r.scope,'Mô hình hoặc recipe':r.model_or_recipe,
         'ID khớp':ratio(int(r.matched_ids),int(r.images)),
         'Giới hạn kết luận':meaning.get(r.status,'Chưa xác định phạm vi huấn luyện.')}
         for r in overlap_main.itertuples()])
    geometry=pd.read_csv(OUT/'sampling_geometry_shift.csv')
    geometry=geometry[geometry.partition.isin(ACTIVE)]
    save('report_sampling_geometry.csv',[{'Bộ ảnh':r.partition,
         'Ảnh dọc':ratio(round(r.portrait_pct*5),500),
         'Ảnh vuông':ratio(round(r.square_pct*5),500),
         'Token ảnh Smol trung bình':vi(r.mean_smol_image_tokens_split_on),
         'Token caption P50':vi(r.median_reference_caption_tokens_smol),
         'Lệch tâm vật thể P50':vi(r.median_largest_object_center_offset,2)}
         for r in geometry.itertuples()])
    object_summary=read('object_shift_pool_vs_val_summary.json')
    shift=pd.read_csv(OUT/'object_shift_pool_vs_val.csv')
    save('report_object_shift.csv',[{'Lớp':r['Lớp'],
         f"Nhóm {vi(object_summary['pool_images'])} ảnh":ratio(int(r['Ảnh nhóm chưa khớp']),object_summary['pool_images']),
         'Toàn val2014':ratio(int(r['Ảnh toàn val']),object_summary['val_images']),
         'Hiệu (điểm %)':vi(r['Hiệu (điểm %)'],1),
         'CI 95% (điểm %)':f"[{vi(r['CI 95% dưới'],1)}; {vi(r['CI 95% trên'],1)}]"}
         for _,r in shift.iterrows()])
    significant=shift.loc[(shift['CI 95% dưới']>0)|(shift['CI 95% trên']<0),'Lớp'].tolist()
    crop=pd.read_csv(OUT/'center_crop_image_class_coverage.csv')
    save('report_crop_coverage.csv',[{'Phạm vi':r['Phạm vi'],'Số cặp ảnh–lớp':int(r['Cặp ảnh–lớp']),
         'Mọi instance ở ngoài':ratio(int(r['Mọi instance ngoài']),int(r['Cặp ảnh–lớp'])),
         'Bị cắt một phần':ratio(int(r['Cặp bị cắt một phần']),int(r['Cặp ảnh–lớp']))}
         for _,r in crop.iterrows()])
    center=read('coco_object_centering.json')
    coverage=json.loads((ROOT/'eda_results/vlm_extensions/coco_caption_label_completeness.json').read_text(encoding='utf-8'))
    lexical=pd.read_csv(ROOT/'eda_results/vlm_extensions/coco_caption_label_completeness.csv')
    lexical_total=int(lexical.instance_classes.sum())
    lexical_selected=int(lexical.instance_classes_found_selected.sum())
    lexical_union=int(lexical.instance_classes_found_union.sum())
    token_status='Chưa đo: phép đếm processor ở mức 768 đang chạy.'
    if (OUT/'smol_tokens_768_per_image.csv').exists():
        token=pd.read_csv(OUT/'smol_tokens_768_per_image.csv')
        earlier=pd.read_csv(OUT/'processor_per_image.csv')
        earlier=earlier[(earlier.partition.isin(ACTIVE))&(earlier.model=='SmolVLM2-2.2B')][['partition','mode','image_tokens']]
        token=pd.concat([token[['partition','mode','image_tokens']],earlier],ignore_index=True)
        token=token.groupby(['partition','mode','image_tokens']).size().rename('images').reset_index()
        save('report_smol_token_distribution.csv',[{'Bộ ảnh':r.partition,'Chế độ':{'split_on':'Chia ô mặc định','split_768':'Chia ô 768','split_off':'Tắt chia ô'}[r.mode],
             'Token ảnh':int(r.image_tokens),'Số ảnh':ratio(int(r.images),500)} for r in token.itertuples()])
        token_status=table('eda_results/hgai/report_smol_token_distribution.csv','Phân bố token ảnh theo processor')
    lock=read('layer_rule_lock.json')
    protocol=read('experiment_config.json')
    design=read('report_pending_design.json')
    run=json.loads((ROOT/'experiment_outputs/smol_baseline/chair_dev_common_unmatched500_nf4_split_on_n20.run.json').read_text(encoding='utf-8'))
    seconds_per_image=run['seconds_this_run']/run['rows']
    layers=json.loads((ROOT/'model_assets/SmolVLM2-2.2B-Instruct/config.json').read_text(encoding='utf-8'))['text_config']['num_hidden_layers']
    full=design['coarse_scan']['full_dev_images']
    coarse=design['coarse_scan']['dev_subset_images']
    current_grid=len(read('smol_alpha_grid_lock.json')['alpha_grid'])
    def pilot_rate(path):
        entry=json.loads((ROOT/path).read_text(encoding='utf-8'))
        return entry['seconds_this_run']/entry['rows']
    edge_rate=pilot_rate('experiment_outputs/alpha_probe20/smol_split_768_alpha_0.000.run.json')
    off_rate=pilot_rate('experiment_outputs/alpha_probe20/smol_split_off_alpha_0.000.run.json')
    pope_rate=sum(pilot_rate(f'experiment_outputs/smol_baseline/pope_{v}_nf4_split_on_n60.run.json') for v in ('random','popular','adversarial'))/3
    tiny_pilot20=ROOT/'experiment_outputs/tiny_alpha_probe20_v2/tiny_pad_alpha_0.000.run.json'
    tiny_pilot10=ROOT/'experiment_outputs/tiny_alpha_probe10/tiny_pad_alpha_0.000.run.json'
    tiny_rate=(pilot_rate('experiment_outputs/tiny_alpha_probe20_v2/tiny_pad_alpha_0.000.run.json')
               if tiny_pilot20.exists() else
               pilot_rate('experiment_outputs/tiny_alpha_probe10/tiny_pad_alpha_0.000.run.json')
               if tiny_pilot10.exists() else None)
    def chosen_width(model,depth):
        result_path=OUT/f'layer_rule_v2_100_{model}.json'
        if not result_path.exists() and model=='smol':result_path=OUT/'layer_rule_v2_20_smol.json'
        if result_path.exists():
            found=json.loads(result_path.read_text(encoding='utf-8'))['v2_band']
            if found is not None:return found[1]-found[0]+1
        return 14
    smol_width=chosen_width('smol',24)
    tiny_width=chosen_width('tiny',32)
    smol_narrow=math.ceil(24/4)
    tiny_narrow=math.ceil(32/4)
    tiny_adapter_path=OUT/'tiny_last3_adapter5_validation.json'
    tiny_adapter=read('tiny_last3_adapter5_validation.json') if tiny_adapter_path.exists() else None
    tiny_last3_invalid=bool(tiny_adapter and tiny_adapter['nonfinite_images']>0)
    tiny_last_valid=28 if tiny_last3_invalid else 31
    budget=[]
    curve_windows=read('curve_measure_windows.json')['models']
    def curve_window(model):
        return f'{vi(curve_windows[model]["minutes"],2)} phút từ dòng đầu đến dòng cuối; chưa gồm nạp model'
    def budget_row(model,branch,configs,units,rate,measured='—'):
        hours=configs*units*rate/3600 if rate is not None else None
        budget.append({'Mô hình':model,'Nhánh':branch,'Cấu hình':configs,'Ảnh/câu mỗi cấu hình':units,
                       'Tổng lượt sinh':configs*units,'Giây/lượt từ pilot':vi(rate,2) if rate is not None else 'chưa đo',
                       'Giờ quy đổi':vi(hours,2) if hours is not None else 'chưa ước tính','Thời gian đã đo':measured,
                       '_hours':hours})
    budget_row('Smol','Dev baseline + lưới α (dải kiểm chứng)',1+current_grid,full,seconds_per_image)
    budget_row('Smol','Dò độ dài trên dải v2: baseline + 5 liều',1+current_grid,20,seconds_per_image)
    budget_row('Smol','Đo lại hai đường cong trên 100 ảnh dev',1,100,None,curve_window('smol'))
    budget_row('Smol','VAR mọi bước sinh, phân tích mô tả',1,20,None)
    budget_row('Smol',f'Liệt kê đủ dải rộng {smol_width} layer; gồm phân phối và oracle dev',24-smol_width+1,coarse,seconds_per_image)
    budget_row('Smol',f'Liệt kê đủ dải hẹp {smol_narrow} layer; gồm phân phối và oracle dev',24-smol_narrow+1,coarse,seconds_per_image)
    budget_row('Smol','Dải 5–18 và dải co, 5 liều',2*current_grid,coarse,seconds_per_image)
    budget_row('Smol','Mọi layer',1,coarse,seconds_per_image)
    budget_row('Smol','Eval lọc: 512 token, baseline/HGAI',2,full,seconds_per_image)
    budget_row('Smol','Eval ngẫu nhiên: 512 token, baseline/HGAI',2,full,seconds_per_image)
    budget_row('Smol','Eval lọc: chia ô 768, baseline/HGAI',2,full,edge_rate)
    budget_row('Smol','Eval lọc: tắt chia ô, baseline/HGAI',2,full,off_rate)
    budget_row('Smol','Eval lọc: detailed64/concise64, 2 nhánh',4,full,None)
    budget_row('Smol','POPE: 3 biến thể, baseline/HGAI',6,3000,pope_rate)
    budget_row('Tiny','Dev baseline + 5 liều',6,full,tiny_rate)
    budget_row('Tiny','Dò độ dài trên dải v2: baseline + 5 liều',6,20,tiny_rate)
    budget_row('Tiny','Dò liều output: baseline + 6 liều trên 20 ảnh',7,20,tiny_rate)
    budget_row('Tiny','Đo hai đường cong trên 100 ảnh dev',1,100,None,curve_window('tiny'))
    budget_row('Tiny',f'Liệt kê đủ dải rộng {tiny_width} layer; gồm phân phối và oracle dev',tiny_last_valid-tiny_width+2,coarse,tiny_rate)
    budget_row('Tiny',f'Liệt kê đủ dải hẹp {tiny_narrow} layer; gồm phân phối và oracle dev',tiny_last_valid-tiny_narrow+2,coarse,tiny_rate)
    budget_row('Tiny','Dải 5–18 và dải co (trùng ở 32 layer)',current_grid,coarse,tiny_rate)
    if not tiny_last3_invalid:budget_row('Tiny','Mọi layer',1,coarse,tiny_rate)
    budget_row('Tiny','Eval lọc: 512/64/concise64, 2 nhánh',6,full,None)
    budget_row('Tiny','Eval ngẫu nhiên: 512, baseline/HGAI',2,full,None)
    budget_row('Tiny','POPE: 3 biến thể, baseline/HGAI',6,3000,None)
    budget_row('LLaVA','Eval ngẫu nhiên: baseline và 5–18/α=0,5',2,full,None)
    budget_row('LLaVA','Đối chiếu FP16–NF4 trên 30 ảnh',2,30,None)
    known_hours=sum(row['_hours'] for row in budget if row['_hours'] is not None)
    total_units=sum(row['Tổng lượt sinh'] for row in budget)
    budget.append({'Mô hình':'Tổng','Nhánh':'Phần có tốc độ pilot / toàn kế hoạch',
                   'Cấu hình':'—','Ảnh/câu mỗi cấu hình':'—','Tổng lượt sinh':total_units,
                   'Giây/lượt từ pilot':'các hàng còn thiếu chưa đo',
                   'Giờ quy đổi':f'{vi(known_hours,2)} giờ phần quy đổi; tổng đầy đủ chưa xác định','Thời gian đã đo':'—','_hours':None})
    save('report_compute_budget.csv',[{k:v for k,v in row.items() if k!='_hours'} for row in budget])
    cfg=json.loads((ROOT/'model_assets/TinyLLaVA-3.1B/config.json').read_text(encoding='utf-8'))
    tiny_public=read('tiny_checkpoint_public_metadata.json')
    def weight_description(repo):
        shards=tiny_public[repo]['weight_shards']
        return f"{vi(sum(s['bytes'] for s in shards))} byte; SHA-256: "+' / '.join(s['sha256'] for s in shards)
    tiny_validation_path=OUT/'tiny_checkpoint_validation.json'
    tiny_validation=json.loads(tiny_validation_path.read_text(encoding='utf-8')) if tiny_validation_path.exists() else None
    tiny_loaded=bool(tiny_validation and tiny_validation.get('loaded') and tiny_validation.get('attention_intervention'))
    save('report_tiny_checkpoint_comparison.csv',[
        {'Checkpoint':'bczhou/TinyLLaVA-3.1B','Cấu hình':f"Phi-2, SigLIP; {cfg['num_hidden_layers']} layer; pad 384",
         'Trọng số / hash':weight_description('bczhou/TinyLLaVA-3.1B'),
         'Nạp trong môi trường hiện tại':'Đã nạp và sinh caption trên 1 ảnh dev.' if tiny_loaded else ('Chưa đo forward; trọng số đã đối chiếu.' if (OUT/'tiny_weights_provenance.json').exists() else 'Chưa đo: shard trọng số tải dở.'),
         'Can thiệp attention':'Đã xác nhận modified_calls > 0 trên 1 ảnh dev.' if tiny_loaded else 'Mã adapter có đường Phi-2; chưa kiểm tra forward đủ trọng số.'},
        {'Checkpoint':'tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B','Cấu hình':f"Phi-2, SigLIP; {tiny_public['tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B']['text_layers']} layer; ảnh vuông 384; model_type=tinyllava",
         'Trọng số / hash':weight_description('tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B'),
         'Nạp trong môi trường hiện tại':'Không tương thích adapter tiny_llava_phi hiện tại; chưa thử đường remote code.',
         'Can thiệp attention':'Adapter hiện tại chưa nối lớp TinyLlavaForConditionalGeneration; cần port riêng.'}])
    people=pd.read_csv(ROOT/'eda_results/tables/person_reference_one_caption_per_image.csv')
    save('report_person_reference.csv',[{'Bộ dữ liệu':r['Dataset'],
         'Caption chọn':ratio(int(r['Selected Captions']),int(r['All Images'])),
         'Lượt nhắc nhóm người':int(r['Person-family Mentions']),
         'Caption nhắc nhóm người':ratio(int(r['Selected Captions with Person Family']),int(r['Selected Captions']))}
         for _,r in people.iterrows()])
    supercats=pd.read_csv(OUT/'coco_supercategories.csv')
    total=int(supercats['Count'].sum())
    save('report_supercategories.csv',[{'Siêu danh mục':r.Supercategory,
         'Số instance':ratio(int(r.Count),total)} for r in supercats.itertuples()])
    process=pd.read_csv(OUT/'processor_geometry_summary.csv')
    fixed=pd.read_csv(OUT/'fixed_mode_geometry_summary.csv')
    def proc(model,mode):
        return process[(process.partition=='eval_common_unmatched500')&(process.model==model)&(process['mode']==mode)].iloc[0]
    def fixed_row(model,mode):
        return fixed[(fixed.dataset=='COCO')&(fixed.model==model)&(fixed['mode']==mode)].iloc[0]
    split=proc('SmolVLM2-2.2B','split_on')
    off=proc('SmolVLM2-2.2B','split_off')
    tiny_geom=fixed_row('TinyLLaVA-3.1B','pad')
    llava_geom=fixed_row('LLaVA-1.5-7B','hgai_center_crop')
    save('report_processor_geometry.csv',[
        {'Mô hình/chế độ':'SmolVLM2 chia ô mặc định','Token ảnh':'81 × số ô','Đệm/cắt':'0% crop hợp tile',
         'Méo trung vị':vi(split.median_distortion,2),'Hình học':f"Pixel nguồn/pixel tile {vi(split.median_local_source_per_pixel,3)} (phóng {vi(1/split.median_local_source_per_pixel,1)} lần); ảnh global bị bóp vuông, méo {vi(split.median_global_distortion,2)}."},
        {'Mô hình/chế độ':'SmolVLM2 tắt chia ô','Token ảnh':'81','Đệm/cắt':'resize vuông',
         'Méo trung vị':vi(off.median_distortion,2),'Hình học':'Thay cả số token lẫn hình học.'},
        {'Mô hình/chế độ':'TinyLLaVA pad384','Token ảnh':'729','Đệm/cắt':f"pad P50 {vi(tiny_geom.median_pad_pct,1)}%",
         'Méo trung vị':vi(tiny_geom.median_distortion,2),'Hình học':'Giữ tỷ lệ ảnh bằng đệm vuông.'},
        {'Mô hình/chế độ':'LLaVA CLIP336 cắt giữa','Token ảnh':'576','Đệm/cắt':f"crop P50 {vi(llava_geom.median_crop_pct,1)}%",
         'Méo trung vị':vi(llava_geom.median_distortion,2),'Hình học':'Bỏ mép ảnh trước encoder.'}])
    parts=[]
    def add(s): parts.append(s.strip()+'\n')
    add('''# EDA phục vụ khái quát hóa HGAI sang VLM nhỏ

COCO val2014 là tập đánh giá chính. SmolVLM2-2.2B và TinyLLaVA-3.1B là hai mô hình chính; LLaVA-1.5-7B là mốc kiểm chứng. **CHAIR là thước đo chính, POPE kiểm tra chéo**; VizWiz cung cấp bối cảnh về khác biệt dữ liệu.

Năm phát hiện chính:

1. Toàn bộ **40.504/40.504** ảnh COCO val2014 giải mã được; **35.504** ảnh cũng có ID trong train2017.
2. Mix SFT LLaVA khớp **16.618/40.504** ID COCO khi tính cả alias VG/GQA; recipe ShareGPT4V khớp **37.527/40.504**, để lại **2.977** ảnh chưa khớp nguồn đã kiểm tra.
3. SmolVLM2 tạo **1.053 token ảnh** cho ảnh 640×480 khi chia 13 ô, còn tắt chia ô có **81 token**; TinyLLaVA pad384 có **729 token**.
4. Center crop LLaVA làm **5.750/116.592 (4,9%)** cặp ảnh–lớp mất mọi instance khỏi vùng nhìn thấy.
5. Quy tắc layer v1 trên **20 ảnh dev SmolVLM2** trả “không tìm thấy ranh giới”; phép dò α trên 20 ảnh cho thấy ở α=0,10 có **4/20** caption chia ô mặc định co quá nửa.''')
    add(f'''## 1. Dữ liệu và nguồn huấn luyện

COCO val2014 có {vi(coverage['images'])} ảnh và {vi(coverage['captions'])} caption. Kiểm tra toàn vẹn giải mã được **40.504/40.504 ảnh**; đối chiếu ID COCO cho thấy **35.504/40.504** ảnh val2014 cũng nằm trong train2017. Đây là giao nhau của hai split, không tự chứng minh VLM đã học các ảnh này.

**SFT** là giai đoạn fine-tune có giám sát; **recipe** là công thức phối trộn nguồn huấn luyện, không phải danh sách chắc chắn từng mẫu mà checkpoint đã thấy. LLaVA được kiểm tra bằng đường dẫn COCO trực tiếp trong mix SFT 665K và nối `image_id` Visual Genome/GQA sang `coco_id`: 5.741 ID trực tiếp, 10.877 ID chỉ qua alias, hợp là 16.618. TinyLLaVA được đối chiếu riêng với recipe LLaVA và ShareGPT4V, gồm SFT và pretraining có ID công khai. Với SmolVLM2, tám shard ID của hai slice OneVision ShareGPT4V-COCO/LLaVA được quét; khớp chỉ đánh dấu ứng viên trong nguồn upstream. Các nguồn khác, lịch sử mô hình nền và mẫu thực sự được rút từ recipe SmolVLM2 chưa có manifest đầy đủ. Phép nối ID vì vậy không chứng minh hai VLM nhỏ chưa từng thấy ảnh.''')
    add(table('eda_results/hgai/report_training_overlap.csv','Khớp ID theo nguồn đã kiểm tra'))
    add('Trong 12 phép đối chiếu ID ở ba manifest còn dùng, cả dev_common_unmatched500 và eval_common_unmatched500 đều khớp 0/500 ở bốn nguồn/recipe đã kiểm tra; eval_random500 lần lượt khớp 205/500 với LLaVA SFT, 205/500 với TinyLLaVA theo recipe LLaVA, 460/500 với recipe ShareGPT4V và 171/500 ứng viên upstream SmolVLM2. Đây là phép khớp ID, không phải bằng chứng checkpoint đã học ảnh.')
    add('''## 2. Các bộ ảnh cố định và phân bố object

Ba manifest dùng cho thí nghiệm gồm dev_common_unmatched500, eval_common_unmatched500 và eval_random500. Dev dùng để chọn cấu hình; hai bộ eval dùng để báo kết quả.''')
    add(table('eda_results/hgai/report_active_manifests.csv','Quy mô và độ khớp nguồn của ba bộ ảnh'))
    add(table('eda_results/hgai/report_sampling_geometry.csv','Dịch chuyển hình học giữa các bộ ảnh'))
    add(f'''Nhóm chưa khớp nguồn ShareGPT4V đã kiểm tra gồm {vi(object_summary['pool_images'])} ảnh. Số lớp trung bình mỗi ảnh là {vi(object_summary['pool_mean_classes_per_image'],2)}, so với {vi(object_summary['val_mean_classes_per_image'],2)} trên toàn val; số instance trung bình tương ứng là {vi(object_summary['pool_mean_instances_per_image'],2)} và {vi(object_summary['val_mean_instances_per_image'],2)}. Hiệu trong hình/bảng là **nhóm 2.977 trừ toàn val2014**; CI Wald 95% được tính từ **nhóm 2.977 và phần bù 37.527 ảnh không giao nhau**, rồi nhân hệ số 37.527/40.504 để cùng thang hiệu với toàn val. Có 80 phép so lớp riêng; CI từng lớp chưa điều chỉnh cho đa kiểm định.''')
    add(f"Đồ trong nhà như bottle/cup/dining table tăng, phương tiện như car/train/truck giảm; chênh lệch từng lớp đều dưới 2 điểm phần trăm trước làm tròn. Có {vi(len(significant))} lớp có CI 95% chưa điều chỉnh không chứa 0; [bảng đủ 80 lớp](#phụ-lục-a-bảng-và-hình-nền).")
    add('![Các lớp có khoảng tin cậy của hiệu không chứa 0](eda_results/hgai/sampling_object_shift_pool.png)')
    membership_path=OUT/'unmatched_pool_val2017_membership.json'
    if membership_path.exists():
        membership=read('unmatched_pool_val2017_membership.json')
        add(f"Theo ID của annotation COCO val2017, toàn bộ {vi(membership['pool_images'])} ảnh nhóm này thuộc val2017: **2.977 = 5.000 ảnh val2017 trừ 2.023 ảnh khớp** nguồn đã kiểm tra. Đây là phép nối ID, không phải bảo đảm sạch mọi nguồn huấn luyện.")
    else:
        add(f'''Kiểm chứng nhóm {vi(object_summary['pool_images'])} ảnh với annotation COCO val2017: **chưa đo**, vì tệp annotation val2017 chưa có trong repo. Chưa xác nhận số ảnh nằm trong và ngoài val2017.''')
    add('''## 3. Hình học ảnh và token thị giác

### 3.1. Token ảnh SmolVLM2

Số token là số placeholder `<image>` sau processor, không tính token văn bản, hàng/cột hay patch nội bộ.''')
    add(token_status)
    add('''SmolVLM2 nén mỗi ô ảnh thành **81 token** đưa vào LLM; ảnh 640×480 tạo **13 ô = 1.053 token**, gồm các ô cục bộ và ảnh toàn cục. Khi chia ô, ảnh toàn cục được resize vuông nên có méo, còn tile cục bộ giữ gần tỷ lệ hơn. Bảng dưới phân biệt số token, phần ảnh bị cắt/đệm và độ méo; nhiều token hơn không đồng nghĩa có thêm thông tin nguồn.''')
    add(table('eda_results/hgai/report_processor_geometry.csv','Hình học và ngân sách token theo processor'))
    add('''### 3.2. Cắt giữa, pad và độ phủ object

Nhánh LLaVA theo mã HGAI cắt vuông giữa trước CLIP 336px. Bảng tính theo cặp ảnh–lớp COCO; “mọi instance ở ngoài” là lớp không có instance nào giao vùng cắt; “bị cắt một phần” là có ít nhất một instance còn giao vùng cắt nhưng không nằm trọn trong đó.''')
    add(table('eda_results/hgai/report_crop_coverage.csv','Độ phủ cặp ảnh–lớp sau cắt giữa'))
    add('LLaVA cắt, TinyLLaVA pad, SmolVLM2 giữ toàn ảnh, nên chênh lệch recall giữa ba mô hình có một phần do object có được nhìn thấy hay không.')
    add('''### 3.3. Độ lệch tâm của vật thể

Độ lệch tâm là khoảng cách tâm bbox tới tâm ảnh, chia cho nửa đường chéo ảnh. Tính trên instance COCO, gồm cả crowd.''')
    c=center['largest_object']
    add(f"Trên {vi(c['n'])} ảnh có instance, p50/p90/p95/p99 của bbox lớn nhất lần lượt là {vi(c['p50'],2)}/{vi(c['p90'],2)}/{vi(c['p95'],2)}/{vi(c['p99'],2)}; {vi(center['images_without_instances'])} ảnh không có instance. VizWiz chưa có bbox tương ứng.")
    add('''## 4. Caption và nhãn CHAIR

### 4.1. Độ dài caption tham chiếu

Độ dài caption được đếm bằng tokenizer của từng checkpoint, không gồm prompt hoặc visual token.''')
    add(table('eda_results/hgai/caption_token_summary.csv','Độ dài caption theo tokenizer'))
    add('''### 4.2. Nhãn tham chiếu cho CHAIR

CHAIR dùng hợp nhãn instance và đối tượng từ mọi caption chuẩn. Phép đối chiếu từ vựng trong EDA chỉ chẩn đoán độ phủ của caption được chọn.''')
    add(f"Một caption thu hồi {ratio(lexical_selected,lexical_total)} lớp instance; hợp mọi caption thu hồi {ratio(lexical_union,lexical_total)}.")
    add(f'''## 5. Giao thức và thiết kế thí nghiệm

### 5.1. Mốc LLaVA

Mã đánh giá HGAI dùng prompt **“Please help me describe the image in detail.”**, 500 ảnh COCO, giải mã greedy (`do_sample=False`, `num_beams=1`) và tối đa 512 token. Bài báo công bố CHAIR-S **{vi(protocol['published_reference_metrics']['CHAIR_S_greedy_512_pct'],1)}** ở 512 token và **{vi(protocol['published_reference_metrics']['CHAIR_S_greedy_64_pct'],1)}** ở 64 token; đây là số tham chiếu, chưa phải kết quả tái lập ở đây. HGAI của tác giả dùng layer **5–18 tính cả hai đầu**, α=0,5 và gộp mean. [Bảng 2 ở 512 token]({protocol['published_reference_metrics']['table_512']}), [bảng 3 ở 64 token]({protocol['published_reference_metrics']['table_64']}).

`eval_data_loader.py` của mã HGAI đưa ảnh vào CLIP 336px theo đường **cắt giữa**; pad vuông là điều kiện khác. LLaVA chỉ có hai hàng kiểm chứng: baseline và HGAI 5–18/α=0,5 trên nhánh mẫu ngẫu nhiên. Kết quả NF4 phải báo riêng với mốc FP16; đối chiếu FP16–NF4 cần cùng checkpoint, ảnh, processor, prompt và giải mã trên khoảng 30 ảnh tại máy đủ VRAM, rồi báo hiệu CHAIR ghép cặp. Checkpoint LLaVA chưa có trọng số đầy đủ trong môi trường này nên hàng kiểm chứng và đối chiếu precision **chưa chạy**.''')
    add('''### 5.2. Điều kiện caption

| Điều kiện | Token tối đa | Vai trò |
| :--- | ---: | :--- |
| detailed512 | 512 | Chính |
| detailed64 | 64 | Đối chứng độ dài |
| concise64 | 64 | Đối chứng prompt |''')
    add('''Mỗi cặp baseline–HGAI giữ nguyên ảnh, prompt, processor, tokenizer và giải mã. CHAIR-S/I đi cùng recall và F1 của object, số token sinh, tỷ lệ chạm trần, tỷ lệ từ chối/rỗng, phần 4-gram lặp, Distinct-1, thời gian và bộ nhớ. **Guard** là ngưỡng loại cấu hình suy giảm chất lượng trên 500 ảnh dev: recall/F1 không giảm quá 0,02; refusal, rỗng, chạm trần và lặp 4-gram không tăng quá 0,05; độ dài trung bình còn ít nhất 50% baseline. Đây là quy tắc chọn đã khóa, không phải kết quả tác giả HGAI. Khoảng tin cậy chênh lệch baseline–HGAI dùng bootstrap **ghép cặp theo ảnh**; CHAIR-I lấy tổng tử số chia tổng mẫu số trong mỗi mẫu bootstrap.''')
    boundary_path=OUT/'layer_boundary_result.json'
    boundary=json.loads(boundary_path.read_text(encoding='utf-8')) if boundary_path.exists() else None
    assert boundary is None or boundary['rule_sha256']==lock['rule_sha256']
    add(f'''### 5.3. Layer và α của VLM nhỏ

Quy tắc khóa có SHA-256 `{lock['rule_sha256']}`. Chuẩn hóa min–max riêng đường cong VAR và độ nhạy che ảnh trên toàn bộ layer. Nếu một đường cong hằng, chứa giá trị không hữu hạn hoặc có dưới 8 layer thì trả “không tìm thấy ranh giới”. Tìm mọi dải liền mạch tối đại mà cả hai đường cong chuẩn hóa ≥0,60; giữ dải rộng ít nhất max(2, ⌈0,10L⌉) layer. Chọn dải có tổng min(hai giá trị chuẩn hóa) lớn nhất; hòa thì chọn dải bắt đầu sớm nhất. Nếu không có dải đạt điều kiện, trả “không tìm thấy ranh giới”. VAR là tổng attention tới token ảnh ở bước sinh đầu, trung bình qua head và ảnh; độ nhạy che ảnh là sai khác chuẩn hóa của hidden state khi thay ảnh bằng màu RGB (127,127,127).''')
    if boundary:
        band=boundary['band']
        phrase=f"layer {band['start_inclusive']}–{band['end_inclusive']}" if isinstance(band,dict) else band
        add(f"Trên **SmolVLM2-2.2B, {vi(boundary['images'])} ảnh dev**, đường cong VAR và độ nhạy che ảnh cho kết quả theo quy tắc v1: **{phrase}**. [Dữ liệu từng layer](eda_results/hgai/layer_curves_summary.csv). VAR ở đây chỉ đo **bước sinh đầu**, với query token đầu cố định; chưa đo trung bình mọi bước sinh. Nếu đo thêm mọi bước, kết quả phải báo riêng. Không dùng điểm CHAIR để xác định dải.")
    else:
        add('Đường cong VAR và độ nhạy che ảnh: **chưa đo**, do phép tính GPU chưa hoàn tất.')
    add('![Đường cong VAR và độ nhạy che ảnh: giá trị thô và chuẩn hóa](eda_results/hgai/layer_curves_raw_normalized.png)')
    add('Độ nhạy che ảnh là sai khác hidden state **cộng dồn theo độ sâu**: layer sau nhận biểu diễn đã qua các layer trước. Vì vậy dải v2 chỉ gợi ý **ranh giới dưới** của vùng ảnh bắt đầu có ảnh hưởng lớn; dải chạm layer cuối không chứng minh từng layer cuối là quan trọng. Mức tăng giữa hai layer kề nhau dưới đây là phân tích mô tả, không tham gia quy tắc v1/v2 hay lựa chọn dải.')
    add('![Mức tăng độ nhạy che ảnh giữa hai layer liên tiếp](eda_results/hgai/layer_mask_increment.png)')
    sensitivity=pd.read_csv(OUT/'layer_rule_sensitivity.csv')
    save('report_layer_sensitivity.csv',[{'Ngưỡng':vi(r.threshold,2),'Layer 0':('Có' if r.include_layer_0 else 'Không'),
         'Số layer xét':int(r.layers_considered),'Dải mô tả':r.band} for r in sensitivity.itertuples()])
    add('Bảng độ nhạy dưới đây áp cùng phép chọn dải cho ngưỡng 0,50/0,60/0,70; khi bỏ layer 0, min–max được tính lại trên layer 1…L−1. Đây là phân tích mô tả, **không thay quy tắc v1 đã khóa**.')
    add(table('eda_results/hgai/report_layer_sensitivity.csv','Độ nhạy của ranh giới theo ngưỡng và layer 0'))
    curves=pd.read_csv(OUT/'layer_curves_summary.csv').set_index('layer')
    var0=float(curves.loc[0,'var'])
    var_low=curves.loc[3:5,'var']
    var_mid=float(curves.loc[12,'var'])
    mask9=float(curves.loc[9,'mask_sensitivity'])
    mask10=float(curves.loc[10,'mask_sensitivity'])
    mask22=float(curves.loc[22,'mask_sensitivity'])
    mask23=float(curves.loc[23,'mask_sensitivity'])
    add(f'Trên 20 ảnh SmolVLM2, VAR ở layer 0 là {vi(var0,3)}, vùng thấp 3–5 là {vi(var_low.min(),3)}–{vi(var_low.max(),3)}, rồi nổi lên tới {vi(var_mid,3)} ở layer 12. Độ nhạy che ảnh tăng từ {vi(mask9,3)} ở layer 9 lên {vi(mask10,3)} ở layer 10, và từ {vi(mask22,3)} ở layer 22 lên {vi(mask23,3)} ở layer 23. Dải 5–18 tuyệt đối và dải co 3–14 đều phủ vùng VAR nổi lên quanh layer 12, nhưng đều bỏ đỉnh độ nhạy ở layer cuối. Trong min–max v1, đỉnh VAR ở layer 0 và đỉnh độ nhạy ở layer 23 chi phối hai thang; VAR của mọi layer còn lại đều dưới ngưỡng chuẩn hóa 0,60, nên hai đường không thể đồng thời đạt ngưỡng ở các layer đó.')
    v2_lock=read('layer_rule_v2_lock.json')
    v2_20=read('layer_rule_v2_20_smol.json')
    def band_text(value):
        return 'không tìm thấy ranh giới' if value is None else f'layer {value[0]}–{value[1]}'
    add(f'Quy tắc **v2** đã khóa riêng, SHA-256 `{v2_lock["rule_sha256"]}`: chuẩn hóa riêng từng đường bằng `clip((x−P10)/(P90−P10), 0, 1)`, với P10/P90 là phân vị 10/90 qua layer; ngưỡng chung 0,50. Mọi bước chọn dải còn lại giữ như v1: dải đạt ngưỡng liền mạch tối đại, rộng ít nhất max(2, ⌈0,10L⌉), chọn tổng min lớn nhất, hòa chọn đầu sớm nhất; không có dải thì trả “không tìm thấy ranh giới”. **v2 được đặt sau khi đã thấy đường cong SmolVLM2 20 ảnh**; tham số không được sửa theo kết quả tiếp theo. Trên chính 20 ảnh này, v1 trả “không tìm thấy ranh giới”, v2 trả **{band_text(v2_20["v2_band"])}**, chạm layer cuối. Kết quả này chỉ mô tả cùng tập ảnh đã dùng để quan sát đường cong.')
    bootstrap_rows=[]
    if (OUT/'layer_curves100_dev_ids.csv').exists():
        add('Cả hai phép đo 100 ảnh dùng cùng [danh sách ID dev seed 927](eda_results/hgai/layer_curves100_dev_ids.csv), không lấy ảnh eval.')
    for model,label in [('smol','SmolVLM2-2.2B'),('tiny','bczhou/TinyLLaVA-3.1B')]:
        path=OUT/f'layer_rule_v2_100_{model}.json'
        if path.exists():
            result=json.loads(path.read_text(encoding='utf-8'))
            add(f'Trên 100 ảnh dev seed {result["seed"]} của {label}, v1: **{band_text(result["v1_band"])}**; v2: **{band_text(result["v2_band"])}**. Bootstrap 1.000 lần ghép theo ảnh cho tần suất từng cặp đầu–cuối trong [tệp kết quả](eda_results/hgai/layer_rule_v2_100_{model}.json).')
            if model=='tiny':
                add('Với TinyLLaVA NF4, forward decoder dùng SDPA như nhánh sinh caption; xác suất attention của query được đọc ở float32. Forward eager cho NaN ở layer 29–31 trên 100/100 ảnh nên không dùng để áp quy tắc ([kiểm kê giá trị lỗi](eda_results/hgai/tiny_eager_invalid_audit.json)). CSV SDPA đã kiểm tra mọi giá trị hữu hạn.')
            add(f'![Đường cong layer của {label} trên 100 ảnh dev](eda_results/hgai/layer_rule_v2_100_{model}_curves.png)')
            bootstrap_rows.extend({'Mô hình':label,'Dải v2':('không tìm thấy ranh giới' if entry['band']=='none' else entry['band'].replace('-', '–')),
                'Số lần / 1.000':int(entry['count']),'Tần suất':vi(100*entry['frequency'],1)+'%'}
                for entry in result['v2_bootstrap_frequencies'])
        else:
            add(f'Đo lại VAR/độ nhạy che ảnh trên 100 ảnh dev cho {label}, rồi áp v1/v2 và bootstrap 1.000 lần theo ảnh: **chưa chạy xong** vì phép forward GPU trên checkpoint {label} chưa hoàn tất.')
    if bootstrap_rows:
        save('report_layer_v2_bootstrap.csv',bootstrap_rows)
        add(table('eda_results/hgai/report_layer_v2_bootstrap.csv','Tần suất dải v2 qua bootstrap theo ảnh'))
    all_step=OUT/'var_all_steps_smol20_summary.csv'
    if all_step.exists():
        allstep_values=pd.read_csv(all_step).set_index('layer')['var_all_steps']
        add(f'VAR trung bình trên mọi bước sinh của cùng 20 ảnh dev SmolVLM2 là phân tích mô tả riêng: giải mã greedy NF4 với attention eager, lấy trung bình các bước query của từng ảnh đến EOS hoặc trần 512 rồi lấy trung bình đều theo ảnh. Ở layer 0, giá trị giảm từ {vi(var0,3)} ở bước đầu xuống {vi(allstep_values.loc[0],3)} khi lấy mọi bước; layer 12 tương ứng {vi(var_mid,3)} và {vi(allstep_values.loc[12],3)}. [Đường cong theo layer](eda_results/hgai/var_all_steps_smol20_summary.csv) không dùng để sửa v1 hoặc v2.')
        add('![VAR ở bước đầu và trung bình mọi bước sinh trên 20 ảnh dev SmolVLM2](eda_results/hgai/var_all_steps_smol20.png)')
    else:
        add('Đường cong VAR trung bình trên mọi bước sinh của 20 ảnh dev: **chưa chạy** vì cần đọc attention tại từng bước sinh; không dùng phép đo này để sửa v1 hoặc v2.')
    add('Quy tắc v1 và v2 trên LLaVA-1.5-7B: **chưa chạy**, vì checkpoint trọng số đầy đủ chưa có ở môi trường này; do đó chưa có dải đo để so với mốc 5–18. Không suy dải LLaVA từ đường cong của mô hình khác.')
    attention_check=OUT/'layer_attention_validation.json'
    if attention_check.exists():
        check=read('layer_attention_validation.json')
        save('report_layer_attention_validation.csv',[{'Mô hình':r['model'],'Ảnh dev':r['images'],
            'Vị trí ảnh':', '.join(map(str,r['image_positions'])),
            'Vị trí đầu':', '.join(map(str,r['first_position'])),
            'Vị trí cuối':', '.join(map(str,r['last_position'])),
            'Sai số tổng hàng lớn nhất':vi(r['max_row_sum_error'],7),
            'Sai lệch eager–SDPA lớn nhất':vi(r['max_eager_sdpa_abs_error'],7)}
            for r in check['models']])
        add('Tổng hàng attention được tính trên xác suất eager sau khi lưu ở FP16; tiêu chí kiểm tra là |tổng−1| < 0,0005 trên từng hàng và head, còn bảng báo sai số lớn nhất của 20 ảnh. Đối chiếu SDPA tái dựng xác suất từ cùng query/key ở float32 trước phép can thiệp: TinyLLaVA xét layer 0–28, SmolVLM2 xét toàn bộ layer 0–23. Layer 29–31 của TinyLLaVA không được dùng trong cực đại vì eager cho giá trị không hữu hạn.')
        add('Sai lệch xác suất cực đại của TinyLLaVA là '+vi(check['models'][1]['max_eager_sdpa_abs_error'],5)+'; đường cong VAR TinyLLaVA vì vậy phụ thuộc cách tính xác suất attention trong NF4/FP16. Đây là giới hạn của phép đo và không được dùng để sửa các ngưỡng hoặc dải v1/v2 đã khóa.')
        add(table('eda_results/hgai/report_layer_attention_validation.csv','Kiểm chứng phép đo VAR trên 20 ảnh dev'))
    else:
        add('Kiểm chứng VAR trên 20 ảnh dev (vị trí ảnh, tổng hàng attention, đối chiếu eager–SDPA ở layer 0–28): **chưa chạy**; cần forward hai backend trên cùng đầu vào và kiểm tra sai số từng hàng.')
    if tiny_adapter:
        add(f'Can thiệp TinyLLaVA α=0,02 tại layer 29–31 trên {tiny_adapter["images"]} ảnh dev: {tiny_adapter["nonfinite_images"]}/{tiny_adapter["images"]} ảnh có NaN/Inf. '+('Các dải chứa layer 29–31 và điều kiện “mọi layer” bị loại khỏi kế hoạch.' if tiny_last3_invalid else 'Các dải chứa layer 29–31 và điều kiện “mọi layer” vẫn được giữ trong kế hoạch.'))
    else:
        add('Kiểm tra adapter TinyLLaVA tại layer 29–31 trên 5 ảnh dev: **chưa chạy**; NaN ở phép đo eager không đủ để loại điều kiện sinh caption dùng SDPA.')
    probe=design['alpha_probe']
    doses='/'.join(vi(a,3 if a in (0.035,0.075) else 2) for a in probe['alphas'])
    alpha_path=OUT/'alpha_probe20_summary.csv'
    if alpha_path.exists():
        alpha=pd.read_csv(alpha_path)
        assert len(alpha)==len(probe['alphas'])*len(probe['processor_modes'])
        names={'split_on':'Chia ô mặc định','split_768':'Chia ô 768','split_off':'Tắt chia ô'}
        save('report_alpha_probe20.csv',[{'Chế độ':names[r.mode], 'α':vi(r.alpha,3 if r.alpha in (0.035,0.075) else 2),
            'Tỷ lệ độ dài TB':vi(r.mean_length_ratio,2),
            'Co quá nửa':ratio(int(r.less_than_half_count),int(r.images)),
            'Caption trùng hệt':ratio(int(r.identical_caption_count),int(r.images)),
            'Khoảng cách sửa từ TB':vi(r.mean_word_edit_distance,2)} for r in alpha.itertuples()])
        add(f"Phép dò α trên {vi(probe['dev_images'])} ảnh dev, dải layer {probe['layer_start_inclusive']}–{probe['layer_end_inclusive']}, với ba chế độ token ảnh và các liều {doses}. Tỷ lệ độ dài là trung bình của 20 tỷ số token sinh theo cặp can thiệp/baseline; caption chạm trần 512 token dùng độ dài quan sát. Khoảng cách sửa là Levenshtein ở mức từ, phân tách theo khoảng trắng, giữ chữ hoa và dấu câu. Bảng không dùng điểm CHAIR.")
        add(table('eda_results/hgai/report_alpha_probe20.csv','Phép dò liều α ghép cặp trên 20 ảnh dev'))
        extra=pd.read_csv(OUT/'alpha_probe20_extra.csv')
        save('report_alpha_probe20_extra.csv',[{'Chế độ':r.mode_label,
            'α':vi(r.alpha,3 if r.alpha in (.035,.075) else 2),
            'Trung vị tỷ lệ độ dài':vi(r.median_length_ratio,2),
            'Chạm trần 512':ratio(int(r.ceiling_512_count),int(r.images)),
            'Có lặp 4-gram':ratio(int(r.repetition_any_count),int(r.images)),
            'Phần 4-gram lặp TB':vi(r.mean_repeated_4gram_fraction,3)} for r in extra.itertuples()])
        add('Tỷ lệ chạm trần và lặp là chỉ báo suy giảm output, không dùng thay recall/F1. Lặp được đo bằng tỷ phần 4-gram từ lặp trong từng caption, tách từ theo khoảng trắng và đưa về chữ thường.')
        add(table('eda_results/hgai/report_alpha_probe20_extra.csv','Trung vị độ dài, chạm trần và lặp trong phép dò α'))
        add('Lưới α khóa trước khi đo CHAIR trên 500 ảnh dev: '+', '.join(vi(a,3 if a in (0.035,0.075) else 2) for a in read('smol_alpha_grid_lock.json')['alpha_grid'])+'. Guard đã khóa trên toàn bộ 500 ảnh dev quyết định cấu hình hợp lệ.')
        tiny_probe_path=OUT/'tiny_alpha_probe20_v2_summary.csv'
        if tiny_probe_path.exists():
            tiny_probe=pd.read_csv(tiny_probe_path)
            save('report_tiny_alpha_probe20_v2.csv',[{'α':vi(r.alpha,2),
                'Tỷ lệ độ dài TB':vi(r.mean_length_ratio,2),
                'Trung vị tỷ lệ độ dài':vi(r.median_length_ratio,2),
                'Co quá nửa':ratio(int(r.less_than_half_count),int(r.images)),
                'Caption trùng hệt':ratio(int(r.identical_caption_count),int(r.images)),
                'Khoảng cách sửa từ TB':vi(r.mean_word_edit_distance,2),
                'Chạm trần 512':ratio(int(r.ceiling_512_count),int(r.images)),
                'Có lặp 4-gram':ratio(int(r.repetition_4gram_count),int(r.images))}
                for r in tiny_probe.itertuples()])
            add('Phép dò trên **bczhou/TinyLLaVA-3.1B**, 20 ảnh đầu theo manifest dev đã khóa, NF4, pad384, layer 5–18, prompt detailed512, các liều 0,02/0,05/0,10/0,20/0,35/0,50. Mọi tỷ lệ độ dài và caption trùng hệt được ghép theo cùng ảnh baseline; khoảng cách sửa là Levenshtein mức từ. Không dùng CHAIR hoặc eval để chọn α.')
            add(table('eda_results/hgai/report_tiny_alpha_probe20_v2.csv','Phép dò liều α của TinyLLaVA trên 20 ảnh dev'))
            tiny_grid_path=OUT/'tiny_alpha_grid_lock.json'
            if tiny_grid_path.exists():
                tiny_grid=read('tiny_alpha_grid_lock.json')
                add(f'Lưới α TinyLLaVA đã khóa riêng: α thấp={vi(tiny_grid["alpha_low"],2)} là liều nhỏ nhất làm ít nhất 10/20 caption khác baseline; α cao={vi(tiny_grid["alpha_high"],2)} là liều lớn nhất có không quá 4/20 ảnh co quá nửa. Năm mức cách đều theo thang log, kể cả hai đầu: '+', '.join(vi(value,4) for value in tiny_grid['alpha_grid'])+f'. SHA-256 `{tiny_grid["rule_sha256"]}`. Lưới SmolVLM2 giữ nguyên.')
            else:
                add('Lưới α TinyLLaVA: **chưa khóa** vì các điều kiện đầu mút của quy tắc trên output chưa cho năm mức phân biệt hợp lệ. Lưới SmolVLM2 giữ nguyên.')
        else:
            add('Phép dò liều α cho **bczhou/TinyLLaVA-3.1B** trên 20 ảnh dev ở sáu liều 0,02/0,05/0,10/0,20/0,35/0,50: **chưa chạy xong** vì phép sinh caption GPU đang tiếp tục; chưa khóa lưới TinyLLaVA. Lưới SmolVLM2 giữ nguyên.')
    else:
        add(f"Phép dò α trên {vi(probe['dev_images'])} ảnh dev, dải layer {probe['layer_start_inclusive']}–{probe['layer_end_inclusive']}, ba mức token và các liều {doses}: **chưa chạy**. Cần hoàn tất {vi(probe['dev_images']*len(probe['alphas'])*len(probe['processor_modes']))} lượt sinh caption can thiệp ghép với baseline trên GPU; tệp khóa lưới α hiện hành giữ nguyên. Lưới đề xuất gồm cả 0,075 và 0,10; guard đã khóa trên toàn bộ 500 ảnh dev sẽ quyết định loại cấu hình, trừ mức tạo caption trùng hệt baseline ở mọi ảnh. Chưa có số về độ dài, ảnh co quá nửa, caption trùng hệt hoặc khoảng cách chỉnh sửa cho phép dò này.")
    add('''### 5.4. Đối chứng layer và ngân sách tính toán

Liệt kê **toàn bộ dải liền mạch cùng độ rộng** trên cùng tập con **125 ảnh dev** cố định; một tập dải vừa cho phân phối đối chứng vừa cho oracle ở đúng độ rộng đó. Độ rộng thứ nhất là độ rộng dải v2 của từng mô hình; nếu v2 không trả dải thì dùng 14 layer. Độ rộng thứ hai là ⌈L/4⌉: 6 layer ở SmolVLM2 và 8 layer ở TinyLLaVA. Với độ rộng w và L layer, số dải là L−w+1. Phân vị của dải v2 trong phân phối cùng độ rộng cần tính từ các lượt chạy dev; phép quét này **chưa chạy** vì cần sinh caption cho toàn bộ dải trên 125 ảnh, nên chưa có phân vị hoặc điểm oracle để báo. Oracle là **cận trên chọn trên dev** để mô tả; điểm CHAIR trong phép quét không được dùng để thay dải v2 hoặc chọn cấu hình đem sang eval. Hai nhánh định trước gồm **layer 5–18 tuyệt đối** và dải co theo độ sâu từ LLaVA 32 layer: lấy khoảng nửa mở [5,19), chuyển sang độ sâu L bằng `start=floor(5L/32)`, `end=ceil(19L/32)-1` (hai đầu tính vào). Vì vậy SmolVLM2 24 layer dùng dải co 3–14; TinyLLaVA 32 layer có dải co trùng 5–18. Điều kiện “mọi layer” và dải v2 được báo riêng.

Số giờ bên dưới là **quy đổi từ tốc độ pilot cùng mô hình/chế độ**: Smol detailed512 mặc định 20 ảnh, chia ô 768 và tắt chia ô mỗi chế độ 20 ảnh, POPE 60 câu hỏi mỗi biến thể; Tiny dùng baseline detailed512 trên 20 ảnh dev. Các ô “chưa ước tính” không được gán tốc độ của mô hình khác. Tổng giờ toàn chiến dịch chỉ xác định sau khi có tốc độ LLaVA, hai điều kiện prompt 64 token và phép đo attention chuyên biệt.''')
    add(f'SmolVLM2 (24 layer): độ rộng {smol_width} có {24-smol_width+1} dải, độ rộng hẹp {smol_narrow} có {24-smol_narrow+1} dải. TinyLLaVA (32 layer): độ rộng {tiny_width} còn {tiny_last_valid-tiny_width+2} dải, độ rộng hẹp {tiny_narrow} còn {tiny_last_valid-tiny_narrow+2} dải.'+(' Sáu dải TinyLLaVA chạm layer 29–31 bị loại; tổng còn 65 dải. Điều kiện “mọi layer” TinyLLaVA cũng bị loại, còn SmolVLM2 giữ nhánh này.' if tiny_last3_invalid else ' Tổng cộng 71 dải; điều kiện “mọi layer” còn trong kế hoạch.')+' Các số dải chỉ là liệt kê hình học; phân phối điểm trên 125 ảnh dev chưa chạy.')
    add(table('eda_results/hgai/report_layer_band_overlap.csv','Số layer chung giữa dải v2 và các dải định trước'))
    v2_dose=OUT/'v2_alpha_probe20_summary.csv'
    if v2_dose.exists():
        doses=pd.read_csv(v2_dose)
        assert len(doses)==10 and set(doses.model)=={'smol','tiny'}
        save('report_v2_alpha_probe20.csv',[{'Mô hình':r.model,'α':vi(r.alpha,4),
            'Tỷ lệ độ dài TB':vi(r.mean_length_ratio,2),
            'Trung vị tỷ lệ độ dài':vi(r.median_length_ratio,2),
            'Co quá nửa':ratio(int(r.less_than_half_count),20),
            'Caption trùng hệt':ratio(int(r.identical_caption_count),20),
            'Khoảng cách sửa từ TB':vi(r.mean_word_edit_distance,2),
            'Chạm trần 512':ratio(int(r.ceiling_512_count),20),
            'Có lặp 4-gram':ratio(int(r.repetition_4gram_count),20),
            'Phần 4-gram lặp TB':vi(r.mean_repeated_4gram_fraction,3)} for r in doses.itertuples()])
        add('Bảng sau dùng 20 ảnh dev đầu theo manifest cố định, prompt detailed512, giải mã greedy và đúng hai lưới α đã khóa; baseline và can thiệp được ghép theo ảnh. Kết quả chỉ mô tả độ dài/output, không chọn liều hoặc dải. Guard trên 500 ảnh dev vẫn là bước quyết định.')
        add(table('eda_results/hgai/report_v2_alpha_probe20.csv','Dò độ dài trên dải v2, 20 ảnh dev'))
        high=doses[(doses.model=='tiny') & (doses.alpha.between(.22,.23))].iloc[0]
        add(f'Ở TinyLLaVA α≈0,2236, {vi(int(high.ceiling_512_count))}/20 caption chạm trần và {vi(int(high.repetition_4gram_count))}/20 caption có lặp 4-gram. Đây là dấu hiệu suy giảm output ở dải v2 trên tập dò nhỏ; lưới α vẫn giữ nguyên và guard trên 500 ảnh dev quyết định cấu hình hợp lệ.')
    else:
        add('Dò độ dài trên chính dải v2 SmolVLM2 10–23 và TinyLLaVA 12–28, 20 ảnh dev, đúng các lưới α đã khóa: **chưa chạy**; cần sinh caption ghép cặp tại các dải mới. Lưới giữ nguyên; guard trên 500 ảnh dev vẫn là bước quyết định.')
    add('[Danh sách các dải hợp lệ](eda_results/hgai/layer_band_scan_plan.csv) và [125 ID dev seed 42](eda_results/hgai/layer_band_scan_dev125_ids.csv) xác định chính xác các lượt chạy đối chứng. Hai tệp này không chứa điểm CHAIR hoặc kết quả chọn dải.')
    add('Hai ô “Thời gian đã đo” của đường cong 100 ảnh là khoảng từ lúc tệp CSV bắt đầu được ghi đến lần ghi cuối; chúng không gồm nạp trọng số, chuẩn bị dữ liệu hoặc phân tích bootstrap, nên không thay thế tổng giờ quy đổi của chiến dịch.')
    add(table('eda_results/hgai/report_compute_budget.csv','Ngân sách theo thời gian pilot'))
    add(f"Tập con {vi(coarse)} ảnh dev dùng seed {design['coarse_scan']['seed']}; các cấu hình dẫn đầu sau quét thô mới chạy đủ {vi(full)} ảnh dev và qua guard trước hai bộ eval. Ngân sách gồm cả 3 mức token ảnh Smol, hai bộ eval, POPE và hàng LLaVA; số giờ là ước lượng quy mô, chưa phải thời gian hoàn thành đo được.")
    add('''### 5.5. POPE kiểm tra chéo

POPE dùng ba bộ COCO random, popular và adversarial chính thức, mỗi bộ 3.000 câu hỏi cho baseline và HGAI đã khóa từ dev. Giữ nhãn và negative sampling của POPE; báo accuracy, precision, recall, F1, yes-rate và số câu hỏi/ảnh. Không coi 500 ảnh CHAIR là một bộ POPE mới. Bootstrap POPE lấy mẫu theo **ảnh**, giữ mọi câu hỏi của cùng ảnh trong một cụm. POPE và hai bộ eval không dùng để chọn α hoặc dải layer.

### 5.6. Checkpoint TinyLLaVA

Đối chiếu hai định danh checkpoint Phi-2 và SigLIP.''')
    add(table('eda_results/hgai/report_tiny_checkpoint_comparison.csv','Đối chiếu checkpoint TinyLLaVA'))
    if tiny_loaded:
        smoke='Caption của ảnh thử trùng baseline.' if tiny_validation['caption_identical'] else 'Caption của ảnh thử khác baseline.'
        add(f"Ở ảnh dev thử với điều kiện detailed64 và α={vi(tiny_validation['alpha_tested'],2)}, adapter ghi {vi(tiny_validation['modified_calls'])} lần gọi attention được sửa. {smoke} [Dữ liệu kiểm chứng](eda_results/hgai/tiny_checkpoint_validation.json).")
    tiny_weights_ready=(OUT/'tiny_weights_provenance.json').exists()
    add(('Shard bczhou đã tải và đối chiếu SHA-256 tại máy này. ' if tiny_weights_ready else 'Shard bczhou đang tải; chưa đối chiếu đủ byte tại máy này. ')+ '[Nguồn và từng shard](eda_results/hgai/tiny_checkpoint_public_metadata.json).')
    add('Checkpoint được chọn: `bczhou/TinyLLaVA-3.1B` cho nhánh pad và adapter Phi-2 hiện tại. Cấu hình checkpoint `tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B` khai báo lớp mô hình riêng; kết luận về adapter của checkpoint này là suy ra từ cấu hình, chưa kiểm tra forward. [Cấu hình chính thức](https://huggingface.co/tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B/blob/a5501098449440fa319da7b0814dc53bf8530f14/config.json).')
    add('''## Phụ lục A. Bảng và hình nền

### A.1. Nhóm từ chỉ người

Đồng nghĩa CHAIR được khớp theo cụm dài trước trên một caption chọn cho mỗi ảnh.''')
    add(table('eda_results/hgai/report_person_reference.csv','Nhóm từ chỉ người COCO và VizWiz'))
    add('### A.2. Đủ 80 lớp COCO trong phân tích dịch chuyển')
    add(table('eda_results/hgai/report_object_shift.csv','Tỷ lệ ảnh chứa từng lớp COCO-80'))
    add('### A.3. Siêu danh mục và lệch tâm')
    add(table('eda_results/hgai/report_supercategories.csv','12 siêu danh mục COCO val2014'))
    add('![Phân bố độ lệch tâm bbox COCO](eda_results/hgai/coco_object_centering.png)')
    add('''## Tái lập

| Script | Bảng/hình hoặc dữ liệu sinh ra |
| :--- | :--- |
| `audit_report_additions.py`, `audit_smol_768.py` | Dịch chuyển object, độ phủ cắt giữa, phân bố token 768 |
| `audit_report_round.py` | Hình layer 20 ảnh, bảng độ nhạy v1 và chỉ số lặp SmolVLM2 |
| `run_alpha_probe20.py`, `summarize_alpha_probe20.py` | Caption và bảng liều SmolVLM2 |
| `run_tiny_alpha_probe20_v2.py`, `summarize_tiny_alpha_probe20_v2.py` | Caption, bảng liều và khóa lưới TinyLLaVA riêng |
| `lock_layer_rule_v2.py` | Tệp khóa v2 và SHA-256, tạo đúng một lần |
| `measure_layer_curves100.py`, `measure_layer_curves100_tiny_sdpa.py`, `analyze_layer_rule_v2.py` | Đường cong 100 ảnh, kết quả v1/v2, bootstrap 1.000 lần và hình; TinyLLaVA dùng forward SDPA và đọc attention float32 |
| `analyze_layer_diagnostics.py`, `validate_layer_attention.py` | Hình mức tăng độ nhạy, bảng chồng lấn và kiểm chứng VAR trên 20 ảnh dev |
| `capture_curve_windows.py` | Đóng băng thời gian ghi CSV từ ảnh đầu đến ảnh cuối của hai đường cong 100 ảnh |
| `check_tiny_last3_adapter.py`, `filter_layer_band_plan.py` | Kiểm tra NaN/Inf tại layer 29–31 và lọc kế hoạch dải TinyLLaVA |
| `run_v2_alpha_probe20.py`, `summarize_v2_alpha_probe20.py` | Caption và bảng dò độ dài trên dải v2 với lưới α giữ nguyên |
| `measure_var_all_steps_smol20.py` | Đường cong VAR trung bình mọi bước sinh |
| `enumerate_layer_band_plan.py` | Danh sách đầy đủ dải và 125 ID dev seed 42 |
| `build_hgai_report_final.py`, `build_hgai_notebook.py` | Các bảng `report_*.csv`, Markdown và notebook từ cùng nguồn |

Các phép sinh caption và đo attention cần checkpoint cùng GPU tương ứng; chỉ dựng lại báo cáo từ CSV/JSON đã đo không cần GPU.

Từ thư mục repo `_archive`, dựng lại `.md` và notebook bằng `./.venv-eda/Scripts/python.exe build_hgai_report_final.py` rồi `./.venv-eda/Scripts/python.exe build_hgai_notebook.py`. `publish_hgai_report.py` xuất hai bản đọc ở thư mục cha và kiểm tra các bảng/hình khớp nhau. Không chạy lại các script đo khi chỉ cần dựng lại báo cáo; manifest và các tệp khóa đang có là đầu vào cố định.

## Tài liệu nguồn

1. [HGAI: Devils in Middle Layers, bài báo](https://arxiv.org/abs/2411.16724), [mã đánh giá và data loader của tác giả](https://github.com/ZhangqiJiang07/middle_layers_indicating_hallucinations).
2. [SmolVLM2-2.2B-Instruct, model card và processor](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct); [recipe SmolVLM2](https://github.com/huggingface/smollm/tree/main/vision/smolvlm2).
3. [TinyLLaVA bczhou](https://huggingface.co/bczhou/TinyLLaVA-3.1B), [TinyLLaVA Phi-2 SigLIP](https://huggingface.co/tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B), [bài TinyLLaVA](https://arxiv.org/abs/2402.14289).
4. [LLaVA-1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b), [LLaVA mix665K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K), [ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V), [OneVision](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data).
5. [CHAIR](https://aclanthology.org/D18-1437/), [POPE và dữ liệu câu hỏi chính thức](https://github.com/RUCAIBox/POPE).
6. [COCO](https://cocodataset.org/), [Visual Genome](https://visualgenome.org/), [VizWiz](https://vizwiz.org/tasks-and-datasets/image-quality-issues/).''')
    REPORT.write_text('\n'.join(parts),encoding='utf-8')
    print('Wrote',REPORT,len(parts),'sections and tables')


if __name__=='__main__':main()
