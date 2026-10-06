"""Regression checks for split leakage, processor counts and portable report output."""
import ast
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, unquote

import pandas as pd
from PIL import Image, ImageOps

from hgai_cpu_audit import ROOT,OUT,PREP,sha,prepare
from lock_common_unmatched import prepare_common, FLAGS


def main():
    lock=prepare()
    common_lock=prepare_common()
    partitions={}
    verified_git_sources=[]
    if (ROOT/'.git').exists():
        # Check bytes that a fresh checkout receives, not only this machine's CSVs.
        for name,digest in lock['source_sha256'].items():
            blob=subprocess.run(['git','cat-file','blob',':'+name],cwd=ROOT,capture_output=True)
            if blob.returncode==0:
                assert hashlib.sha256(blob.stdout).hexdigest()==digest, 'Git source bytes differ from locked source: '+name
                verified_git_sources.append(name)
    for filename,digest in {**lock['manifest_sha256'], **common_lock['manifest_sha256']}.items():
        path=PREP/filename
        assert sha(path)==digest
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==500
        assert len({r['image_id'] for r in rows})==500
        assert all((ROOT/r['image_path']).is_file() for r in rows)
        for row in rows:
            image_path=ROOT/row['image_path']
            assert hashlib.md5(image_path.read_bytes()).hexdigest()==row['md5'], 'Image bytes changed: '+row['image_path']
            with Image.open(image_path) as image:
                assert ImageOps.exif_transpose(image).size==(row['width'],row['height'])
        partitions[path.stem]=rows
    for a,ra in partitions.items():
        for b,rb in partitions.items():
            if a>=b:continue
            assert not {r['image_id'] for r in ra}&{r['image_id'] for r in rb}
            assert not {r['md5'] for r in ra}&{r['md5'] for r in rb}
    assert all(not r['llava_known_match'] for r in partitions['eval_unmatched500'])
    frame=pd.read_csv(OUT/'processor_per_image.csv')
    assert len(frame)==7500
    assert frame.groupby(['partition','model','mode']).size().eq(500).all()
    assert not frame.duplicated(['partition','image_id','model','mode']).any()
    assert set(frame.partition)==set(partitions)
    for (partition,model,mode),group in frame.groupby(['partition','model','mode']):
        expected={r['image_id']:(r['width'],r['height']) for r in partitions[partition]}
        assert set(group.image_id)==set(expected), (partition,model,mode)
        assert all((r.width,r.height)==expected[r.image_id] for r in group.itertuples())
    assert (frame[frame['mode']=='split_off'].image_tokens==81).all()
    smol=frame[frame.model=='SmolVLM2-2.2B']
    assert (smol.image_tokens==smol.tiles_including_global*81).all()
    assert (frame[frame.model=='TinyLLaVA-3.1B'].image_tokens==729).all()
    assert (frame.distortion_factor>=1).all()
    summary=pd.read_csv(OUT/'image_token_summary.csv')
    for _,row in summary.iterrows():
        selected=frame[(frame.partition==row.partition)&(frame.model==row.model)&(frame['mode']==row['mode'])]
        assert abs(selected.image_tokens.mean()-row['mean'])<1e-9
    overlap=pd.read_csv(OUT/'small_vlm_training_overlap.csv')
    assert len(overlap)==40504
    assert overlap.llava_sft_union.sum()==16618
    assert overlap.tiny_share_recipe_known.sum()==37527
    for part in ['eval_common_unmatched500', 'dev_common_unmatched500']:
        ids={r['image_id'] for r in partitions[part]}
        selected=overlap[overlap.image_id.isin(ids)]
        assert len(selected)==500 and selected[FLAGS].eq(0).all().all()
        assert selected.partition.eq(part).all()
        assert all(not r[flag] for r in partitions[part] for flag in FLAGS)
    assert common_lock['total_common_unmatched']==2977
    config=json.loads((OUT/'experiment_config.json').read_text())
    assert config['main_partition']=='eval_common_unmatched500'
    assert config['dev_partition']=='dev_common_unmatched500'
    assert config['paper_protocol_partition']=='eval_random500'
    assert config['published_reference_metrics']['CHAIR_S_greedy_64_pct']==23.8
    if (OUT/'onevision_id_provenance.json').exists():
        p=json.loads((OUT/'onevision_id_provenance.json').read_text())
        assert len(p)==8,'OneVision ID audit incomplete; do not publish a full-source count'
        raw_ids=pd.read_csv(ROOT/'scratch/onevision_sharegpt4v_ids.csv',dtype=str)
        assert sum(r['rows'] for r in p)==len(raw_ids)
        assert set(raw_ids.source_slice)=={'sharegpt4v(coco)','sharegpt4v(llava)'}
        coco_ids={int(i) for i in raw_ids[raw_ids.source_slice=='sharegpt4v(coco)'].source_id if i.isdigit()}
        assert overlap.smol_checked_upstream_candidate.sum()==overlap.image_id.isin(coco_ids).sum(), 'Leakage summary must refresh after ID fetch completes'
    person=pd.read_csv(ROOT/'eda_results/tables/person_reference_one_caption_per_image.csv')
    assert list(person['Selected Captions with Person Family (%)'])==[37.52,8.66]
    mentions=pd.read_csv(ROOT/'eda_results/tables/coco80_caption_mentions.csv')
    for _,r in person.iterrows():
        column='coco_images_with_mention' if r.Dataset=='MS-COCO val2014' else 'vizwiz_images_with_mention'
        g=mentions[mentions.category=='person']
        assert int(r['Selected Captions with Person Family'])==int(g.iloc[0][column])
    equivalence=json.loads((OUT/'tiny_processor_equivalence.json').read_text())
    assert equivalence['images']==20 and equivalence['max_absolute_difference']<=1e-6
    environment=json.loads((OUT/'processor_environment.json').read_text())
    assert environment['processor_images']==2500
    assert environment['common_manifest_lock_sha256']==sha(PREP/'manifest_lock_common_v9.json')
    assert environment['geometry_postprocessing']['code_sha256']==sha(ROOT/'hgai_geometry_finalize.py')
    captions=pd.read_csv(OUT/'caption_tokens_per_image.csv')
    assert len(captions)==2*(40504+7542)
    assert not captions.duplicated(['dataset','model','image_id']).any()
    review=pd.read_csv(OUT/'vizwiz_rotation_review40.csv')
    assert len(review)==40 and review.review_status.eq('reviewed_by_assistant_visual').all()
    report=(ROOT/'EDA_COCO_vs_VizWiz_Report.md').read_text(encoding='utf-8')
    assert '30,92' not in report and '30.92' not in report
    assert '41,028%' in report
    local_links=[]
    for match in re.finditer(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',report):
        link=match.group(1)
        if not link.startswith(('https://','http://','#')):
            assert (ROOT/link).exists(),link
            local_links.append(link)
    nb=json.loads((ROOT/'EDA_COCO_vs_VizWiz_Report.ipynb').read_text(encoding='utf-8'))
    assert nb['metadata']['hgai_report']['version']==9
    attachments=Counter()
    codes=[]
    for cell in nb['cells']:
        source=cell['source'] if isinstance(cell['source'],str) else ''.join(cell['source'])
        if cell['cell_type']=='code':
            ast.parse(source)
            assert cell['execution_count'] is not None,'Execute notebook in a fresh kernel before validation'
            assert all(o['output_type']!='error' for o in cell['outputs'])
            codes.append(cell['execution_count'])
        else:
            assert not re.search(r'!\[[^\]]*\]\((?!attachment:)',source),'Relative image remains in notebook'
            for name,data in cell.get('attachments',{}).items():
                attachments[name]+=1
                assert any(data.values())
                assert source.count('attachment:'+name)==1
    assert all(n==1 for n in attachments.values())
    assert attachments['08_clip_domain_alignment.png']==1 and attachments['09_vizwiz_quality_labels.png']==1
    assert codes==list(range(1,len(codes)+1))
    for filename in ['hgai_cpu_audit.py','hgai_geometry_finalize.py','small_vlm_training_overlap.py',
                     'fetch_hgai_assets.py','fetch_training_metadata.py','fetch_onevision_ids.py',
                     'build_hgai_report.py','build_hgai_notebook.py','llava_pilot.py','person_synonyms.py',
                     'caption_sampling_sensitivity.py','eda_coco_vs_vizwiz.py','vizwiz_rotation_review.py',
                     'lock_common_unmatched.py','sampling_fix_diagnostics.py','fetch_smol_baseline.py',
                     'fetch_hgai_metrics.py','smol_baseline.py','score_hgai_baseline.py','run_smol_pilots.py',
                     'export_hgai_html.py','rebuild_hgai.py','run_eda_notebook.py']:
        ast.parse((ROOT/filename).read_text(encoding='utf-8'),filename=filename)
    html_path=OUT/'EDA_HGAI_Small_VLM_Report.html'
    html_local_links=[]
    if html_path.exists():
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(html_path.read_text(encoding='utf-8'),'html.parser')
        figures=[im for im in soup.find_all('img') if im.get('src','').startswith('data:image/')]
        assert len(figures)==len(attachments)
        assert not [im for im in soup.find_all('img') if not im.get('src','').startswith('data:image/')]
        for anchor in soup.find_all('a',href=True):
            link=urlsplit(anchor['href'])
            if link.scheme or link.netloc or not link.path:continue
            assert (html_path.parent/unquote(link.path)).exists(),anchor['href']
            html_local_links.append(anchor['href'])
        assert len(html_local_links)==len(local_links)
    scored_runs=[]
    from score_hgai_baseline import parse_pope, chair_class, score_pope, score_chair
    assert [parse_pope(a) for a in ['Yes.', 'No.', 'There is not a car.', 'yes, there is.']]==[1,0,0,1]
    import nltk
    nltk.data.path.insert(0,str(ROOT/'model_assets/nltk_data'))
    evaluator=chair_class()([],str(ROOT/'coco/annotations'))
    assert evaluator.caption_to_words('A baby bird beside a wine glass and a toilet seat.')[1]==['bird','wine glass','toilet']
    assert evaluator.caption_to_words('Two women ride bicycles.')[1]==['person','bicycle']
    for path in (ROOT/'experiment_outputs/smol_baseline').glob('*.metrics.json'):
        output=path.with_name(path.name.replace('.metrics.json','.jsonl'))
        metric=json.loads(path.read_text())
        run_config=json.loads(output.with_suffix('.config.json').read_text())
        rows=[json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==run_config['requested_rows']
        assert metric['output_sha256']==sha(output)
        assert metric['config_sha256']==sha(output.with_suffix('.config.json'))
        assert metric['model_precision']==run_config['precision']
        assert metric['evaluator_code_sha256']==sha(ROOT/'score_hgai_baseline.py')
        if run_config['requested_rows']<run_config['benchmark_rows']:
            assert metric['role']=='feasibility_pilot'
        if metric['task']=='chair':
            manifest=PREP/(run_config['partition']+'.jsonl')
            original=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
            assert sha(manifest)==run_config['manifest_sha256']
            assert [r['image_id'] for r in rows]==[r['image_id'] for r in original[:len(rows)]]
            fresh,details=score_chair(rows)
            for key in ['hallucinated_captions','hallucinated_object_mentions','generated_object_mentions',
                        'CHAIR_S_pct','CHAIR_I_pct','object_micro_recall','object_micro_precision','object_micro_F1']:
                assert fresh[key]==metric[key], (path,key)
            assert abs(metric['CHAIR_S_pct']-100*metric['hallucinated_captions']/len(rows))<1e-9
            if metric['generated_object_mentions']:
                assert abs(metric['CHAIR_I_pct']-100*metric['hallucinated_object_mentions']/metric['generated_object_mentions'])<1e-9
        else:
            variant=run_config['partition'].removeprefix('official-pope-')
            manifest=ROOT/'eda_resources/pope_official'/('coco_pope_'+variant+'.json')
            assert sha(manifest)==run_config['manifest_sha256']
            original=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
            for generated,expected in zip(rows,original):
                assert generated['question_id']==expected['question_id'] and generated['label']==expected['label']
                assert generated['prompt']==expected['text']
                assert generated['image_id']==int(Path(expected['image']).stem.split('_')[-1])
            fresh,_=score_pope(rows)
            assert all(fresh[key]==metric[key] for key in fresh),path
            assert sum(metric[k] for k in ['TP','TN','FP','FN'])==len(rows)
            assert abs(metric['precision']-metric['TP']/(metric['TP']+metric['FP']))<1e-9
            assert all(0<=metric[k]<=1 for k in ['accuracy','precision','recall','F1','yes_rate'])
        scored_runs.append(output.relative_to(ROOT).as_posix())
    result={'status':'passed','locked_partitions':len(partitions),'images_per_partition':500,
            'actual_image_bytes_verified':sum(len(rows) for rows in partitions.values()),
            'git_locked_sources_verified':verified_git_sources,
            'processor_rows':len(frame),'caption_token_rows':len(captions),
            'executed_notebook_code_cells':len(codes),'embedded_figures':len(attachments),
            'local_report_links_checked':len(local_links),'verified_scored_runs':scored_runs,
            'local_html_links_checked':len(html_local_links),
            'report_sha256':sha(ROOT/'EDA_COCO_vs_VizWiz_Report.md'),
            'notebook_sha256':sha(ROOT/'EDA_COCO_vs_VizWiz_Report.ipynb')}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
