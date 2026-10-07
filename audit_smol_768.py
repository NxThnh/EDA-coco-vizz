"""Count actual SmolVLM2 image placeholders for three locked manifests on CPU."""
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps
from transformers import SmolVLMProcessor

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'eda_results/hgai'
names=['dev_common_unmatched500','eval_common_unmatched500','eval_random500']
processor=SmolVLMProcessor.from_pretrained(ROOT/'model_assets/SmolVLM2-2.2B-Instruct',local_files_only=True)
processor.image_processor.do_image_splitting=True
processor.image_processor.size={'longest_edge':768}
image_id=processor.tokenizer.convert_tokens_to_ids('<image>')
rows=[]
for name in names:
    manifest=pd.read_csv(ROOT/'data_prepared/hgai'/f'{name}.csv')
    for index,row in enumerate(manifest.itertuples(index=False),1):
        with Image.open(ROOT/row.image_path) as raw:
            image=ImageOps.exif_transpose(raw).convert('RGB')
        encoded=processor(images=[image],text='<image>Describe the image.',return_tensors='pt')
        count=int((encoded['input_ids']==image_id).sum().item())
        assert count==int(encoded['pixel_values'].shape[1])*processor.image_seq_len
        rows.append({'partition':name,'image_id':row.image_id,'model':'SmolVLM2-2.2B',
                     'mode':'split_768','image_tokens':count})
        if index%100==0: print(name,index,flush=True)
frame=pd.DataFrame(rows)
frame.to_csv(OUT/'smol_tokens_768_per_image.csv',index=False,encoding='utf-8-sig')
print('Saved',len(frame),'measured rows',flush=True)
