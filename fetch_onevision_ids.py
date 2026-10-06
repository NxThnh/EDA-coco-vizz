"""Read only parquet ID columns using HTTP range requests, never image columns."""
import io
import json
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pyarrow.parquet as pq
import requests

ROOT=Path(__file__).resolve().parent


class RangeFile(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.pos=0;self.downloaded=0;self.session=requests.Session()
        for attempt in range(6):
            try:
                r=self.session.get(url,headers={'Range':'bytes=0-0'},timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt==5:raise
                time.sleep(2)
        if r.status_code!=206 or 'Content-Range' not in r.headers:
            raise ValueError('Server lacks HTTP ranges; refusing full image parquet download')
        self.size=int(r.headers['Content-Range'].split('/')[-1])
        self.direct_url=r.url
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        n=min(n if n>=0 else self.size-self.pos,self.size-self.pos)
        if n==0:return b''
        if n>16*1024*1024:raise ValueError('Requested range too large for metadata-only read')
        for attempt in range(8):
            try:
                r=self.session.get(self.direct_url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},timeout=90)
                r.raise_for_status()
                if r.status_code!=206 or len(r.content)!=n:raise ValueError('HTTP range response mismatch')
                self.pos+=n;self.downloaded+=n
                return r.content
            except requests.RequestException:
                if attempt==7:raise
                time.sleep(2)


def main():
    info=json.loads((ROOT/'scratch/LLaVA-OneVision-Data_dataset_api.json').read_text())
    ids_path=ROOT/'scratch/onevision_sharegpt4v_ids.csv'
    prov_path=ROOT/'eda_results/hgai/onevision_id_provenance.json'
    rows=pd.read_csv(ids_path,dtype=str).to_dict('records') if ids_path.exists() else []
    provenance=json.loads(prov_path.read_text()) if prov_path.exists() else []
    finished={p['filename'] for p in provenance}
    for item in info['siblings']:
        name=item['rfilename']
        if not name.startswith(('sharegpt4v(coco)/','sharegpt4v(llava)/')) or not name.endswith('.parquet'):continue
        if name in finished:continue
        print('Reading ID metadata:',name,flush=True)
        url=f"https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data/resolve/{info['sha']}/{quote(name)}"
        with RangeFile(url) as source:
            frame=pq.ParquetFile(source).read(columns=['id'],use_threads=False).to_pandas()
            rows.extend({'source_slice':name.split('/')[0],'source_id':str(i)} for i in frame.id)
            provenance.append({'repo':'lmms-lab/LLaVA-OneVision-Data','revision':info['sha'],
                               'filename':name,'rows':len(frame),'range_bytes_read':source.downloaded,
                               'columns':['id'],'note':'Upstream candidate pool, not released exact SmolVLM2 sampled training manifest.'})
            print(name,len(frame),'metadata bytes',source.downloaded,flush=True)
        pd.DataFrame(rows).to_csv(ROOT/'scratch/onevision_sharegpt4v_ids.csv',index=False,encoding='utf-8-sig')
        (ROOT/'eda_results/hgai/onevision_id_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')


if __name__=='__main__':main()
