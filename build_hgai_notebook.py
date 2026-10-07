"""Build a portable notebook from the report's single source; tables execute beside prose."""
import base64
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'EDA_COCO_vs_VizWiz_Report.md'
TARGET=ROOT/'EDA_COCO_vs_VizWiz_Report.ipynb'


def markdown(source,attachments=None):
    return {'cell_type':'markdown','metadata':{},'source':source,'attachments':attachments or {}}


def code(source):
    return {'cell_type':'code','metadata':{},'source':source,'execution_count':None,'outputs':[]}


def markdown_cells(text):
    cells=[]
    # Split at major sections so reading and executing the notebook remains manageable.
    for section in re.split(r'(?m)(?=^## )',text):
        if not section.strip():continue
        attachments={}
        def embed(match):
            label,path=match.groups()
            asset=(ROOT/path).resolve()
            if not asset.is_relative_to(ROOT):raise ValueError('Image path outside project')
            if not asset.is_file():raise FileNotFoundError(asset)
            name=asset.name
            mime='image/png' if asset.suffix=='.png' else 'image/jpeg'
            if name in attachments:raise ValueError('Repeated image in section: '+name)
            attachments[name]={mime:base64.b64encode(asset.read_bytes()).decode('ascii')}
            return f'![{label}](attachment:{name})'
        section=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',embed,section)
        cells.append(markdown(section.strip(),attachments))
    return cells


def main():
    cells=[code('''from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "EDA_COCO_vs_VizWiz_Report.md").is_file())
assert (ROOT / "EDA_COCO_vs_VizWiz_Report.md").is_file()''')]
    text=REPORT.read_text(encoding='utf-8')
    # Preserve the exact tables emitted by build_hgai_report.table in both artifacts.
    cells.extend(markdown_cells(text))
    for i,cell in enumerate(cells):cell['id']=f'hgai-report-{i:03d}'
    nb={'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{
        'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},
        'hgai_report':{'source':'EDA_COCO_vs_VizWiz_Report.md'}}}
    TARGET.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    print('Notebook built:',len(cells),'cells; embedded figures exactly once.')


if __name__=='__main__':main()
