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
import json, html, subprocess, sys
import pandas as pd
from IPython.display import display, HTML
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "hgai_cpu_audit.py").is_file())
# The report normally reads audited outputs. Enable only for a deliberate CPU rebuild.
REBUILD = False
if REBUILD:
    cpu_python = ROOT / ".venv-eda/Scripts/python.exe"
    for script in ["hgai_cpu_audit.py", "verify_tiny_processor.py", "small_vlm_training_overlap.py",
                   "caption_sampling_sensitivity.py", "vizwiz_rotation_review.py", "sampling_fix_diagnostics.py"]:
        subprocess.run([str(cpu_python), str(ROOT/script)], cwd=ROOT, check=True)
pd.set_option("display.max_columns", None)
def render_csv(relative, columns=None):
    frame = pd.read_csv(ROOT/relative)
    if columns:
        frame = frame[columns]
    display(HTML('<div class="audit-table">' + frame.to_html(index=False, escape=True, na_rep='—',
                      float_format=lambda v: f'{v:.3f}') + '</div><style>.audit-table{overflow-x:auto;max-width:100%}'
                 '.audit-table td,.audit-table th{padding:6px;white-space:normal;min-width:6em;max-width:24em}</style>'))
print("Measured HGAI EDA artifacts loaded; completed GPU pilots are reported separately from full HGAI experiments.")''')]
    text=REPORT.read_text(encoding='utf-8')
    pos=0
    for match in re.finditer(r'<!-- CSV (.*?) -->\n.*?<!-- /CSV -->',text,re.S):
        cells.extend(markdown_cells(text[pos:match.start()]))
        spec=json.loads(match.group(1))
        cells.append(code('render_csv('+repr(spec['path'])+', '+repr(spec['columns'])+')'))
        pos=match.end()
    cells.extend(markdown_cells(text[pos:]))
    for i,cell in enumerate(cells):cell['id']=f'hgai-v9-{i:03d}'
    nb={'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{
        'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},
        'hgai_report':{'version':9,'source':'EDA_COCO_vs_VizWiz_Report.md','date':'2026-10-06'}}}
    TARGET.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    print('Notebook built:',len(cells),'cells; embedded figures exactly once.')


if __name__=='__main__':main()
