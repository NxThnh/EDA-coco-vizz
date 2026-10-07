"""Final parity and editorial checks for both generated report artifacts."""
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent if (HERE.parent/'EDA_COCO_vs_VizWiz_Report.md').is_file() else HERE
md=(ROOT/'EDA_COCO_vs_VizWiz_Report.md').read_text(encoding='utf-8')
nb=json.loads((ROOT/'EDA_COCO_vs_VizWiz_Report.ipynb').read_text(encoding='utf-8'))
notebook='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='markdown')
bad=['đã được đo','đã được thay','Bảng cũ','bộ cũ','lưới α cũ','bản 8','bản 9',
     'không còn','chỉ chèn một lần','dưới đoạn giới thiệu này',
     'không phải dòng bị sao chép','người dùng','trợ lý','không cần lặp lại trong bảng']
for name,source in [('Markdown',md),('Notebook',notebook)]:
    hits=[x for x in bad if x.casefold() in source.casefold()]
    print(name,'cụm cấm:',len(hits),hits)
    assert not hits

def tables(text):
    output=[]
    for marker,body in re.findall(r'<!-- CSV (.*?) -->\s*(.*?)<!-- /CSV -->',text,re.S):
        spec=json.loads(marker)
        assert (ROOT/spec['path']).is_file(),spec['path']
        lines=[line for line in body.splitlines() if line.startswith('|')]
        output.append((spec['title'],lines[0],len(lines)-2))
    return output

left,right=tables(md),tables(notebook)
assert left==right
print('Bảng khớp:',len(left))
for title,header,count in left:print(f'  {title}: {count} dòng; {header}')
main=md.split('## Phụ lục')[0]
for name in ['eval_unmatched500','dev500']:
    assert not re.search(r'(?<!\w)'+name+r'(?!\w)',main)
print('Manifest ngừng dùng trong phần chính: 0')
for bad_number in ['40,137','38.07%','0.244']:
    assert bad_number not in md and bad_number not in notebook
print('Mẫu định dạng số sai: 0')
images=re.findall(r'!\[[^]]*\]\(attachment:([^)]+)\)',notebook)
assert len(images)==len(set(images))
print('Hình attachment:',len(images),'mỗi hình một lần')
