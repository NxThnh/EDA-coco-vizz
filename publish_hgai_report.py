"""Build both final reader-facing artifacts at the clean workspace root."""
import json
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

import build_hgai_notebook
import build_hgai_report_final

ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parent
MARKDOWN = ROOT / 'EDA_COCO_vs_VizWiz_Report.md'
NOTEBOOK = ROOT / 'EDA_COCO_vs_VizWiz_Report.ipynb'


def table_shapes(source):
    result = []
    for marker, body in re.findall(r'<!-- CSV (.*?) -->\s*(.*?)<!-- /CSV -->', source, re.S):
        spec = json.loads(marker)
        assert (ROOT / spec['path']).is_file(), spec['path']
        lines = [line for line in body.splitlines() if line.startswith('|')]
        result.append((spec['title'], lines[0], len(lines) - 2))
    return result


def verify():
    md = MARKDOWN.read_text(encoding='utf-8')
    nb = nbformat.read(NOTEBOOK, as_version=4)
    text = '\n'.join(str(cell.source) for cell in nb.cells if cell.cell_type == 'markdown')
    bad = ['đã được đo', 'đã được thay', 'Bảng cũ', 'bộ cũ', 'lưới α cũ',
           'bản 8', 'bản 9', 'không còn', 'chỉ chèn một lần',
           'dưới đoạn giới thiệu này', 'không phải dòng bị sao chép',
           'người dùng', 'trợ lý', 'không cần lặp lại trong bảng']
    for name, source in [('Markdown', md), ('Notebook', text)]:
        hits = [phrase for phrase in bad if phrase.casefold() in source.casefold()]
        assert not hits, (name, hits)
    left, right = table_shapes(md), table_shapes(text)
    assert left == right
    main = md.split('## Phụ lục')[0]
    for old in ('eval_unmatched500', 'dev500'):
        assert not re.search(r'(?<!\w)' + old + r'(?!\w)', main)
    for sample in ('40,137', '38.07%', '0.244'):
        assert sample not in md and sample not in text
    figures = re.findall(r'!\[[^]]*\]\(attachment:([^)]+)\)', text)
    assert len(figures) == len(set(figures))
    assert all(not cell.get('outputs') or all(output.output_type != 'error' for output in cell.outputs)
               for cell in nb.cells if cell.cell_type == 'code')
    print('QA: forbidden phrases 0; matching tables', len(left),
          '; retired manifest mentions 0; bad number samples 0; unique attachments', len(figures))
    for title, header, count in left:
        print('TABLE', title, count, header)


def main():
    build_hgai_report_final.main()
    text = (ARCHIVE / MARKDOWN.name).read_text(encoding='utf-8')
    text = text.replace('eda_results/', '_archive/eda_results/')
    text = text.replace('experiment_outputs/', '_archive/experiment_outputs/')
    text = text.replace('data_prepared/', '_archive/data_prepared/')
    MARKDOWN.write_text(text, encoding='utf-8')
    build_hgai_notebook.ROOT = ROOT
    build_hgai_notebook.REPORT = MARKDOWN
    build_hgai_notebook.TARGET = NOTEBOOK
    build_hgai_notebook.main()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(notebook, timeout=120, kernel_name='python3',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.write(notebook, NOTEBOOK)
    verify()
    build_hgai_notebook.ROOT = ARCHIVE
    build_hgai_notebook.REPORT = ARCHIVE / MARKDOWN.name
    build_hgai_notebook.TARGET = ARCHIVE / NOTEBOOK.name
    build_hgai_notebook.main()
    archive_notebook = nbformat.read(build_hgai_notebook.TARGET, as_version=4)
    NotebookClient(archive_notebook, timeout=120, kernel_name='python3',
                   resources={'metadata': {'path': str(ARCHIVE)}}).execute()
    nbformat.write(archive_notebook, build_hgai_notebook.TARGET)
    import export_hgai_html
    export_hgai_html.main()


if __name__ == '__main__':
    main()
