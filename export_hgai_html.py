"""Export embedded notebook figures and resolve document links from the HTML folder."""
import os
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter

from hgai_cpu_audit import ROOT, OUT


def main():
    notebook = nbformat.read(ROOT / 'EDA_COCO_vs_VizWiz_Report.ipynb', as_version=4)
    html, _ = HTMLExporter(exclude_input=True).from_notebook_node(notebook)
    target = OUT / 'EDA_HGAI_Small_VLM_Report.html'
    soup = BeautifulSoup(html, 'html.parser')
    corrected = 0
    for anchor in soup.find_all('a', href=True):
        link = urlsplit(anchor['href'])
        if link.scheme or link.netloc or not link.path:
            continue
        asset = (ROOT / unquote(link.path)).resolve()
        if not asset.is_relative_to(ROOT) or not asset.exists():
            raise ValueError('Broken local report link: ' + anchor['href'])
        relative = os.path.relpath(asset, target.parent).replace('\\', '/')
        anchor['href'] = urlunsplit(('', '', quote(relative, safe='/'), link.query, link.fragment))
        corrected += 1
    target.write_text(str(soup), encoding='utf-8')
    print(f'HTML exported with embedded figures and {corrected} verified local document links.')


if __name__ == '__main__':
    main()
