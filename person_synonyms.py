"""One shared lexical definition of person, matching Figure 4's CHAIR synonyms."""
import re
from pathlib import Path

synonyms={}
for line in (Path(__file__).resolve().parent/'eda_resources/chair_synonyms.txt').read_text(encoding='utf-8').splitlines():
    terms=[term.strip().casefold() for term in line.split(',') if term.strip()]
    if terms:synonyms.update((term,terms[0]) for term in terms)
pattern=re.compile(r'\b(?:'+'|'.join(re.escape(term) for term in sorted(synonyms,key=len,reverse=True))+r')\b',re.I)


def person_mentions(caption):
    return sum(synonyms[m.group().casefold()]=='person' for m in pattern.finditer(caption))
