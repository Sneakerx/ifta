"""Sanity-check citation keys in the paper match bib entries."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "paper"
tex = (ROOT / "overdrive_stability.tex").read_text()
bib = (ROOT / "overdrive_stability.bib").read_text()

cites = set()
for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", tex):
    for k in m.group(1).split(","):
        cites.add(k.strip())

bibkeys = set(re.findall(r"@\w+\{(\w+),", bib))
print(f"Citations in tex: {len(cites)}")
print(f"Bib entries: {len(bibkeys)}")
print(f"Missing from bib: {sorted(cites - bibkeys)}")
print(f"Unused bib entries: {sorted(bibkeys - cites)}")
