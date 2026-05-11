"""Rough word count of the paper."""

import re
from pathlib import Path

text = (Path(__file__).resolve().parent.parent / "paper" / "overdrive_stability.tex").read_text()
text = re.sub(r"%.*", "", text)
text = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", " ", text)
text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
text = re.sub(r"\{|\}", " ", text)
words = [w for w in text.split() if w and any(c.isalpha() for c in w)]
print(f"Words: {len(words)}")
print(f"Approx pages at 600 wpp: {len(words) / 600:.1f}")
