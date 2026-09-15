"""Refuse to run against a different inherited implementation."""
import hashlib,json
from pathlib import Path

def verify(team):
    pins=json.loads((Path(__file__).parent/'compatibility.json').read_text())
    bad=[]
    for rel,expected in pins.items():
        path=Path(team)/rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:bad.append(rel)
    if bad:raise RuntimeError('SOURCE_CHANGED_SINCE_CAPTURE: '+', '.join(bad)+'. Nothing was installed/commanded; do not bypass this check.')
    return True
