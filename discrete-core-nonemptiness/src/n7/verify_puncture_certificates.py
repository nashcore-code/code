#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

path=Path(sys.argv[1] if len(sys.argv)>1 else "data/surplus_puncture_certificates.json")
data=json.loads(path.read_text())
assert data["count"]==len(data["cells"])==298
for idx,c in enumerate(data["cells"]):
    m=c["m"]; k=c["kappa"]; supports=c["supports"]; h=c["floor"]; ws=c["puncture_witnesses"]
    assert len(supports)==m and len(h)==7 and len(ws)==7
    def util(chosen):
        return [sum((supports[q]>>i)&1 for q in chosen) for i in range(7)]
    # The stored floor itself must not be implementable.  Exhaustive because m<=7.
    from itertools import combinations
    assert not any(all(u>=v for u,v in zip(util(S),h)) for S in combinations(range(m),k)), idx
    for i,S in enumerate(ws):
        assert len(S)<=k and len(set(S))==len(S)
        target=h.copy();target[i]-=1
        assert all(u>=v for u,v in zip(util(S),target)), (idx,i)
print("Verified 298 primitive holes and all 2,086 singleton puncture witnesses.")
