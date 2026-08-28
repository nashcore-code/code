#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re
from fractions import Fraction
from pathlib import Path
PAT=re.compile(r'^PASS certs=(\d+) fixed=(\d+) adaptive=(\d+) saturation_skips=(\d+) open_floor_checks=(\d+) exact_price_LPs=(\d+) min_singleton_or_sum=(\S+) min_exact_price=(\S+)$')
def frac(s):return None if s=='NA' else Fraction(s)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('dir',type=Path);ap.add_argument('glob');ap.add_argument('out',type=Path);a=ap.parse_args()
 sums=dict(certs=0,fixed=0,adaptive=0,saturation_skips=0,open_floor_checks=0,exact_price_LPs=0);mins={'min_singleton_or_sum':None,'min_exact_price':None};files=[]
 for p in sorted(a.dir.glob(a.glob)):
  lines=p.read_text(errors='strict').splitlines();hits=[PAT.match(x) for x in lines];hits=[x for x in hits if x]
  if len(hits)!=1:raise SystemExit(f'expected one PASS line in {p}, found {len(hits)}')
  m=hits[0];vals=list(m.groups());
  for key,v in zip(sums,vals[:6]):sums[key]+=int(v)
  for key,v in zip(mins,vals[6:]):
   z=frac(v)
   if z is not None and (mins[key] is None or z<mins[key]):mins[key]=z
  files.append(p.name)
 out={**sums,**{k:('NA' if v is None else f'{v.numerator}/{v.denominator}') for k,v in mins.items()},'log_count':len(files),'logs':files}
 a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
