#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, re, struct
from pathlib import Path

HARD_MAGIC = 0x3843454C4C533031
REC = struct.Struct('<QIBBBBd')  # key,h,k,Bmask,flags,reserved,eps

STAT_RE = re.compile(r'^k=(\d+)\s+(.*)$')
HEAD_RE = re.compile(r'^input=(.*?)\s+offset=(\d+)\s+matrices=(\d+)\s+records=(\d+)$')

def parse_out(p: Path):
    lines=p.read_text(errors='strict').splitlines()
    if not lines: raise ValueError(f'empty log {p}')
    m=HEAD_RE.match(lines[0])
    if not m: raise ValueError(f'bad first line {p}: {lines[0]!r}')
    inp,off,nmat,nrec=m.groups(); stats={}
    for line in lines[1:]:
        z=STAT_RE.match(line)
        if not z: continue
        k=int(z.group(1)); vals={}
        for item in z.group(2).split():
            a,b=item.split('=',1); vals[a]=int(b)
        stats[k]=vals
    return {'path':p,'input':inp,'offset':int(off),'matrices':int(nmat),'records':int(nrec),'stats':stats}

def sha256(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('chunk_dir',type=Path)
    ap.add_argument('out_hard',type=Path)
    ap.add_argument('out_json',type=Path)
    ap.add_argument('--total',type=int,required=True)
    a=ap.parse_args()
    chunks=[]
    for p in a.chunk_dir.glob('chunk_*.out'):
        chunks.append(parse_out(p))
    chunks.sort(key=lambda x:x['offset'])
    cur=0
    for c in chunks:
        if c['offset']!=cur:
            raise SystemExit(f'coverage error: expected offset {cur}, found {c["offset"]} in {c["path"]}')
        cur += c['matrices']
    if cur!=a.total: raise SystemExit(f'coverage ends at {cur}, expected {a.total}')

    aggregate={k:{} for k in range(2,7)}
    total_records=0
    last_id=None
    seen_ids=0
    a.out_hard.parent.mkdir(parents=True,exist_ok=True)
    with a.out_hard.open('wb') as out:
        out.write(struct.pack('<QQ',HARD_MAGIC,0))
        for c in chunks:
            bp=c['path'].with_suffix('.bin')
            raw=bp.read_bytes()
            if len(raw)<16: raise SystemExit(f'truncated {bp}')
            magic,n=struct.unpack_from('<QQ',raw,0)
            if magic!=HARD_MAGIC: raise SystemExit(f'bad magic {bp}: {magic:#x}')
            if n!=c['records']: raise SystemExit(f'count mismatch {bp}: header {n}, log {c["records"]}')
            if len(raw)!=16+n*REC.size: raise SystemExit(f'size mismatch {bp}: {len(raw)} vs {16+n*REC.size}')
            payload=memoryview(raw)[16:]
            # Verify deterministic local/global ordering and no duplicate record id.
            for j in range(n):
                key,h,k,bmask,flags,reserved,eps=REC.unpack_from(payload,j*REC.size)
                if reserved != 0:
                    raise SystemExit(f'nonzero hard-record reserved byte at {bp}, index {j}')
                rid=(key,k,h,bmask,flags)
                if last_id is not None and rid<=last_id:
                    raise SystemExit(f'non-increasing or duplicate record at {bp}, index {j}: {rid} after {last_id}')
                last_id=rid; seen_ids+=1
            out.write(payload)
            total_records += n
            for k,s in c['stats'].items():
                agg=aggregate.setdefault(k,{})
                for name,v in s.items(): agg[name]=agg.get(name,0)+v
        out.seek(8); out.write(struct.pack('<Q',total_records))
    if total_records!=seen_ids: raise AssertionError
    summary={
        'kernel_count':a.total,
        'chunk_count':len(chunks),
        'hard_record_count':total_records,
        'hard_record_size':REC.size,
        'coverage':[0,a.total],
        'by_residual_budget':{str(k):v for k,v in sorted(aggregate.items()) if v},
        'hard_file':str(a.out_hard),
        'hard_file_bytes':a.out_hard.stat().st_size,
        'hard_file_sha256':sha256(a.out_hard),
        'input_sha256':sha256(Path(chunks[0]['input'])),
        'chunks':[{'offset':c['offset'],'matrices':c['matrices'],'records':c['records'],'file':c['path'].with_suffix('.bin').name} for c in chunks],
    }
    a.out_json.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:summary[k] for k in ('kernel_count','chunk_count','hard_record_count','hard_file_bytes','hard_file_sha256')},indent=2))
    for k,v in summary['by_residual_budget'].items():
        print('k='+k, ' '.join(f'{x}={y}' for x,y in v.items()))

if __name__=='__main__': main()
