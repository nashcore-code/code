#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, struct
from pathlib import Path

OUT = struct.Struct('<QI4BdHbB4sdd8H')
if OUT.size != 64: raise RuntimeError(OUT.size)

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read_data(p:Path):
    raw=p.read_bytes()
    if len(raw)<8:raise ValueError(f'truncated {p}')
    n=struct.unpack_from('<Q',raw,0)[0]
    if len(raw)!=8+n*OUT.size:raise ValueError(f'size mismatch {p}: n={n} bytes={len(raw)}')
    return n,memoryview(raw)[8:]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('scan_summary',type=Path)
    ap.add_argument('cert_dir',type=Path)
    ap.add_argument('out_cert',type=Path)
    ap.add_argument('out_fail',type=Path)
    ap.add_argument('out_json',type=Path)
    a=ap.parse_args()
    S=json.loads(a.scan_summary.read_text())
    expected_cert={a.cert_dir/f"cert_{int(ch['offset']):07d}.bin" for ch in S['chunks']}
    expected_fail={a.cert_dir/f"fail_{int(ch['offset']):07d}.bin" for ch in S['chunks']}
    actual_cert=set(a.cert_dir.glob('cert_*.bin'))
    actual_fail=set(a.cert_dir.glob('fail_*.bin'))
    if actual_cert!=expected_cert or actual_fail!=expected_fail:
        mismatch=sorted(str(p) for p in (actual_cert^expected_cert)|(actual_fail^expected_fail))
        raise SystemExit(f'certificate chunk set does not match scan plan: {mismatch}')
    ncert=nfail=0; fixed=adaptive=0; by_k={}
    last=None
    a.out_cert.parent.mkdir(parents=True,exist_ok=True)
    with a.out_cert.open('wb') as co,a.out_fail.open('wb') as fo:
        co.write(struct.pack('<Q',0));fo.write(struct.pack('<Q',0))
        for ch in S['chunks']:
            tag=f"{ch['offset']:07d}"
            cp=a.cert_dir/f'cert_{tag}.bin'; fp=a.cert_dir/f'fail_{tag}.bin'
            nc,cd=read_data(cp); nf,fd=read_data(fp)
            if nc+nf!=ch['records']:
                raise SystemExit(f'proposal count mismatch offset {tag}: cert {nc}+fail {nf}!={ch["records"]}')
            for label,n,data in (('certificate',nc,cd),('failure',nf,fd)):
                for j in range(n):
                    vals=OUT.unpack_from(data,j*OUT.size)
                    key,h,k,bmask,flags,hard_reserved,eps,cm,deficit,typ,cert_reserved,sg,coal,*allcm=vals
                    if hard_reserved != 0:
                        raise SystemExit(f'nonzero embedded hard-record reserved byte in {label} {tag}, index {j}')
                    if cert_reserved != b'\0'*4:
                        raise SystemExit(f'nonzero certificate reserved bytes in {label} {tag}, index {j}')
                    if typ not in (0,1):
                        raise SystemExit(f'unknown certificate type {typ} in {label} {tag}, index {j}')
                    if typ==0 and not (0 <= deficit < 8):
                        raise SystemExit(f'bad fixed deficit {deficit} in {label} {tag}, index {j}')
                    if typ==1 and deficit != -1:
                        raise SystemExit(f'bad adaptive deficit {deficit} in {label} {tag}, index {j}')
                    if label == 'certificate':
                        rid=(key,k,h,bmask,flags)
                        if last is not None and rid<=last:
                            raise SystemExit(f'nonincreasing cert id {rid} after {last}')
                        last=rid
                        if typ==0: fixed+=1
                        else: adaptive+=1
                        z=by_k.setdefault(str(k),{'fixed':0,'adaptive':0})
                        z['fixed' if typ==0 else 'adaptive']+=1
            co.write(cd);fo.write(fd);ncert+=nc;nfail+=nf
        co.seek(0);co.write(struct.pack('<Q',ncert));fo.seek(0);fo.write(struct.pack('<Q',nfail))
    out={
      'hard_records':S['hard_record_count'],'certificate_records':ncert,'failure_records':nfail,
      'fixed_certificates':fixed,'adaptive_certificates':adaptive,'by_residual_budget':by_k,
      'certificate_file':str(a.out_cert),'certificate_file_bytes':a.out_cert.stat().st_size,'certificate_sha256':sha256(a.out_cert),
      'failure_file':str(a.out_fail),'failure_file_bytes':a.out_fail.stat().st_size,'failure_sha256':sha256(a.out_fail),
    }
    a.out_json.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True))
    if nfail: raise SystemExit(3)
    if ncert!=S['hard_record_count']:raise SystemExit(4)
if __name__=='__main__':main()
