#!/usr/bin/env python3
"""Fail-closed census of the final VF (codec), separate from UB/header init."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

LINE = re.compile(r'\[(\d+)\]\s+\(PC:\s*(0x[\da-f]+)\)\s+(\w+)\s*:\s*\(Binary:\s*\w+\)\s+\(ID:\s*(\d+)\)\s+(\S+)',re.I)
VF = re.compile(r'\bVF\s+addr:\s*(0x[\da-f]+),instr_num:\s*(0x[\da-f]+)',re.I)
FLAG = re.compile(r'(SET_FLAG|WAIT_FLAG)\s+PIPE:(\w+),\s*TRIGGER PIPE:(\w+),\s*FLAG ID:?\s*(\w+)',re.I)

def read(path):
    rows=[]
    for line in path.read_text(errors='replace').splitlines():
        m=LINE.search(line)
        if m:
            t,pc,engine,ident,op=m.groups()
            rows.append((int(t),int(pc,16),engine.upper(),int(ident),op.upper(),line))
    return rows

def audit(directory,bits,mode,n,rows):
    popped=[p for p in directory.glob('**/core*.veccore*.instr_popped_log.dump')
            if p.stat().st_size and any(VF.search(r[5]) for r in read(p))]
    if len(popped)!=1: raise ValueError(f'expected one active core, got {len(popped)} dumps')
    p=popped[0]; disp=read(p); done=read(p.with_name(p.name.replace('instr_popped_log','instr_log')))
    done_by_id={r[3]:r for r in done}
    vfs=[VF.search(r[5]) for r in disp if VF.search(r[5])]
    if not vfs: raise ValueError('no VF dispatch: no evidence')
    start,count=(int(v,16) for v in vfs[-1].groups())
    codec=[r for r in disp if start<=r[1]<start+4*count and r[2].startswith('RV')]
    if not codec: raise ValueError('empty codec VF')
    if any(r[3] not in done_by_id for r in codec): raise ValueError('missing codec completion')
    hist=Counter(r[4] for r in codec); engines=Counter(r[2] for r in codec)
    # CANN uses immediate-address encodings for header accesses. Count both
    # encodings; omitting VLDI/VSTI would silently hide spills in a census.
    loads=hist['RV_VLD']+hist['RV_VLDI'];stores=hist['RV_VST']+hist['RV_VSTI']
    expected_ld=n*rows//(128 if mode>=3 else 64)+((2 if bits==4 else 1) if mode==1 else (1 if mode==5 else 0))
    expected_st=n*rows//128 if mode==3 else n*rows//64+(0 if mode>=4 else n*rows*bits//256)
    if loads!=expected_ld or stores!=expected_st:
        raise ValueError(f'R1: LD/ST {loads}/{stores}, expected {expected_ld}/{expected_st}')
    events=Counter()
    for r in disp:
        # SBITSET alone also configures CTRL in InitSocState; it is not an
        # allocation. SFF is the allocator scan in the documented trap.
        if r[2]=='SCALAR' and r[4]=='SFF':
            raise ValueError('dynamic event allocator on scalar pipeline')
        f=FLAG.search(r[5])
        if f:
            kind,src,dst,flag=f.groups(); key=(src.upper(),dst.upper(),int(flag,0))
            events[(kind.upper(),key)]+=1
            if {src.upper(),dst.upper()}=={'VEC','SCALAR'}:
                raise ValueError('V_S or S_V scalar drain')
    routes={k[1] for k in events}
    if not routes: raise ValueError('no caller event evidence')
    for route in routes:
        if events[('SET_FLAG',route)]!=events[('WAIT_FLAG',route)]: raise ValueError(f'unbalanced {route}')
    # Codebook and arena init are reported too, but not charged to steady codec.
    allvec=[r for r in disp if r[2].startswith('RV')]
    log=(directory/'run.log').read_text(errors='replace')
    verdict=re.search(r'PASS bits=(\d+) mode=(\d+) n=(\d+) rows=(\d+) boundaries=(\d+)',log)
    if not verdict or tuple(map(int,verdict.groups()[:4]))!=(bits,mode,n,rows):
        raise ValueError('missing numeric verdict or mismatched case identity')
    if re.search(r'\[(?:error|fatal)\]|invalid ldst addr|su_ccu_mpu_err',log,re.I):
        raise ValueError('CAModel error')
    for file in directory.glob('**/*'):
        if file.is_file() and file.suffix in ('.log','.dump') and file.name!='run.log':
            if re.search(r'\[(?:error|fatal)\]|invalid ldst addr|su_ccu_mpu_err',file.read_text(errors='replace'),re.I):
                raise ValueError(f'CAModel error in {file}')
    match=re.search(r'PASS .*nmse=([\d.e+-]+).*max_abs=([\d.e+-]+).*max_ulp=(\d+)',log)
    if not match: raise ValueError('missing numeric/guard PASS')
    span=max(done_by_id[r[3]][0] for r in codec)-min(r[0] for r in codec)
    header=[]
    if mode in (1,5):
        hs,hc=(int(v,16) for v in vfs[-2].groups())
        header=[r for r in allvec if hs<=r[1]<hs+4*hc]
        if not header or any(r[3] not in done_by_id for r in header):
            raise ValueError('missing header initialization evidence')
        if sum(r[4] in ('RV_VST','RV_VSTI') for r in header)!=2:
            raise ValueError('header must store two live table blocks')
    header_span=max((done_by_id[r[3]][0] for r in header),default=0)-min((r[0] for r in header),default=0)
    vse=hist['RV_VSELR']
    expected_vse=0 if mode==3 else (n*rows//32 if mode>=4 else n*rows//64*((6 if bits==2 else 9) if mode<2 else (6 if bits==2 else 5)))
    # The first 4-bit search index is lane-invariant (7). BiSheng may CSE
    # that threshold Gather across registers and hoist it out of the loop.
    minimum_vse=expected_vse-(n*rows//64) if bits==4 and mode<2 else expected_vse
    if not minimum_vse<=vse<=expected_vse:
        raise ValueError(f'Gather census {vse}, expected {minimum_vse}..{expected_vse}')
    return dict(bits=bits,mode=mode,n=n,rows=rows,boundary_case=bool(int(verdict[5])),codec_span_ticks=span,
        ticks_per_vector=span/rows,vector_instructions=len(codec),RVECEX=engines['RVECEX'],
        RV_VLD=loads,RV_VST=stores,RV_VSELR=vse,
        init_and_codec_vector_instructions=len(allvec),nmse=float(match[1]) if mode<3 else None,
        header_init_vector_instructions=len(header),header_init_span_ticks=header_span,
        max_abs=float(match[2]) if mode!=3 else None,max_ulp=int(match[3]) if mode!=3 else None,R1='PASS',scalar_drains=0,camodel_errors=0,
        histogram=dict(hist),vf_pc=hex(start),vf_instruction_words=count)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path)
    for k in ('bits','mode','n','rows'): ap.add_argument('--'+k,type=int,required=True)
    a=ap.parse_args()
    try:
        result=audit(a.directory,a.bits,a.mode,a.n,a.rows)
        (a.directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result));return 0
    except (ValueError,OSError) as e: print(f'FAIL trace audit: {e}',file=sys.stderr);return 1
if __name__=='__main__': sys.exit(main())
