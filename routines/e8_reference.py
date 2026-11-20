#!/usr/bin/env python3
"""CPU-only exact E8/4E8 golden. Integer arithmetic uses doubled lattice coordinates.
No device compilation or simulator evidence. Quantizer searches the finite canonical
codebook exactly; shaping an infinite-lattice nearest point is only a heuristic.
"""
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np

G2 = np.zeros((8,8), dtype=np.int64)
G2[0] = 1
for j in range(1,7):
    G2[j,j] = 2; G2[j,j+1] = -2
G2[7,6:8] = 2

def lex_greater(a,b):
    unequal = a != b
    first = np.argmax(unequal,axis=-1)
    return unequal.any(axis=-1) & (np.take_along_axis(a,first[...,None],-1)[...,0] > np.take_along_axis(b,first[...,None],-1)[...,0])

def nearest_e8(num,den=1):
    """Return doubled nearest E8 point to num/den; ties smallest residual lex.
    Rational input and squared distances are exact int64. Translation-equivariant.
    """
    num=np.asarray(num,dtype=np.int64)
    best=None; bestdist=None
    for c in (0,1):
        # n = k+c/2, sum(k) even. Individual ties round upward.
        k=np.floor_divide(2*num-c*den+den,2*den)
        odd=(k.sum(axis=-1)&1)!=0
        for j in range(-1,8):
            for delta in ((0,) if j==-1 else (-1,1)):
                cand=2*k+c
                if j>=0: cand=cand.copy(); cand[...,j]+=2*delta
                valid=~odd if j==-1 else odd
                dist=np.sum((2*num-den*cand)**2,axis=-1)
                if best is None:
                    best=cand.copy(); bestdist=np.where(valid,dist,np.iinfo(np.int64).max)
                else:
                    choose=valid & ((dist<bestdist)|((dist==bestdist)&lex_greater(cand,best)))
                    best=np.where(choose[...,None],cand,best); bestdist=np.where(choose,dist,bestdist)
    return best

def coefficients(labels):
    labels=np.asarray(labels,dtype=np.int64)
    a=np.empty(labels.shape+(8,),dtype=np.int64)
    a[...,0]=2*((labels>>14)&1)+((labels>>15)&1)-2
    for j in range(1,8): a[...,j]=((labels>>(2*(j-1)))&3)-2
    return a

def shape2(z2): return z2-4*nearest_e8(z2,8)
def decode2(labels): return shape2(coefficients(labels)@G2)
def inverse_coeff(q2):
    a=np.empty_like(q2); a[...,0]=q2[...,0]
    t=(q2[...,1:]-q2[...,:1])//2
    a[...,1:6]=np.cumsum(t[...,:5],axis=-1)
    a[...,6]=(t[...,:6].sum(axis=-1)-t[...,6])//2
    a[...,7]=(t[...,:6].sum(axis=-1)+t[...,6])//2
    return a

def pack(q2):
    a=inverse_coeff(q2); r=(a[...,0]+2)%4
    idx=((r&1)<<15)|((r>>1)<<14)
    for j in range(1,8): idx|=((a[...,j]+2)%4)<<(2*(j-1))
    return idx.astype(np.uint16)

def bf16_rne(x):
    x=np.asarray(x,dtype=np.float32); u=x.view(np.uint32)
    return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
def int8_rne(x): return np.clip(np.rint(x),-128,127).astype(np.int8)

def finite_quant(x,codebook,batch=16):
    """Exact finite exhaustive search; tie smallest uint16 label, no full LUT on device implied."""
    x=np.asarray(x,dtype=np.float64); ids=[]
    for start in range(0,len(x),batch):
        d=np.sum((x[start:start+batch,None,:]-codebook[None,:,:])**2,axis=-1)
        ids.append(np.argmin(d,axis=-1))
    return np.concatenate(ids).astype(np.uint16)

def metrics(x,y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64); e=y-x
    energy=float(np.sum(x*x)); squared=float(np.sum(e*e))
    cosine=float(np.sum(x*y)/max(float(np.linalg.norm(x)*np.linalg.norm(y)),1e-30))
    return dict(mse=float(np.mean(e*e)),rmse=float(np.sqrt(np.mean(e*e))),max_abs=float(np.max(np.abs(e))),relative_l2=float(np.linalg.norm(e)/max(np.linalg.norm(x),1e-30)),nmse=squared/max(energy,1e-30),cosine=cosine)

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--fixture-output',type=Path); parser.add_argument('--stress-fixture',action='store_true'); args=parser.parse_args()
    start=time.perf_counter(); rng=np.random.default_rng(9503510)
    manifest=dict(schema='e8_v1',seed=9503510,numpy_version=np.__version__,python_version=platform.python_version(),source=str(Path(__file__).resolve()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scale_order='rows contiguous [32,256]; NumPy float32 rows*rows, np.mean(axis=1,dtype=float32), np.sqrt(dtype=float32); NumPy vectorized/pairwise reduction, not sequential accumulation; zero RMS replaced with float32 one',normalization='float32 rows / float32 scale, then exact finite-codebook squared distances evaluated in float64 on those normalized FP32 values; tie smallest uint16 label',dataset_provenance='default_rng PCG64 seed9503510; algebra shift/subset draws,48 normal probes,24 scaled normal probes precede datasets; BF16 normal8192 float32 then RNE; int8 uniform[-128,128) int16 then int8',files=[])
    def emit(name,array,semantic_dtype=None):
        if args.fixture_output is None: return
        args.fixture_output.mkdir(parents=True,exist_ok=True); array=np.ascontiguousarray(array); raw=array.tobytes(); path=args.fixture_output/name; path.write_bytes(raw)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==hashlib.sha256(raw).hexdigest()
        manifest['files'].append(dict(file=name,dtype=semantic_dtype or array.dtype.str,storage_dtype=array.dtype.str,shape=list(array.shape),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    def bf16_bits(x): return (bf16_rne(np.asarray(x,dtype=np.float32)).view(np.uint32)>>16).astype('<u2')
    labels=np.arange(65536,dtype=np.int64); a=coefficients(labels); z2=a@G2; q2=decode2(labels)
    # Byte tables partition the identical linear projection, before nonlinear shaping.
    lowa=coefficients(np.arange(256)); lowa[:,[0,5,6,7]]=0
    hia=coefficients(np.arange(256)<<8); hia[:,1:5]=0
    low=lowa@G2; high=hia@G2
    fact=low[labels&255]+high[labels>>8]
    emit('basis_doubled.i8',G2.astype('i1')); emit('basis.bf16le',bf16_bits(G2/2),'bfloat16-le')
    for name,table in (('low',low),('high',high)):
        emit(name+'_partial_doubled.i8',table.astype('i1')); emit(name+'_partial.bf16le',bf16_bits(table/2),'bfloat16-le')
    if args.stress_fixture: emit('stress_indices.u16le',labels.astype('<u2'))
    all_even=(q2%2==0).all(axis=1); all_odd=(q2%2!=0).all(axis=1)
    checks=dict(unique_count=int(np.unique(q2,axis=0).shape[0]),membership=bool(np.all((all_even|all_odd)&((q2.sum(axis=1)%4)==0))),inverse_pack=bool(np.array_equal(pack(q2),labels)),canonical_fixed=bool(np.array_equal(shape2(q2),q2)),factor_projection=bool(np.array_equal(z2,fact)),factor_shaping=bool(np.array_equal(shape2(fact),q2)),generator_det=float(np.linalg.det(G2/2)))
    checks['basis_inverse']=bool(np.array_equal(inverse_coeff(z2),a))
    checks['factor_bf16_exact']=bool(np.array_equal(bf16_rne(low.astype(np.float32)/2),low/2) and np.array_equal(bf16_rne(high.astype(np.float32)/2),high/2))
    checks['bf16_rne_ties']=bool(np.array_equal(bf16_rne(np.array([1+2**-8,1+3*2**-8],np.float32)),np.array([1,1+2**-6],np.float32)))
    checks['int8_rne_saturation']=bool(np.array_equal(int8_rne(np.array([-200.,-2.5,-1.5,1.5,2.5,200.])),np.array([-128,-2,-2,2,2,127],np.int8)))
    shifts=rng.integers(-3,4,(24,8))@G2
    subset=z2[rng.integers(0,65536,128)]
    checks['periodicity']=all(np.array_equal(shape2(subset+4*s),shape2(subset)) for s in shifts)
    checks['nearest_translation_true_e8']=all(np.array_equal(nearest_e8(subset+4*s,8),nearest_e8(subset,8)+s) for s in shifts)
    checks['nearest_translation_half_basis']=bool(np.array_equal(nearest_e8(subset+4*G2[0],8),nearest_e8(subset,8)+G2[0]))
    checks['nearest_translation']=all(np.array_equal(nearest_e8(subset+8*s,8),nearest_e8(subset,8)+2*s) for s in shifts)
    codebook=q2.astype(np.float64)/2
    checks['finite_zero_tie_smallest_label']=bool(finite_quant(np.zeros((1,8)),codebook)[0]==np.flatnonzero(np.all(q2==0,axis=1))[0])
    checks['finite_midpoint_tie_smallest_label']=bool(finite_quant(np.array([[0.5,0.5,0,0,0,0,0,0]]),codebook)[0]==np.flatnonzero(np.sum((codebook-np.array([0.5,0.5,0,0,0,0,0,0]))**2,axis=1)==0.5)[0])
    random=rng.normal(size=(48,8)); adversarial=np.vstack([np.zeros((1,8)),np.full((1,8),20.),np.full((1,8),-20.),np.eye(8)*8,-np.eye(8)*8,rng.normal(size=(24,8))*6])
    probes=np.vstack([random,adversarial]); exact=finite_quant(probes,codebook)
    # Rational probe rounding is only for the naive counterexample comparator.
    naive2=shape2(nearest_e8(np.rint(probes*1048576).astype(np.int64),1048576))
    naive=pack(naive2); err=np.sum((probes-codebook[exact])**2,axis=1); naiveerr=np.sum((probes-naive2/2)**2,axis=1)
    mismatch=np.flatnonzero(naiveerr>err+1e-9)
    fidelity={}; cpu={}
    for kind in ('bf16','int8'):
        original=bf16_rne(rng.normal(size=8192).astype(np.float32)) if kind=='bf16' else rng.integers(-128,128,8192,dtype=np.int16).astype(np.int8)
        rows=original.astype(np.float32).reshape(-1,256)
        # Explicit FP32 accumulation and sqrt; zero scale is one.
        scale=np.sqrt(np.mean(rows*rows,axis=1,dtype=np.float32),dtype=np.float32); scale=np.where(scale==0,np.float32(1),scale)
        normalized=(rows/scale[:,None]).reshape(-1,8)
        t=time.perf_counter(); ids=finite_quant(normalized,codebook); cpu[kind+'_finite_quant_seconds']=time.perf_counter()-t
        reconstructed=(codebook[ids].reshape(-1,256)*scale[:,None]).astype(np.float32)
        output=bf16_rne(reconstructed) if kind=='bf16' else int8_rne(reconstructed)
        emit(kind+'_input.'+('bf16le' if kind=='bf16' else 'i8'),bf16_bits(rows) if kind=='bf16' else rows.astype('i1'),'bfloat16-le' if kind=='bf16' else None)
        emit(kind+'_indices.u16le',ids.astype('<u2').reshape(32,32))
        emit(kind+'_scales.f32le',scale.astype('<f4'))
        emit(kind+'_normalized.f32le',normalized.astype('<f4').reshape(32,256))
        emit(kind+'_golden_unrounded.f32le',reconstructed.astype('<f4'))
        emit(kind+'_dequant.'+('bf16le' if kind=='bf16' else 'i8'),bf16_bits(output) if kind=='bf16' else output.astype('i1'),'bfloat16-le' if kind=='bf16' else None)
        emit(kind+'_aux_unscaled_z.i8',q2[ids].astype('i1').reshape(32,256))
        emit(kind+'_aux_scales.f32le',(scale/np.float32(2)).astype('<f4'))
        fidelity[kind]=dict(physical_output=metrics(rows,output),pre_output_cast=metrics(rows,reconstructed),scale_min=float(scale.min()),scale_max=float(scale.max()))
    zero=np.zeros((1,256),np.float32); zs=np.sqrt(np.mean(zero*zero,axis=1,dtype=np.float32)); zs=np.where(zs==0,np.float32(1),zs)
    checks['zero_scale_one']=bool(zs[0]==1)
    checks['aux_int8_z_exact']=bool(np.array_equal(q2,q2.astype(np.int8).astype(np.int64)))
    assert checks['unique_count']==65536 and all(v for k,v in checks.items() if k not in ('unique_count','generator_det'))
    report=dict(status='CPU_ONLY_PASS',checks=checks,codebook=dict(contract='E8/4E8 canonical Voronoi representatives',generator=G2.tolist(),generator_units='half',tie='nearest residual lexicographically smallest; finite quant smallest uint16 label',factor_tables_bf16_bytes=8192,q2_min=int(q2.min()),q2_max=int(q2.max())),finite_quant=dict(cases=len(probes),naive_strictly_worse_cases=len(mismatch),example=None if not len(mismatch) else dict(input=probes[mismatch[0]].tolist(),exact_label=int(exact[mismatch[0]]),naive_label=int(naive[mismatch[0]]),exact_squared_error=float(err[mismatch[0]]),naive_squared_error=float(naiveerr[mismatch[0]]))),fidelity=fidelity,auxiliary_abi='int8 unscaled z=2q, physical scale=s/2; distinct from physical-scale baseline output',cpu_timings=cpu,total_cpu_seconds=time.perf_counter()-start,limitations=['CPU NumPy evidence only; no ccec, CAModel, or hardware results','finite exhaustive quantizer is a correctness oracle, not a production-performance claim','naive comparator inputs rationalized to 2^-20; exact finite distance search uses float64 inputs'])
    if args.fixture_output is not None:
        manifest['validation_checks']=checks; manifest['fidelity']=fidelity
        (args.fixture_output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        report['fixture_manifest']=str(args.fixture_output/'manifest.json')
    out=Path(__file__).resolve().parents[1]/'reports/e8_reference_validation.json'; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
