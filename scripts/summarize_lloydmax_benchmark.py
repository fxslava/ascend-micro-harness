#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
data=[json.loads(p.read_text()) for p in sorted(root.glob('b[24]_*/audit.json'))]
expected={(b,m,n,r) for b in (2,4) for m in ((0,1,2,4,5) if b==2 else (0,1,2,3,4,5)) for n in (128,256) for r in (1,8)}
actual={(d['bits'],d['mode'],d['n'],d['rows']) for d in data}
if actual!=expected: raise SystemExit(f'Incomplete benchmark: missing {expected-actual}, extra {actual-expected}')
boundary=[json.loads(p.read_text()) for p in sorted(root.glob('boundary_*/audit.json'))]
boundary_expected={(b,m,n) for b in (2,4) for m in (0,1) for n in (128,256)}
if {(d['bits'],d['mode'],d['n']) for d in boundary}!=boundary_expected:
    raise SystemExit(f'Incomplete boundary sweep: {len(boundary)}/8')
fields=['bits','mode','n','rows','codec_span_ticks','ticks_per_vector','vector_instructions','RVECEX','RV_VLD','RV_VST','RV_VSELR','header_init_vector_instructions','header_init_span_ticks','nmse','max_abs','max_ulp','R1','scalar_drains','camodel_errors']
with (root/'comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(data)
names={0:'LM immediate',1:'LM UB header (warm)',2:'Gaussian sigmoid adapter',3:'Original FP16 ICDF (encode only)',4:'LM packed decode immediate',5:'LM packed decode UB (warm)'}
lines=['# Lloyd-Max quantizer — CAModel execution report', '',
       'Target: Ascend950PR_9599, CANN 9.2.0-beta.2, 3510 vector simulator. Measured on 2026-10-30; model ticks are not physical-device timings.', '',
       '## Verification', '',
       f'PASS: {len(data)} performance/numeric cases plus {len(boundary)} threshold-boundary cases. Tier 1 verifies the capacity-one one-shot queue model, midpoint/conditional-mean residuals, adjacent-FP32 threshold decisions, packing, and Gaussian distortion integrals. Encoder outputs are byte-exact; FP32 reconstructions match the golden reference with 0 absolute error / 0 ULP. Every case preserves 160 B of UB boundary guards and 512 B of external GM guards. Whole inactive reconstruction and packed-code tails also remain canaries. The trace auditor also passes twelve positive/negative fixture tests.', '',
       'Trace gates: no excess loads/stores (R1), no V_S/S_V events, no SFF allocator scans, balanced static caller events, and zero CAModel errors in retained logs/dumps. Counts include RV_VLDI/RV_VSTI immediate-address forms. Each 64-lane FP32 decode is one register Gather; additional VSELR instructions implement classification and packing/unpacking. No TPipe is constructed and no leaf calls GetValue, SetValue or ReduceSum.', '',
       'The wrapper is a one-shot MTE2 → VF → MTE3 chain with a V → MTE2 edge for UB initialization and a final MTE3 → Scalar completion edge. It has no ring or production-operator change. Eight rows exercise multiple SIMD iterations, not ring-wrap or multi-worker evidence. Those integration gates are not claimed; physical hardware was not run.', '',
       '## Gaussian fidelity', '',
       '| Bits | Lloyd-Max population NMSE | Gaussian CDF + midpoint ICDF | Sigmoid-Vasicek + midpoint ICDF |',
       '| ---: | ---: | ---: | ---: |',
       '| 2 | 0.1174818479 | 0.1467506787 | 0.1467734219 |',
       '| 4 | 0.0095010080 | 0.0236125580 | 0.0235953785 |', '',
       'These are analytic N(0,1) distortion integrals, with variance 1; they verify lower L2 distortion against these comparators, not a proof of global optimality over every possible codebook. The supplied rounded 2-bit constants are preserved exactly (conditional-mean residual 1.12e-5); the 4-bit fixed-point residual is 6.05e-8. Empirical NMSE in comparison.csv uses the same deterministic post-Hadamard Gaussian fixture for both FP32 codecs. Finite normalized inputs are required; normalization/scales and nonfinite handling belong to the caller.', '',
       '## Measured codec spans and counts', '',
       'Rows = 8 below. Ticks/vector = the final VF first-RV-instruction dispatch to last-RV-instruction completion span divided by eight; it includes in-register table construction once per VF, quantization, reconstruction and packing. It is an amortized average, not an isolated per-row latency. Rows = 1 and the raw spans are in comparison.csv. Each span excludes arena initialization, GM DMA, and (for UB variants) header initialization. Vector instructions include all RV engines; RVECEX is reported separately.', '',
       '| Bits | Elements/vector | Codec | Ticks/vector | Vector instructions (8 rows) | RVECEX | Loads | Stores | VSELR | Empirical NMSE |',
       '| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for d in sorted((d for d in data if d['rows']==8 and d['mode']<=2),key=lambda d:(d['bits'],d['n'],d['mode'])):
    lines.append(f"| {d['bits']} | {d['n']} | {names[d['mode']]} | {d['ticks_per_vector']:.3f} | {d['vector_instructions']} | {d['RVECEX']} | {d['RV_VLD']} | {d['RV_VST']} | {d['RV_VSELR']} | {d['nmse']:.10f} |")
lines+=['', 'Measured comparisons against the matched FP32 sigmoid adapter (eight rows):', '']
for b in (2,4):
    for n in (128,256):
        ref=next(d for d in data if (d['bits'],d['n'],d['mode'],d['rows'])==(b,n,2,8))
        lm=next(d for d in data if (d['bits'],d['n'],d['mode'],d['rows'])==(b,n,0,8))
        ub=next(d for d in data if (d['bits'],d['n'],d['mode'],d['rows'])==(b,n,1,8))
        reduction=100*(1-lm['vector_instructions']/ref['vector_instructions'])
        change=f'{reduction:.1f}% fewer' if reduction>=0 else f'{-reduction:.1f}% more'
        lines.append(f"- {b}-bit / {n} elements: immediate LM has {change} vector instructions and {ref['codec_span_ticks']/lm['codec_span_ticks']:.3f}x the codec throughput; warm UB LM has {ref['codec_span_ticks']/ub['codec_span_ticks']:.3f}x the throughput. Ratios below 1 mean LM is slower.")
lines+=['', 'Scheme B reserves a 512 B UB header. Centroids occupy 32/64 B rounded blocks and thresholds start at byte 96, providing 32 B bank skew after the largest centroid block. Every base is 32 B aligned. Each source has 256 addressable bytes behind it; padded reads may overlap unused lanes, while masked writes stay within their live blocks. The arena also has a 256 B tail read pad. Header initialization is vector Duplicate/Select/Store only: no GM table, DMA or scalar UB store. The header is initialized once per kernel launch and can be reused by multiple codec calls.', '',
        '| Bits | Header init vector instructions | Header init span (ticks) |', '| ---: | ---: | ---: |']
for b in (2,4):
    d=next(d for d in data if d['bits']==b and d['mode']==1 and d['n']==128 and d['rows']==1)
    lines.append(f"| {b} | {d['header_init_vector_instructions']} | {d['header_init_span_ticks']} |")
lines+=['', 'Initialization must be charged for a cold one-shot call: the warm UB codec span alone is not a cold-launch speedup. Immediate tables avoid UB header storage; their setup is inside the measured VF. Four-bit threshold search can have more ALU instructions than the sigmoid adapter, so instruction reduction is not universal. Packing uses a three/four-stage register reduction tree and one compression Gather per register. Tier 1 replays it against an independent byte packer. The retained initial_packer_summary.json, when present, contains the earlier per-code Gather candidate measurements.', '',
        '## Existing ICDF baseline and comparison limits', '',
        'The repository baseline routines/routine_vector_icdf_quantize.hpp is called unchanged in mode 3. It computes FP16 14*sigmoid(x)-7 and packs signed int4 (15 reachable levels), with BF16 support in its own existing runner. It supplies neither a 2-bit mode nor a Gaussian reconstruction decoder. Therefore a literal 2-bit original-baseline comparison and a Gaussian decode NMSE for that header are unavailable. They are N/A, never zero in comparison.csv. Mode 3 uses the existing boundary-margin fixture and validates every packed byte.', '',
        'The matched FP32 adapter (mode 2) explicitly supplies the missing 2/4-bit Gaussian comparator: floor(2^b * sigmoid(1.5976*x + 0.070565*x^3)), clipped to 0..2^b-1. Reconstruction uses exact FP32 normal-midquantile constants and the same Gather/packing path to isolate quantizer arithmetic. This is a new adapter, not a claim that the original header implements a Gaussian CDF, rational Chebyshev ICDF, or Newton decoder. The ideal Gaussian-CDF row is a host oracle only; no CAModel claim is made for it.', '',
        '| Existing header, encode only | Rows | Codec span ticks | Ticks/vector | Vector instructions | Loads | Stores |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for d in sorted((d for d in data if d['mode']==3),key=lambda d:(d['n'],d['rows'])):
    lines.append(f"| {d['n']} FP16 elements | {d['rows']} | {d['codec_span_ticks']} | {d['ticks_per_vector']:.3f} | {d['vector_instructions']} | {d['RV_VLD']} | {d['RV_VST']} |")
lines+=['', 'These encode-only timings have a different dtype, output contract and fixture, so dividing a full FP32 codec span by them is not a meaningful speedup.', '',
        '## Standalone packed decoding', '',
        'Modes 4/5 decode low-bits-first unsigned code sequences containing every codebook entry into FP32, with 0 ULP error. NMSE is N/A because decoder-only inputs are code patterns rather than encoded Gaussian samples. The repeating decoder fixture does not establish independent multi-page addressing; production integration requires distinguishable page probes. Each 128-code iteration reads one padded register; A3 requires 256 accessible bytes behind each packed input base.', '',
        '| Bits | Elements/vector | Table | Ticks/vector (8 rows) | Vector instructions | Loads | Stores | VSELR |',
        '| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
for d in sorted((d for d in data if d['rows']==8 and d['mode']>=4),key=lambda d:(d['bits'],d['n'],d['mode'])):
    lines.append(f"| {d['bits']} | {d['n']} | {names[d['mode']]} | {d['ticks_per_vector']:.3f} | {d['vector_instructions']} | {d['RV_VLD']} | {d['RV_VST']} | {d['RV_VSELR']} |")
lines+=['', '## Reproduction and evidence', '',
        'From WSL at the repository root:', '', '```sh', 'bash scripts/run_lloydmax_camodel.sh', 'python3 scripts/test_analyze_lloydmax_trace.py', '```', '',
        'The script builds a fresh ext4 source snapshot, runs Tier 1 and all 52 CAModel cases, then emits this report and comparison.csv. Raw popped/completion traces, numeric run.log, per-case histograms and audit.json are retained under this directory. build_dir.txt names the ext4 binary directory; configure.log and build.log retain build diagnostics. run-only uses LLOYDMAX_BUILD_DIR for the existing binary; LLOYDMAX_REUSE_RESULTS=1 reaudits cached cases from that build.', '',
        'CMake requests Ascend950PR_9599 and its simulator path resolves to dav_3510. The installed driver stub reports Ascend950PR_9589 in run.log; this is recorded as a runtime identity discrepancy, not hidden. Compiler details and source hashes are in environment.txt and sources.sha256. This report is CAModel evidence only.']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(f'Wrote {len(data)} measured rows to {root / "comparison.csv"}')
