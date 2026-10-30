# Lloyd-Max quantizer — CAModel execution report

Target: Ascend950PR_9599, CANN 9.2.0-beta.2, 3510 vector simulator. Measured on 2026-10-30; model ticks are not physical-device timings.

## Verification

PASS: 44 performance/numeric cases plus 8 threshold-boundary cases. Tier 1 verifies the capacity-one one-shot queue model, midpoint/conditional-mean residuals, adjacent-FP32 threshold decisions, packing, and Gaussian distortion integrals. Encoder outputs are byte-exact; FP32 reconstructions match the golden reference with 0 absolute error / 0 ULP. Every case preserves 160 B of UB boundary guards and 512 B of external GM guards. Whole inactive reconstruction and packed-code tails also remain canaries. The trace auditor also passes twelve positive/negative fixture tests.

Trace gates: no excess loads/stores (R1), no V_S/S_V events, no SFF allocator scans, balanced static caller events, and zero CAModel errors in retained logs/dumps. Counts include RV_VLDI/RV_VSTI immediate-address forms. Each 64-lane FP32 decode is one register Gather; additional VSELR instructions implement classification and packing/unpacking. No TPipe is constructed and no leaf calls GetValue, SetValue or ReduceSum.

The wrapper is a one-shot MTE2 → VF → MTE3 chain with a V → MTE2 edge for UB initialization and a final MTE3 → Scalar completion edge. It has no ring or production-operator change. Eight rows exercise multiple SIMD iterations, not ring-wrap or multi-worker evidence. Those integration gates are not claimed; physical hardware was not run.

## Gaussian fidelity

| Bits | Lloyd-Max population NMSE | Gaussian CDF + midpoint ICDF | Sigmoid-Vasicek + midpoint ICDF |
| ---: | ---: | ---: | ---: |
| 2 | 0.1174818479 | 0.1467506787 | 0.1467734219 |
| 4 | 0.0095010080 | 0.0236125580 | 0.0235953785 |

These are analytic N(0,1) distortion integrals, with variance 1; they verify lower L2 distortion against these comparators, not a proof of global optimality over every possible codebook. The supplied rounded 2-bit constants are preserved exactly (conditional-mean residual 1.12e-5); the 4-bit fixed-point residual is 6.05e-8. Empirical NMSE in comparison.csv uses the same deterministic post-Hadamard Gaussian fixture for both FP32 codecs. Finite normalized inputs are required; normalization/scales and nonfinite handling belong to the caller.

## Measured codec spans and counts

Rows = 8 below. Ticks/vector = the final VF first-RV-instruction dispatch to last-RV-instruction completion span divided by eight; it includes in-register table construction once per VF, quantization, reconstruction and packing. It is an amortized average, not an isolated per-row latency. Rows = 1 and the raw spans are in comparison.csv. Each span excludes arena initialization, GM DMA, and (for UB variants) header initialization. Vector instructions include all RV engines; RVECEX is reported separately.

| Bits | Elements/vector | Codec | Ticks/vector | Vector instructions (8 rows) | RVECEX | Loads | Stores | VSELR | Empirical NMSE |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 128 | LM immediate | 56.750 | 404 | 357 | 16 | 24 | 96 | 0.1098487843 |
| 2 | 128 | LM UB header (warm) | 56.500 | 392 | 344 | 17 | 24 | 96 | 0.1098487843 |
| 2 | 128 | Gaussian sigmoid adapter | 72.500 | 464 | 417 | 16 | 24 | 96 | 0.1408690858 |
| 2 | 256 | LM immediate | 105.000 | 772 | 685 | 32 | 48 | 192 | 0.1186626429 |
| 2 | 256 | LM UB header (warm) | 104.500 | 760 | 672 | 33 | 48 | 192 | 0.1186626429 |
| 2 | 256 | Gaussian sigmoid adapter | 131.000 | 896 | 809 | 32 | 48 | 192 | 0.1536168623 |
| 4 | 128 | LM immediate | 86.500 | 640 | 539 | 16 | 32 | 129 | 0.0086720408 |
| 4 | 128 | LM UB header (warm) | 80.000 | 518 | 461 | 18 | 32 | 129 | 0.0086720408 |
| 4 | 128 | Gaussian sigmoid adapter | 64.250 | 477 | 395 | 16 | 32 | 80 | 0.0225521750 |
| 4 | 256 | LM immediate | 156.500 | 1136 | 987 | 32 | 64 | 257 | 0.0093648278 |
| 4 | 256 | LM UB header (warm) | 148.000 | 1014 | 909 | 34 | 64 | 257 | 0.0093648278 |
| 4 | 256 | Gaussian sigmoid adapter | 112.750 | 861 | 731 | 32 | 64 | 160 | 0.0256635605 |

Measured comparisons against the matched FP32 sigmoid adapter (eight rows):

- 2-bit / 128 elements: immediate LM has 12.9% fewer vector instructions and 1.278x the codec throughput; warm UB LM has 1.283x the throughput. Ratios below 1 mean LM is slower.
- 2-bit / 256 elements: immediate LM has 13.8% fewer vector instructions and 1.248x the codec throughput; warm UB LM has 1.254x the throughput. Ratios below 1 mean LM is slower.
- 4-bit / 128 elements: immediate LM has 34.2% more vector instructions and 0.743x the codec throughput; warm UB LM has 0.803x the throughput. Ratios below 1 mean LM is slower.
- 4-bit / 256 elements: immediate LM has 31.9% more vector instructions and 0.720x the codec throughput; warm UB LM has 0.762x the throughput. Ratios below 1 mean LM is slower.

Scheme B reserves a 512 B UB header. Centroids occupy 32/64 B rounded blocks and thresholds start at byte 96, providing 32 B bank skew after the largest centroid block. Every base is 32 B aligned. Each source has 256 addressable bytes behind it; padded reads may overlap unused lanes, while masked writes stay within their live blocks. The arena also has a 256 B tail read pad. Header initialization is vector Duplicate/Select/Store only: no GM table, DMA or scalar UB store. The header is initialized once per kernel launch and can be reused by multiple codec calls.

| Bits | Header init vector instructions | Header init span (ticks) |
| ---: | ---: | ---: |
| 2 | 29 | 31 |
| 4 | 122 | 69 |

Initialization must be charged for a cold one-shot call: the warm UB codec span alone is not a cold-launch speedup. Immediate tables avoid UB header storage; their setup is inside the measured VF. Four-bit threshold search can have more ALU instructions than the sigmoid adapter, so instruction reduction is not universal. Packing uses a three/four-stage register reduction tree and one compression Gather per register. Tier 1 replays it against an independent byte packer. The retained initial_packer_summary.json, when present, contains the earlier per-code Gather candidate measurements.

## Existing ICDF baseline and comparison limits

The repository baseline routines/routine_vector_icdf_quantize.hpp is called unchanged in mode 3. It computes FP16 14*sigmoid(x)-7 and packs signed int4 (15 reachable levels), with BF16 support in its own existing runner. It supplies neither a 2-bit mode nor a Gaussian reconstruction decoder. Therefore a literal 2-bit original-baseline comparison and a Gaussian decode NMSE for that header are unavailable. They are N/A, never zero in comparison.csv. Mode 3 uses the existing boundary-margin fixture and validates every packed byte.

The matched FP32 adapter (mode 2) explicitly supplies the missing 2/4-bit Gaussian comparator: floor(2^b * sigmoid(1.5976*x + 0.070565*x^3)), clipped to 0..2^b-1. Reconstruction uses exact FP32 normal-midquantile constants and the same Gather/packing path to isolate quantizer arithmetic. This is a new adapter, not a claim that the original header implements a Gaussian CDF, rational Chebyshev ICDF, or Newton decoder. The ideal Gaussian-CDF row is a host oracle only; no CAModel claim is made for it.

| Existing header, encode only | Rows | Codec span ticks | Ticks/vector | Vector instructions | Loads | Stores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 FP16 elements | 1 | 76 | 76.000 | 16 | 1 | 1 |
| 128 FP16 elements | 8 | 132 | 16.500 | 72 | 8 | 8 |
| 256 FP16 elements | 1 | 84 | 84.000 | 24 | 2 | 2 |
| 256 FP16 elements | 8 | 197 | 24.625 | 136 | 16 | 16 |

These encode-only timings have a different dtype, output contract and fixture, so dividing a full FP32 codec span by them is not a meaningful speedup.

## Standalone packed decoding

Modes 4/5 decode low-bits-first unsigned code sequences containing every codebook entry into FP32, with 0 ULP error. NMSE is N/A because decoder-only inputs are code patterns rather than encoded Gaussian samples. The repeating decoder fixture does not establish independent multi-page addressing; production integration requires distinguishable page probes. Each 128-code iteration reads one padded register; A3 requires 256 accessible bytes behind each packed input base.

| Bits | Elements/vector | Table | Ticks/vector (8 rows) | Vector instructions | Loads | Stores | VSELR |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 128 | LM packed decode immediate | 20.750 | 115 | 8 | 16 | 32 |
| 2 | 128 | LM packed decode UB (warm) | 22.500 | 103 | 9 | 16 | 32 |
| 2 | 256 | LM packed decode immediate | 36.750 | 203 | 16 | 32 | 64 |
| 2 | 256 | LM packed decode UB (warm) | 38.500 | 191 | 17 | 32 | 64 |
| 4 | 128 | LM packed decode immediate | 23.000 | 163 | 8 | 16 | 32 |
| 4 | 128 | LM packed decode UB (warm) | 21.250 | 103 | 9 | 16 | 32 |
| 4 | 256 | LM packed decode immediate | 39.000 | 251 | 16 | 32 | 64 |
| 4 | 256 | LM packed decode UB (warm) | 37.250 | 191 | 17 | 32 | 64 |

## Reproduction and evidence

From WSL at the repository root:

```sh
bash scripts/run_lloydmax_camodel.sh
python3 scripts/test_analyze_lloydmax_trace.py
```

The script builds a fresh ext4 source snapshot, runs Tier 1 and all 52 CAModel cases, then emits this report and comparison.csv. Raw popped/completion traces, numeric run.log, per-case histograms and audit.json are retained under this directory. build_dir.txt names the ext4 binary directory; configure.log and build.log retain build diagnostics. run-only uses LLOYDMAX_BUILD_DIR for the existing binary; LLOYDMAX_REUSE_RESULTS=1 reaudits cached cases from that build.

CMake requests Ascend950PR_9599 and its simulator path resolves to dav_3510. The installed driver stub reports Ascend950PR_9589 in run.log; this is recorded as a runtime identity discrepancy, not hidden. Compiler details and source hashes are in environment.txt and sources.sha256. This report is CAModel evidence only.
