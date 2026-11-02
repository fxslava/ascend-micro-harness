# TurboQuant CDF, uniform-grid and 5-bit ablation

`tq_cdf_vs_lloydmax_ablation.py` is a standalone CPU research runner. Install
`requirements-tq-cdf.txt`, then run from the repository root:

```powershell
python -m pip install -r tests/research/scripts/requirements-tq-cdf.txt
./tests/research/scripts/run_tq_cdf_vs_lloydmax_ablation.ps1
```

The PowerShell runner verifies the implementation, then benchmarks the sources
sequentially. The defaults cover 3/4/5 bits and all five methods, five paired seeds, synthetic outlier
gains 1/8/64, and NIAH contexts 8192/32768/131072. Results are written under
`reports/tq_cdf_vs_lloydmax/extended/{synthetic,niah}`. Each directory contains:

- `summary_ext.md` (also `summary.md`): metric, timing, paired-statistic tables,
  uniform-vs-CDF comparisons and 5-bit significance counts.
- `runs.csv`: one row per source/condition/seed/bit width/method.
- `needles.csv`: NIAH's individual 1-based needle and paired distractor ranks.
- `paired.csv`: differences, paired t statistics, p-values, confidence bounds,
  and degenerate-test status, stratified by condition and bit width.
- `metadata.json`: arguments, command, versions, timestamp, and runner SHA-256.

The parent `extended/summary_ext.md` combines both sources; `extended/runs.csv`,
`needles.csv` and `paired.csv` contain the complete extended grid. `audit.json`
records source metadata and verifies unchanged non-timing metrics for the prior
3/4-bit runs when those original CSVs are available. Original reports under
`reports/tq_cdf_vs_lloydmax/{synthetic,niah}` are preserved.

The corrected cross-bit study is included in `extended/cross_bit_parity.md`
and `extended/cross_bit.csv`: **uniform 5 vs reference 4 bits**, and
**uniform 4 vs reference 3 bits**. Both uniform grids are compared with all
three reference codecs, pairing the same seeds within every context/kappa.
All retained metrics have paired t statistics and 95% CIs. Additional paired
TOST tests assess explicit working equivalence bands: recall/FPR +/-1/16,
rho +/-0.001, and key/logit NMSE within a multiplicative factor of 1.10
(log ratios). TOST uses alpha .05 and 90% CIs; zero-variance results are
inconclusive. Strict zero-loss noninferiority is recorded separately.
These exploratory bands are stated for interpretation, not treated as proof
that observed recall losses are harmless. Better fidelity is distinct from
equal metrics and from loss-free retrieval.

Run either source directly or change scale granularity:

```powershell
python tests/research/scripts/tq_cdf_vs_lloydmax_ablation.py --source synthetic
python tests/research/scripts/tq_cdf_vs_lloydmax_ablation.py --source niah
python tests/research/scripts/tq_cdf_vs_lloydmax_ablation.py --source niah --group-size 32 --output-dir reports/tq_cdf_vs_lloydmax/group32
```

`--group-size 0` stores one FP16 RMS scale per vector; other values must divide
both 64 and 128. Encoder normalization uses the actual stored FP16 scale.
At 128 coordinates, per-vector effective storage is 3.125/4.125/5.125 bits per
coordinate; group-32 storage is 3.5/4.5/5.5 bits. Integer codes are actually packed
and unpacked at 3/4/5 bits. `levels_used` counts distinct reconstructed values,
while `code_levels_used` counts distinct code indices.

The rotation is the orthogonal, self-inverse `D H D` with one deterministic
Rademacher diagonal and normalized Walsh-Hadamard matrix. Methods share data,
query vectors, rotation, scale granularity and FP16 scale storage. The original
three methods use RMS scales; the new uniform methods use absmax-derived steps.
Analytical decoders evaluate inverse
functions per coordinate; they never replace those functions with a cached
reconstruction codebook. GaussianCDF uses `ndtr/ndtri`; the cubic logistic
fit uses coefficients `(1.5976, 0.070565, 0)` and three Newton updates after
logit. This is the requested Sigmoid-Vasicek regime name, without assigning
historical authorship to those particular coefficients. A published cubic
logistic approximation is described in
[this research paper](https://www.econstor.eu/bitstream/10419/188388/1/v02-i01-p114_60-313-1-PB.pdf).

`TQ-Uniform-Midrise` uses `s=absmax/(2^(b-1)-0.5)`,
`code=clip(floor(v/s+2^(b-1)),0,2^b-1)` and
`decoded=(code-2^(b-1)+0.5)*s`. `TQ-Uniform-Midtread` uses
`s=absmax/(2^(b-1)-1)`, nearest-integer rounding with ties to even, signed
clipping to `[-2^(b-1),2^(b-1)-1]`, and `decoded=q*s`. Its signed codes are
shifted by `2^(b-1)` before packing and converted to a signed/float type before
subtracting that zero point, preventing unsigned wraparound. The symmetric
absmax scale normally leaves the extra most-negative midtread code unused.
Zero groups reconstruct exactly as zero with either codec.

The 5-bit Lloyd-Max codebook solves all 32 Gaussian conditional-mean equations
to convergence in FP64 (maximum centre update below 1e-14), then stores FP32
centres and thresholds. The 3/4-bit baseline preserves its reference 256-update
values. Independent tests check all 32 codes, scalar bitstream ordering,
signed uniform decoding, boundary and zero behavior, FP16 group steps, and
stationarity of the 32-cell Gaussian codebook.

The original suite at `C:/ascend-moe-engine/tests/research` was inspected.
The named `qjl_vs_int4_ablation.py` and `tq_kv_quant_reference.py` were absent.
The standalone implementation adapts the available Gaussian Lloyd-Max update,
packing, anisotropy/adversarial construction, half-mode RoPE, and metric
definitions. An optional compatibility test imports that checkout when present
and checks codebook values, packing, tied-rank Spearman, and RoPE rotation.
The new runner needs no access to that checkout.

NIAH averages groups of four tokens before rotation/quantization. Each query
has one needle and one distractor; FPR counts those 16 paired distractors.
Their cosine band applies **before** compression. All exact ranking baselines
use the same compressed keys. Exact recall is recorded to distinguish
compression difficulty from scalar quantization loss. Random-access Gaussian
generation and fixed-order FP32 reductions make metric results invariant to
stream chunking. Memory is bounded by the chunk, sample budget, and fixed
special/query counts; it does not retain the full cache or a context-sized
ranking array.

The separate RoPE probe has 64 channels, half-mode rotation, theta 160000,
and positions in `[0,131072)`. It reports the requested full-vector angular
deviation after counterfactual quantization. It does not change NIAH logits.
`mean_rope_logit_share` matches the existing DSV4 convention
`|rotary|/(|rotary|+|semantic|)`; it is an absolute-contribution share rather
than a squared-energy statistic. Synthetic Gaussian mode reports projection
fidelity and correlation; retrieval-only fields are empty. For continuity,
`nmse` retains key-vector reconstruction NMSE. The new `logit_nmse` is
`sum((QK^T-QKhat^T)^2)/sum((QK^T)^2)` across all streamed slots and 16 queries;
`logit_relative_rmse` is its square root. Reports distinguish NMSE(key) from
NMSE(QK) explicitly. Paired tests cover both metrics.

Seeds, rather than the 16 needles, are independent observations in paired tests.
P-values are exploratory and unadjusted. Default noninferiority margins are
zero for recall and 0.001 for rho. Zero-variance recall differences are
reported explicitly and cannot prove population equivalence. With zero recall
margin, identical finite-sample recall is descriptive evidence only. Strict
observed degradation is reported separately from the configurable margins.

`codec_us_per_slot` times scale estimation, forward companding, bit packing,
unpacking, and reconstruction. `stream_latency_s` adds data generation,
rotation, exact and quantized scoring, rank counters, and sample capture.
Initialization, warmup, final Spearman/RoPE, and diagnostic reductions are
excluded. Each chunk shuffles method order; BLAS uses one thread by default.
These NumPy/SciPy measurements do not establish performance on Ascend.

This change adds CPU research scripts and reports only. The production-kernel
pipeline protocol in `REFINEMENT_OF_COMBAT_KERNELS.md` is not applicable:
no operator pipeline or buffer lifetime is changed. CAModel and physical-NPU
gates were not executed, and the report makes no claims about them.
