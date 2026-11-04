# BF16 AIC versus fused FP8: trace-based context extrapolation

One KV head, 16 shared query rows, D=256, K=1. All values below are **CAModel microseconds**, not silicon latency.

**Boundary difference:** the BF16 trace includes raw-query rotation and exports partials; the FP8 trace accepts a rotated query and writes normalized context. The span ratio is not a measured full-operator speedup.

## Native observations

| Path | Context | MMADs | First instruction → last completion (µs) |
| --- | ---: | ---: | ---: |
| BF16 AIC | 16 | 72 | 13.116111 |
| BF16 AIC | 128 | 128 | 91.811669 |
| FP8 fused | 128 | 4 | 13.912777 |
| FP8 fused | 384 | 12 | 25.155556 |
| FP8 fused | 768 | 24 | 44.444443 |

## Affine models

`T_BF16(N) = 1.873889 + 0.702638908 × N` µs.

`T_FP8(N) = 5.866669 + 0.050231478 × N` µs.

The BF16 body costs 11.242223 µs per 16-token strip; the FP8 body costs 3.214815 µs per 64-token tile. Fixed overhead is charged once, not once per strip.

The new fit uses two complete modulo-period points (6 and 12 tiles). Its N=128 holdout residual is -1.616480 µs. The old rotation-PC fingerprint matches: True.

## Same-context spans

| Context | BF16 partial-export span | FP8 normalized-output span | Span ratio | New missing-stage budget |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 91.812 (observed) | 13.913 (observed) | 6.60× | 77.899 µs |
| 384 | 271.687 (extrapolated) | 25.156 (observed) | 10.80× | 246.532 µs |
| 768 | 541.501 (extrapolated) | 44.444 (observed) | 12.18× | 497.056 µs |
| 1024 | 721.376 (extrapolated) | 57.304 (extrapolated) | 12.59× | 664.072 µs |
| 4096 | 2879.883 (extrapolated) | 211.615 (extrapolated) | 13.61× | 2668.268 µs |
| 8192 | 5757.892 (extrapolated) | 417.363 (extrapolated) | 13.80× | 5340.529 µs |
| 16384 | 11513.910 (extrapolated) | 828.859 (extrapolated) | 13.89× | 10685.051 µs |

## End-to-end interpretation

`T_old_full = T_BF16 + C_old_normalize/output`; `T_new_full = T_FP8 + C_new_query_rotation` for a common raw-query → normalized rotated-basis context boundary.

The new path wins that boundary if `C_new_query_rotation − C_old_normalize/output` is below the budget column. Those costs are not present in both captures, so a unique full end-to-end number cannot be measured from these traces.

For an explicit **same-transform-cost proxy**, charge the new path the entire old pre-score prefix (3.337222 µs) and charge the old path zero normalization cost. This prefix includes setup/ingress as well as rotation; it is not an isolated measurement of the new prologue.

| Context | BF16, normalization free (µs) | FP8 + old-prefix proxy (µs) | Conditional ratio |
| ---: | ---: | ---: | ---: |
| 128 | 91.812 | 17.250 | 5.32× |
| 384 | 271.687 | 28.493 | 9.54× |
| 768 | 541.501 | 47.782 | 11.33× |
| 1024 | 721.376 | 60.641 | 11.90× |
| 4096 | 2879.883 | 214.952 | 13.40× |
| 8192 | 5757.892 | 420.700 | 13.69× |
| 16384 | 11513.910 | 832.196 | 13.84× |

Inverse basis/gating and host dispatch are additional costs if the target boundary requires them. Their equality between paths would be another assumption.

## Assumptions and limits

* Fixed 16 shared query rows, D=256, one KV head/request, K=1, one MIX block.
* Native span is max(ts+dur)-min(ts), across all core/engine events; busy times are never summed.
* Old fit uses N=16 and N=128; page_tokens differ (16 versus 128). Rotation MMAD PC sequence is checked.
* New fit uses full modulo periods N=384 and N=768; N=128 is a short-context holdout.
* Extrapolation assumes constant per-token work and no new memory/cache/scheduling regime.
* Full end-to-end also needs the missing boundary stages. Host/ACL launch overhead is not in the trace.
* The proxy scenario charges the new path the entire observed old pre-score prefix and charges old normalization zero; this is an assumption, not a measured end-to-end time.
* FP8 versus BF16, Cube PV versus Vector PV, and different fixtures prevent attributing ratios solely to scheduling.

## Artifacts

* `comparison.json` / `comparison.csv`: exact inputs, hashes, model coefficients and projections.
* `comparison.png` / `comparison.svg`: observed points and dashed extrapolations.
* `matched_context_128_overlay.json`: two independent real N=128 traces, each shifted to start at zero and given distinct process names. Durations are unchanged; the overlay does not imply concurrent execution.

Sources:

* N=16: `reports/decode_trace_comparison_20261103/old_bf16_n16_source/trace.json`; SHA-256 `688a83ba9eb0a117aeb30908bfc707f0c29ce93c9952feaa67e7d898b6ad9990`.
* N=128: `reports/decode_comparison_aic_128_20261103T182054Z/trace.json`; SHA-256 `3565df3e590b5ded27d412aff76ff5bf1bf29a6baffd1c910e177fcd4e023b7b`.
* N=128: `reports/decode_comparison_fused-opt_128_20261103T182054Z/trace.json`; SHA-256 `404d05988721212dbdfa8be0d54f07b98c3cf2e4f6776487790022f25b10ae6e`.
* N=384: `reports/fused_decode_opt_json_20261103T180755Z/trace.json`; SHA-256 `caecb264ed504f68f90ff1c8881a522e5c61a0471943547f01b98968be0fd22a`.
* N=768: `reports/decode_comparison_fused-opt_768_20261103T182824Z/trace.json`; SHA-256 `7a5fe77c4cf501b28eff62b3698d2f814980e2fdaca8ab46957ea125ba9952d9`.
