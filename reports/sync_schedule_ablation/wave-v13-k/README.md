# Candidate K measurement

Candidate K ran in one isolated CANN 9.2 CAModel launch with 4 tiles, 208
tokens, and a 150-second cap. It finished in **31.24 seconds** at **0 BF16
ULP**. The retained workspace telemetry also passed. See the
[CAModel output excerpt](camodel-result.txt).

| Four-tile native schedule | Ticks |
| --- | ---: |
| Wave-v11 Slot 02, measured | 34,935 |
| K, predicted | 34,739 |
| K, measured | 37,203 |

K regressed by **2,268 ticks (6.49%)** versus Wave-v11, which remains the
native reference. The [patch](candidate-k.patch) moves slot-1 key-scale
preparation ahead of the second score wait for the final pair. Both waits
still precede the paired VF, and value-scale preparation stays in place.

The scoped [AKA comparison](analyzer-review.json) found zero new diagnostics.
The build passed strict compiler checks and the CPU oracle. The AIC object
matched Wave-v11 byte for byte. See the [build gates](build-gates.txt),
[fence audit](fence-audit.txt), [model ticks](model-ticks.txt), and
[exit status](launch-status.txt).

The evidence comes from a local WSL archive. The scripts capture the original
workflow and assume the signed Wave-v11 Slot 02 source and build, its
evaluation manifest, the generated-wrapper helpers staged for Phase 0, and
the same CANN installation. The [run manifest](inputs.sha256) records the
exact inputs, including the original binaries and loader dependencies. The
full analyzer traces and device symbol dump remain in the local reports.
The binary slot was unavailable when packaging this report, so the manifest
could not be fully rechecked. The manifest SHA-256 is
`e5975ad11db1090070df0b060a4b44fad18e31cfa4555da2a8e303ec3cb00b2a`.

This is a harness experiment. The production SIMD checkpoint is
`7c31a112c5e2250e55dbe37c3355079f7ab76629` in `turboquant-core`, tagged
`wave-v11-production-simd`.
