# A hardware WRITE -> READ validation target for the TurboQuant KV cache

`tests/npu/validation/test_turboquant_e2e_npu.cpp`, registered as
`test_turboquant_e2e_npu` in `cmake/NpuTargets.cmake` and added to `turboquant_npu_all`.

This is the **target and its gate design**, plus what the gates do and do not establish. It is
not a run record: the binary has not been executed here, by instruction and for want of a
device. §6 says what is missing and why.

---

## 1. What it is

One process, two real operators, one cache, nothing between the two launches but a stream
synchronise:

| | operator | library |
|---|---|---|
| write | `turboquant_reshape_and_cache_bf16`, built with `TURBOQUANT_INGRESS_NZ` | `turboquant_bf16_nz_npu` (new) |
| read | `turboquant_fused_decode` (`turboquant_mm_fused_decode_opt_kv4fp8_*`) | `turboquant_fused_decode_opt_npu` |

Before the NZ ingress instance landed (`4e28825`, `00cf08c`) a round trip was not expressible
at all: the prefill's shipping strip contract and the fused decode's NZ-tiled contract name
different bytes, with co-byte matchings that share not one pair, and
`reports/turboquant_e2e_roundtrip.md` records that finding together with the two isolated
halves that stood in for it. `tests/cpu/test_turboquant_kv_layout_bridge.cpp` check 1 is the
host gate on the NZ writer naming the same byte and the same nibble the decode reads; this
target is the hardware end of the same claim.

`turboquant_bf16_npu`, the shipping strip instance, is untouched and is still what every other
harness here links. `TURBOQUANT_INGRESS_NZ` reaches only the new library and the new
executable. The driver carries an `#error` on the define's absence, so a build that linked the
strip instance by mistake fails at compile time instead of launching, passing no gate and being
very hard to read. That control was exercised: without the define the compile fails with that
message.

## 2. The two block sizes, which are different numbers

This is the one thing about the launch that is not obvious, and getting it wrong produces a
cache that is three-quarters canary with every other byte bit-exact.

The cache geometry is **64-token physical pages** — the fused decode's `block_size`, and
`Launch::block_size` carries exactly 64. The prefill quantises in **16-token strips** (`kTokens`,
a compile-time constant of that kernel), and four strips share one page.

The prefill's `block_size` **launch scalar is 16, not 64**. Under the strip contract a strip
filled a whole page, so that kernel's trip-count arithmetic uses the one scalar for both
meanings:

```
share = tokensPerCore / blockSize;   live = (numTokens - strip) / blockSize
```

which is a count of *strips*. Passing 64 there makes `share` 1, and each grid block then writes
16 of its 64 tokens. The 64-token page reaches the prefill the only way the NZ instance needs it
to — through the slot mapping, whose entries are global slots `physical_page * 64 + row`. The
kernel's own comment says so: *"this instance needs no num_blocks and no blockSize in its
addressing beyond the page it is given."*

So the launch is `tokens_per_core = 64` (one whole physical page per grid block),
`block_size = 16` (the strip width, which is what the trip count means by it),
`grid = total tokens / 64`, and `reuse_mode4_flags = true` — the mode-2 rendezvous is the only
entry point that can stream more than one strip per block, and four strips per block is what
fills a page.

Gate A is what makes that reading *checkable* rather than asserted: it compares the whole cache
image byte for byte, so a wrong strip count shows up as pre-fill bytes where nibbles belong.

## 3. The gates

**Gate A — the write, bit-exact, 0 ULP.** Every byte of both packed planes and every bit of
every scale float against `tests/common/turboquant_nz_ingress_oracle.hpp`'s golden NZ image,
which places `turboquant_oracle.hpp`'s layout-independent INT4 codes and power-of-two scales
through the *production* address functions (`NzTiledChannelByte` / `NzTiledChannelNibble` /
`NzTiledScaleIndex`). An equality, not a tolerance: the quantiser is integer arithmetic over
integer-valued BF16 activations, which is what `test_turboquant_bf16_validation.cpp` already
relies on.

Which bytes the golden claims were *written* is derived, not assumed: the image is built twice,
flooded `0x00` and `0xFF` (scales `0.0f` and `1.0f`), and a byte that agrees between the two
builds is one a strip wrote. Every other byte must still carry the `0x7FC0BEEF` pre-fill, which
covers in one comparison the unmapped physical pages, the rows of a mapped page no strip owns,
and any over-wide MTE3 burst. The written count is additionally checked against the closed form
`strips * 16 * 4 * 128` per plane.

**Gate B — the read, against an FP64 oracle over the bytes the device actually wrote.** The
reconstruction is the one `tests/cpu/test_turboquant_e2e_reference.cpp` pins against the
fused-decode fixture's own expectation:

```
value = (signed_nibble + 0.5) * stored_scale / kGain,   score scaled by inv_sqrt_len
```

The oracle re-reads the *downloaded* cache rather than the golden image, so Gate B is a
statement about the decode alone even if Gate A were to fail.

It is a tolerance, and that is a real difference from the fused decode's own hardware ladder,
which runs at zero fp16 ULP. That ladder can: its fixture is built so every value is a small
integer and the whole accumulation is exact. This one cannot — `kGain` is 2.9832882881, so a
dequantised element is not representable in binary, and the device sums 256 such products in
FP32 where the model sums them in FP64. The budgets (4 fp16 ULP, 2 bf16 ULP, both overridable
with `--ulp-fp16` / `--ulp-bf16`) are **derived** from that arithmetic — a 256-term FP32 dot is
about `sqrt(256) * 2^-24` relative, carried through a softmax of O(1) spread, against about
`2^-11` per fp16 ULP — and are **not measured on silicon**. The run reports the achieved worst
next to the budget whatever the budget is. A hardware run that reports a worse figure is data
about the budget first and about the kernel second.

**Gate C — discrimination, measured rather than argued.** A constant query cannot tell a correct
key channel order from any permutation of it, and the NZ contract's distinguishing feature is
precisely that one byte carries channels `(c, c + 128)` of one token. So the query varies per
channel, in exact binary fractions, and Gate C recomputes the oracle twice more over the same
cache — once with the query's halves swapped (the NZ nibble pairing), once with the channel
order rotated by one — and requires each to differ from the true oracle by at least 64 fp16 ULP.
Without this, Gate B could pass vacuously on a permuted cache.

The query *magnitude* is chosen rather than hoped for: the driver computes the worst `|score|`
from the golden image before any decode launch and scales the query by a power of two so the
worst score lands in [1.5, 3]. A flat softmax, or one so peaked it is a hard argmax, would
weaken Gate B without failing it.

**Canaries**, all required intact after both launches: `0x5A5A5A5A` guard bands 64 bytes on each
side of every device allocation; `0x7FC0BEEF` over every cache byte and scale float no strip
writes; the output tail past the last live channel; the Split-K workspace (every live partial
finite at S > 1, the *whole* workspace still canary at S == 1, where the operator takes its
fused path and `NeedsReduction()` is false); and every input allocation byte-identical
afterwards — which is also how the run establishes that the decode did not write the cache it
was reading.

## 4. The ladder

4 KV heads throughout (the prefill's head count is a compile-time constant, so it is not a free
parameter of this round trip), D = 256, 64-token pages, and a fragmented block table — logical
page `p` of request `r` lives at physical `2*(r*pages+p)+1`, so the physical page index is never
the logical one and over half the physical pages stay canary for the whole run.

| # | requests | context | q heads / KV head | S | what it is for |
|---|---|---|---|---|---|
| 1 | 1 | 64 | 2 | 1 | one page, one tile: the fused path, workspace untouched |
| 2 | 1 | 256 | 2 | 1,2,4 | four pages, the split sweep |
| 3 | 2 | 128 | 2 | 1,2 | two requests, two block-table rows, one cache |
| 4 | 1 | 1024 | 2 | 1,2,4,8 | 16 pages: 16 prefill grid blocks, and at S=1 one decode worker walks 16 tiles, wrapping the three-slot ring five times |
| 5 | 1 | 128 | 16 | 1,2 | 64 query heads over 4 KV heads: full GQA width |

Both element instances of the decode read the **same** cache on every rung: the prefill is
element-independent, so the write runs once per rung and the read runs twice.

## 5. What was verified here, with no device

The host half of the driver was exercised against the same shared headers it uses, with no
device and no simulator, over all five rung geometries. Every rung:

* the written byte count equals the closed form exactly — 32768 / 131072 / 131072 / 524288 /
  65536 bytes per plane, and 512 / 2048 / 2048 / 8192 / 1024 scale floats;
* leaves unmapped physical pages to carry the pre-fill (3 of 4, 6 of 10, 6 of 10, 18 of 34,
  4 of 6);
* picks a query shift that puts the worst `|score|` inside [1.5, 3] (2.41, 2.58, 2.51, 2.98,
  1.60);
* produces an oracle output that varies across channels, with values of **both signs** on rungs
  1, 3 and 5;
* separates both Gate C controls from the true oracle by far more than the 64-ULP threshold —
  the weakest separation observed anywhere was 357 fp16 ULP, the strongest 28128.

That check found one real defect before it could reach hardware.
`fused_decode_opt_test::reference_bits` **throws** outside [1, 15], because the fused-decode
fixture's results are positive normals in that band by construction; this round trip's outputs
are weighted means of signed dequantised values and carry either sign and any magnitude. The
driver now uses the general round-half-to-even converters
(`ascend_micro::sim::float_to_half_rne` / `float_to_bfloat16_rne`) with a sign-aware ordinal for
the ULP distance. Had the first version shipped, every rung with a negative output channel would
have aborted on the host *after* a prefill and a decode launch.

### The build

Clean and strict, directly in WSL against the host CANN 9.2.0-beta.2 — no container.
`-Wall -Wextra -Werror` on the driver, `-Wall -Werror` plus the BiSheng system-header
reclassification on the kernel library. Over the whole 396-line build log:

* **0** compiler warnings (`file:line:col: warning:`) and **0** compiler errors;
* **0** linker errors, no undefined symbol;
* the binary links at `build_npu/lib/test_turboquant_e2e_npu`, beside
  `libturboquant_bf16_nz_npu.so` and `libturboquant_fused_decode_opt_npu.so`.

Three classes of warning do appear, and none is ours:

* ~43 gmake `Clock skew detected` / `modification time N s in the future` lines. The source
  tree is on `/mnt/c`, whose timestamp granularity under WSL runs slightly ahead of the build
  clock. Environmental, and present for every target in this harness.
* 2 × CANN's own `[WARNING]: Multiple kernel functions are detected. It is recommended to
  define only one kernel function per file.` The BF16 prefill source has two `__global__`
  entry points — the mode-2 rendezvous and the mode-4 single-shot fallback — so this is
  pre-existing and `turboquant_bf16_npu` emits it identically.
* 4 × CMake `Manually-specified variables were not used: ASCEND_PYTHON_EXECUTABLE`, from
  CANN's nested ExternalProjects, not from this repository's cmake.

§7 has the commands.

## 6. What is NOT established, explicitly

* **No run.** The binary has not been executed, here or anywhere. Nothing in this document is
  evidence about either kernel's behaviour on silicon. The development host has no NPU
  (`npu-smi` and `/dev/davinci*` both absent), and executing the binary was out of scope by
  instruction.
* **No CAModel baseline for the round trip.** `AGENTS.md` asks for the
  `REFINEMENT_OF_COMBAT_KERNELS.md` sequence before a pipeline refactor is tagged for physical
  hardware. This change is a new *verification target* and alters no kernel, no pipeline and no
  buffer lifetime — the one kernel source it touches is not modified, only instantiated a second
  time behind an existing build-time define that `cmake/SimTargets.cmake` already registers a
  CAModel library for. The baseline was therefore neither run nor needed for this change, and no
  CAModel, `npusim` or python model was run: the task prohibited it. What the simulator leg has
  for the round trip is still the two halves in isolation at one tile
  (`scripts/run_turboquant_e2e_roundtrip.sh`), which is a canary and not a baseline. **There is
  no CAModel twin of this target yet**, and that is the gap to close before anything here is
  cited as round-trip evidence.
* **No second prefill execution wave.** Every rung's prefill grid is at most 16 blocks, inside
  the 32-block physical wave. `test_turboquant_bf16_validation.cpp` owns that ladder and reaches
  33, 48, 64 and 128.
* **No ragged context.** Every context is a whole number of 64-token pages, because the prefill
  has no partial-page path at all — its launcher refuses the geometry.
  `test_turboquant_fused_decode_opt_npu.cpp` owns the ragged rungs.
* **No inter-head scale padding canary.** A scale slot is 8 floats and the codec reads lanes
  `[head]` and `[4 + head]`; at 4 KV heads all eight are live, so that canary is vacuous here by
  construction. The run reports this rather than skipping it.
* **No timing, no overlap or throughput claim.**
* **The score multiplier is not disambiguated.** `Launch::scale` and `Launch::inv_sqrt_len` are
  both set to the same value, exactly as `tests/cpu/test_turboquant_e2e_reference.cpp` left it.
* **The Gate B budgets are derived, not measured.** See §3.

## 7. Commands

Configure and build, directly in WSL with the host CANN 9.2.0-beta.2 — no container:

```bash
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
cmake -B build_npu -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DSOC_VERSION=Ascend950PR_9599 \
    -DASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}
cmake --build build_npu --target test_turboquant_e2e_npu -j "$(nproc)"
```

The binary lands at `build_npu/lib/test_turboquant_e2e_npu` — `CMAKE_RUNTIME_OUTPUT_DIRECTORY`
is `${CMAKE_BINARY_DIR}/lib` for this project, not `bin`. Its RPATH is
`$ORIGIN;${ASCEND_HOME_PATH}/lib64`, so it finds the two adjacent kernel libraries without a
library path.

On hardware:

```bash
build_npu/lib/test_turboquant_e2e_npu
```

`--rung N` (repeatable) runs one ladder rung, `--element fp16|bf16|both` picks the decode
instance, and `--ulp-fp16 N` / `--ulp-bf16 N` move the Gate B budgets. Report the worst achieved
ULP figures with the run: they are the first measurement of them that will exist.
