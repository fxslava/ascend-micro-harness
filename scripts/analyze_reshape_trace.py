#!/usr/bin/env python3
"""Audit the MIX_AIC_1_2 reshape-and-cache probe from its CAModel traces.

Usage:
  analyze_reshape_trace.py [--mode2 | --mode4-single-shot]
                           [--strips-per-block S]
                           <aiv0_popped> <aiv0_complete>
                           <aiv1_popped> <aiv1_complete>
                           <aic_popped>  <aic_complete>

The recycled two-slot channel entry point is the default, matching the kernels.
--mode4-single-shot says the run dispatched the fallback that gives every event
its own id, which changes the expected per-id census. Neither flag relaxes the
order check: that one is what caught the two stale releases in the first place,
and it is the audit the recycled policy has to keep passing. --mode4-reuse is
still accepted, and now selects the default.

The fused attention audit next door checks a three-stage ring whose Vector
cores have different roles (recurrence and CV staging). This kernel has a
four-stage ring whose two Vector cores have the *same* role on alternating
chunks and whose Cube page staging never touches the Unified Buffer, so the
flag graph and the engine census are different and reusing that script would
either pass vacuously or fail for the wrong reason.

Seven checks, each fatal:

  1. intra-core SET_FLAG / WAIT_FLAG balance per core and per
     (pipe, trigger pipe, flag id), so no HardEvent token leaks, and event ids
     6 and 7 left alone;
  2. cross-core SET_INTRA_BLOCK / WAIT_INTRA_BLOCK pairing and census per
     flag id for the policy the run dispatched, and -- the check that
     matters -- that in TICK ORDER every wait was released by an announcement
     that had already issued and no id ever carried two outstanding ones. A
     recycled mode-4 id is a token a wait consumes -- but only if the
     announcement was drained to the scalar stream first. Two revisions of
     this operator that recycled ids without that drain measured a second
     wait passing 2179 and 1032 cycles before the announcement it was
     waiting for. Set and wait counts balance under reuse, so counting alone
     passes vacuously and only the order check can see it;
  3. mode-4 only: chunk ownership -- the Cube announced CHUNKS panels, half
     to each Vector
     core, and waited one panel-free flag per chunk from each half plus the two
     priming tokens, so neither Vector core was outrun and neither was skipped;
  4. both Vector cores executed, with a per-engine instruction census, and each
     one egressed its own half of the chunks;
  5. the Cube staged every page from Global Memory into L1 without a Unified
     Buffer hop: zero Global Memory stores anywhere on the AIC and at least one
     MTE2 move per page plus the rotation table;
  6. the Cube census matches the design: CHUNKS * GROUPS * GROUPS Mmads, one
     per (output group, source group) pair, and exactly one L0C -> UB Fixpipe
     per chunk. The whole rotation is Cube work now, so a Vector core that
     still applied part of it, or a Fixpipe per channel group, would show here
     rather than only in the numbers;
  7. four-engine concurrency: MTE2, Cube, Vector and MTE3 in flight at the same
     time, which is what the four-stage schedule claims and what a serialised
     implementation would fail.

Then it reports per-chunk cadence, cycle spans, merged engine overlap, and the
Vector register-traffic split that motivated moving the rotation onto the Cube.
Ticks come from one CAModel clock in all three dumps, so the intervals are
directly comparable.
"""

import re
import sys
from collections import Counter
from pathlib import Path

LINE = re.compile(
    r"\[(\d+)\].*?\)\s+(\S+)\s+:\s+\(Binary:.*?\)\s+\(ID:\s*(\d+)\)\s+(.+)"
)
FLAG = re.compile(r"PIPE:(\w+), TRIGGER PIPE:(\w+), FLAG ID:(0x[0-9a-f]+|\d+)")
SYNC = re.compile(r"(SET|WAIT)_INTRA_BLOCK\s+PIPE:(\w+).*?sync_id:(\d+)")
# Mode 2 is a different INSTRUCTION, not a variant of mode 4: NotifyEventImpl
# lowers mode 4 to set_intra_block and every other mode to
# ffts_cross_core_sync. In the dumps that is SET_CROSS_CORE / WAIT_FLAG_DEV with
# the id in an XT operand -- packed into GetffstMsg's
# `1 + (mode << 4) + (flagId << 8)` on the set, raw on the wait. The two id
# spaces are independent on Ascend 950PR, so m4:8 and m2:8 are never pooled.
M2_SET = re.compile(r"SET_CROSS_CORE\s+PIPE:(\w+).*?XT:X\d+=(0x[0-9a-fA-F]+|\d+)")
M2_WAIT = re.compile(r"WAIT_FLAG_DEV\s+PIPE:(\w+).*?XT:X\d+=(0x[0-9a-fA-F]+|\d+)")

# The BF16 prefill instance's mode-2 channels, from the kernel.
M2_PANEL_READY = 8    # AIC -> both AIVs, one broadcast per landing generation
M2_PANEL_FREE = 9     # both AIVs -> AIC, one quorum wait per generation
M2_PAGE_DRAINED = 10  # AIC -> both AIVs, the inter-wave join

# AscendC::INTRA_NUM: a mode-4 flag id space of 16 per sub block, with ids
# 11..14 library reserved inside each half.
INTRA_NUM = 16
RESERVED_FLAG_IDS = {11, 12, 13, 14, 27, 28, 29, 30}

# turboquant_a8 geometry.
CHUNKS = 8          # four value heads then four key heads
GROUPS = 8          # 32-channel groups per head
# The Cube applies the whole rotation: one Mmad per (output group, source
# group) pair, eight of them accumulating into each output group's L0C region,
# and then one Fixpipe for the finished panel.
MMADS_PER_CHUNK = GROUPS * GROUPS
FIXPIPES_PER_CHUNK = 1
LAND_DEPTH = 2      # landing panels per Vector core
# The resident rotation blocks reach L1 once per block, as kTableBlocks separate
# ND -> Nz bursts rather than one flat copy. At C0 = 16 a row-major 32 x 32
# block is NOT its own Nz image, so routine_stage_page_zn_bf16.hpp converts one
# block at a time (reports/reshape_and_cache_a8.md on the BF16 widening). The
# INT8 instance copies the blob flat, which is one burst -- so this count is the
# one number in check 5 that differs between the two instances.
TABLE_BLOCKS = 16
STAGE_DEPTH = 4     # L1 page ring depth, so four columns of pipeline fill
PER_CORE = CHUNKS // 2

# Flag ids, from the kernel. Two policies, selected by the entry point the run
# dispatched, so the audit has to be told which one it is looking at.
#
# single-shot (--mode4-single-shot): one id per cross-core event, never
# reused. Per
# Vector core, over its PER_CORE chunks j, product-ready is Owner*16 + j and
# panel-free is Owner*16 + 4 + j for the Cube's wait and
# Owner*16 + 4 + j + LAND_DEPTH for the AIV's release; Owner*16 + 10 carries
# the inter-wave drain join.
#
# recycled (the default): a bounded bilateral two-slot ring, product-ready on
# Owner*16 + slot, panel-free on Owner*16 + LAND_DEPTH + slot, drain on
# Owner*16 + 2*LAND_DEPTH. Every id then carries PER_CORE/LAND_DEPTH
# announcements, so check 2 becomes an ORDER check instead of a count check:
# the one thing that distinguishes a working ring from the two runs that
# looked like they refuted it is whether each wait was released by its OWN
# announcement. Those two were unfenced announcements overtaking their own
# pipeline (guidelines 6.2a), not a non-consuming latch -- which is why this
# check, and not a per-id count, is the one that must keep passing.
PAGE_DRAINED_LOCAL = 4 + PER_CORE + LAND_DEPTH          # 10
PRODUCT_READY = {core * 16 + j for core in (0, 1) for j in range(PER_CORE)}
PANEL_FREE = {core * 16 + 4 + i
              for core in (0, 1) for i in range(PER_CORE + LAND_DEPTH)}
PAGE_DRAINED = {core * 16 + PAGE_DRAINED_LOCAL for core in (0, 1)}

REUSE_PRODUCT_READY = {core * 16 + s for core in (0, 1) for s in range(LAND_DEPTH)}
REUSE_PANEL_FREE = {core * 16 + LAND_DEPTH + s
                    for core in (0, 1) for s in range(LAND_DEPTH)}
REUSE_PAGE_DRAINED = {core * 16 + 2 * LAND_DEPTH for core in (0, 1)}


def flag_tables(reuse):
    """(product-ready, panel-free, page-drained, owner map) for one policy."""
    if reuse:
        ready, free, drained = (REUSE_PRODUCT_READY, REUSE_PANEL_FREE,
                                REUSE_PAGE_DRAINED)
    else:
        ready, free, drained = PRODUCT_READY, PANEL_FREE, PAGE_DRAINED
    owner = {flag: flag // 16 for flag in ready | free | drained}
    return ready, free, drained, owner


# Expected announcements per id. Under the single-shot policy every id carries
# exactly one; under the recycled policy each landing channel carries one per
# generation that maps onto it, and the drain channel still carries one.
def expected_sets(flag, reuse, ready, free, drained):
    if flag in drained:
        return 1
    if not reuse:
        return 1
    if flag in ready:
        return PER_CORE // LAND_DEPTH
    if flag in free:
        # One priming set plus one release per generation on that slot.
        return 1 + PER_CORE // LAND_DEPTH
    return 1

VECTOR_PIPES = {"RVECEX", "RVECST", "RVECLD", "RVECSU", "RVECLP", "VEC"}
CUBE_PIPES = {"CUBE", "FIXP", "MTE1", "MTE2"}


def read(path):
    """Return {instruction id: (tick, pipe, operation)}."""
    records = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        match = LINE.search(line)
        if match:
            tick, pipe, instr_id, operation = match.groups()
            records[int(instr_id)] = (int(tick), pipe, operation)
    return records


def merge(intervals):
    spans = []
    for begin, end in sorted(intervals):
        if spans and begin <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([begin, end])
    return [(a, b) for a, b in spans]


def intersect(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        start, stop = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if start < stop:
            out.append((start, stop))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def hard_event_balance(name, records):
    sets, waits = Counter(), Counter()
    for _, (_, _, operation) in records.items():
        flag = FLAG.search(operation)
        if not flag:
            continue
        key = (flag.group(1), flag.group(2), int(flag.group(3), 0))
        if "SET_FLAG" in operation:
            sets[key] += 1
        elif "WAIT_FLAG" in operation:
            waits[key] += 1
    if sets != waits:
        raise SystemExit(
            f"{name}: intra-core event leak: sets={dict(sets)} waits={dict(waits)}"
        )
    reserved = [k for k in sets if k[2] in (6, 7)]
    if reserved:
        raise SystemExit(f"{name}: event IDs 6 and 7 are reserved, saw {reserved}")
    return sets


def cross_core_events(records, rebase=0):
    """[(tick, kind, absolute flag id)] for one core, in tick order.

    Counting cannot see a stale announcement: a reused id balances its sets and
    waits however early a wait was released. Order can, which is what caught
    both stale releases, so the order is kept.
    """
    out = []
    for tick, _pipe, operation in records.values():
        sync = SYNC.search(operation)
        if sync:
            kind, _, sync_id = sync.groups()
            out.append((tick, kind, int(sync_id) + rebase))
    return sorted(out)


def mode2_events(records):
    """[(tick, kind, flag, pipe)] for every mode-2 call on one core.

    A set whose packed message does not carry mode 2 is skipped rather than
    mis-attributed: the same mnemonic serves every mode above 4.
    """
    out = []
    for tick, _pipe, operation in records.values():
        got = M2_SET.search(operation)
        if got:
            msg = int(got.group(2), 0)
            if (msg >> 4) & 0x3 == 2:
                out.append((tick, "SET", msg >> 8, got.group(1)))
            continue
        got = M2_WAIT.search(operation)
        if got:
            out.append((tick, "WAIT", int(got.group(2), 0), got.group(1)))
    return sorted(out)


def cross_core(records, rebase=0):
    """{(kind, absolute flag id): count} for one core.

    An AIV prints sync_id with the sub-block bits already resolved while the
    AIC prints the id as written, so AIV1's ids are rebased by INTRA_NUM. That
    the id space is genuinely per sub block was settled by a deadlock, not by
    reading the library: guidelines section 6.2d.
    """
    out = Counter()
    for _, (_, _, operation) in records.items():
        sync = SYNC.search(operation)
        if sync:
            kind, _, sync_id = sync.groups()
            out[(kind, int(sync_id) + rebase)] += 1
    return out


def dispatch_ticks(records, keep):
    return sorted(tick for _, (tick, pipe, operation) in records.items()
                  if keep(pipe, operation))


def cadence(name, ticks, expected=None, fill=STAGE_DEPTH):
    if expected is not None and len(ticks) != expected:
        raise SystemExit(
            f"{name}: expected {expected} per-chunk events, saw {len(ticks)}"
        )
    if len(ticks) < 2:
        print(f"  {name:<28}: {len(ticks)} event(s), no interval")
        return []
    deltas = [b - a for a, b in zip(ticks, ticks[1:])]
    steady = deltas[fill - 1:] if len(deltas) > fill else []
    print(f"  {name:<28}: {len(ticks)} events, deltas "
          + " ".join(str(d) for d in deltas))
    if steady:
        mid = sorted(steady)[len(steady) // 2]
        print(f"  {'':<28}  steady state ({len(steady)} intervals): "
              f"min {min(steady)}, median {mid}, max {max(steady)}")
    return steady


def busy(records, complete, keep):
    intervals = []
    for instr_id, (begin, pipe, operation) in records.items():
        if instr_id not in complete or not keep(pipe, operation):
            continue
        end = complete[instr_id][0]
        if end <= begin:
            end = begin + 1
        intervals.append((begin, end))
    return merge(intervals)


def main(aiv0_popped, aiv0_complete, aiv1_popped, aiv1_complete,
         aic_popped, aic_complete, policy="mode2", strips=1):
    v0_in, v0_out = read(aiv0_popped), read(aiv0_complete)
    v1_in, v1_out = read(aiv1_popped), read(aiv1_complete)
    c_in, c_out = read(aic_popped), read(aic_complete)

    # ---- 1. intra-core HardEvent balance, per core --------------------
    per_core_events = {
        "AIV0": hard_event_balance("AIV0", v0_in),
        "AIV1": hard_event_balance("AIV1", v1_in),
        "AIC": hard_event_balance("AIC", c_in),
    }

    # ---- 2. cross-core census and SHAPE -------------------------------
    # The tick ORDER is deliberately not checked here any more, and that is a
    # strengthening rather than a relaxation. A WAIT_INTRA_BLOCK /
    # WAIT_FLAG_DEV is logged at the tick it was ISSUED, not the tick it was
    # released (reports/decode_splitk_aic_page_race.md section 4), and a wait
    # that blocked is issued BEFORE the announcement that releases it by
    # definition. Comparing the two issue columns therefore reported "a stale
    # set released it" on every run this operator has ever had, including runs
    # whose output is bit-exact: measured on the single-shot policy at one
    # strip, the first wait on flag 0 tripped it. A check that no correct run
    # can pass is not a gate.
    #
    # The release tick can only be read from the scalar drain behind the wait,
    # which is what scripts/audit_mode4_tick_order.py derives, with the
    # quorum rule for mode 2 and the "stall floor means not measured" caveat
    # that goes with it. That script is a MANDATORY companion to this one, and
    # the line this one prints names the command so a reader cannot take a
    # green census here for an ordering result.
    if policy == "mode2":
        # Mode 2 does not address an individual Vector core: the AIC's counter
        # advances once BOTH AIVs have announced, and one AIC announcement
        # releases both. So the shape is fixed -- one setter and two waiters on
        # each broadcast channel, two setters and one waiter on the quorum --
        # and the shape is worth checking on its own, because a mode-2 call
        # naming a pipe its own core may not use compiles to NOTHING on arch
        # 3510, silently. A dropped call shows up here as a missing row.
        gens = CHUNKS // 2 * strips
        want = {
            ("AIC", "SET", M2_PANEL_READY): gens,
            ("AIC", "WAIT", M2_PANEL_FREE): gens + LAND_DEPTH,
            ("AIC", "SET", M2_PAGE_DRAINED): 1,
            ("AIV0", "WAIT", M2_PANEL_READY): gens,
            ("AIV0", "SET", M2_PANEL_FREE): gens + LAND_DEPTH,
            ("AIV0", "WAIT", M2_PAGE_DRAINED): 1,
            ("AIV1", "WAIT", M2_PANEL_READY): gens,
            ("AIV1", "SET", M2_PANEL_FREE): gens + LAND_DEPTH,
            ("AIV1", "WAIT", M2_PAGE_DRAINED): 1,
        }
        seen = Counter()
        pipes = {}
        for name, records in (("AIV0", v0_in), ("AIV1", v1_in), ("AIC", c_in)):
            for tick, kind, flag, pipe in mode2_events(records):
                seen[(name, kind, flag)] += 1
                pipes.setdefault((name, kind, flag), set()).add(pipe)
        for key, count in sorted(want.items()):
            if seen[key] != count:
                raise SystemExit(
                    f"mode-2 channel m2:{key[2]}: {key[0]} issued {seen[key]} "
                    f"{key[1]}s, expected {count}. A count of 0 is the "
                    f"signature of a call naming a pipe that core may not use, "
                    f"which arch 3510 drops silently"
                )
        extra = sorted(k for k in seen if k not in want)
        if extra:
            raise SystemExit(
                f"unexpected mode-2 calls in the trace: {extra}. Readiness, "
                f"release and the drain join must never share an id, and no "
                f"role may use a channel it is not a party to"
            )
        ids = {M2_PANEL_READY, M2_PANEL_FREE, M2_PAGE_DRAINED}
        print(f"cross-core discipline: mode-2 quorum and broadcast, "
              f"{len(ids)} ids {sorted(ids)} for {gens} landing generations "
              f"over {strips} strip(s) per block")
        print(f"  m2:{M2_PANEL_READY} release broadcast: AIC {gens} "
              f"announcements -> AIV0 {gens} waits + AIV1 {gens} waits")
        print(f"  m2:{M2_PANEL_FREE} readiness quorum: AIV0 and AIV1 "
              f"{gens + LAND_DEPTH} announcements each "
              f"({LAND_DEPTH} priming + {gens} per generation) -> AIC "
              f"{gens + LAND_DEPTH} waits ({gens} in the schedule + "
              f"{LAND_DEPTH} epilogue drains), so the counters end at zero and "
              f"nothing leaks to the next wave")
        print(f"  m2:{M2_PAGE_DRAINED} inter-wave join: AIC 1 -> both AIVs 1")
        print(f"  pipes: " + ", ".join(
            f"{k[0]} {k[1]} m2:{k[2]} on {'/'.join(sorted(pipes[k]))}"
            for k in sorted(want) if k in pipes))
        announced = {0: gens, 1: gens}
        released = {0: seen[("AIV0", "SET", M2_PANEL_FREE)],
                    1: seen[("AIV1", "SET", M2_PANEL_FREE)]}
        cross_sets = sum(v for k, v in seen.items() if k[1] == "SET")
        cross_waits = sum(v for k, v in seen.items() if k[1] == "WAIT")
    else:
        ready_ids, free_ids, drained_ids, flag_owner = flag_tables(False)
        per_core_flags = {0: cross_core(v0_in, 0), 1: cross_core(v1_in, INTRA_NUM)}
        aiv = per_core_flags[0] + per_core_flags[1]
        aic = cross_core(c_in, 0)
        ids = {flag for _, flag in list(aiv) + list(aic)}
        print(f"cross-core discipline: mode-4 single-shot channels, "
              f"{len(ids)} ids in the trace")
        if strips != 1:
            raise SystemExit(
                "the mode-4 fallback runs one strip per block: its eleven ids "
                "per sub block are exactly the eight-chunk schedule"
            )
        for flag in sorted(ids):
            sets = aiv[("SET", flag)] + aic[("SET", flag)]
            waits = aiv[("WAIT", flag)] + aic[("WAIT", flag)]
            if sets != 1 or waits != 1:
                raise SystemExit(
                    f"mode-4 flag {flag} carries {sets} sets and {waits} "
                    f"waits, expected 1 of each under the single-shot policy"
                )
            users = sorted(core for core in (0, 1)
                           if any(k[1] == flag for k in per_core_flags[core]))
            if users and flag_owner.get(flag) != users[0]:
                raise SystemExit(
                    f"mode-4 flag {flag} is used by AIV{users[0]}, expected "
                    f"AIV{flag_owner.get(flag)}"
                )
        for flag in sorted(ids):
            if aiv[("SET", flag)] != aic[("WAIT", flag)] or \
               aic[("SET", flag)] != aiv[("WAIT", flag)]:
                raise SystemExit(
                    f"cross-core flag {flag} unpaired: AIV set "
                    f"{aiv[('SET', flag)]} / AIC wait {aic[('WAIT', flag)]}, "
                    f"AIC set {aic[('SET', flag)]} / AIV wait "
                    f"{aiv[('WAIT', flag)]}"
                )
        unknown = sorted(ids - ready_ids - free_ids - drained_ids)
        if unknown:
            raise SystemExit(f"unexpected mode-4 flag ids in the trace: {unknown}")
        if ids & RESERVED_FLAG_IDS:
            raise SystemExit(
                f"mode-4 flag ids {sorted(ids & RESERVED_FLAG_IDS)} are "
                f"reserved by the library's own SyncAll handshake"
            )
        cross_sets = sum(v for k, v in (aiv + aic).items() if k[0] == "SET")
        cross_waits = sum(v for k, v in (aiv + aic).items() if k[0] == "WAIT")
        if cross_sets != cross_waits:
            raise SystemExit(
                f"cross-core total mismatch: {cross_sets} sets, "
                f"{cross_waits} waits"
            )

        # ---- 3. chunk ownership, both ways ----------------------------
        # The Cube announces one panel per chunk and the owning Vector core
        # hands that panel back, so each core's own ids must carry exactly
        # PER_CORE product-ready flags and LAND_DEPTH + PER_CORE panel-free
        # flags (the extra two are the priming tokens, drained by the Cube
        # epilogue).
        announced = {
            core: sum(v for k, v in aic.items()
                      if k[0] == "SET" and k[1] in ready_ids
                      and flag_owner.get(k[1]) == core)
            for core in (0, 1)
        }
        released = {
            core: sum(v for k, v in per_core_flags[core].items()
                      if k[0] == "SET" and k[1] in free_ids)
            for core in (0, 1)
        }
        for core in (0, 1):
            if announced[core] != PER_CORE:
                raise SystemExit(
                    f"the Cube announced {announced[core]} panels to "
                    f"AIV{core}, expected {PER_CORE}: chunk ownership is not "
                    f"half and half"
                )
            if released[core] != LAND_DEPTH + PER_CORE:
                raise SystemExit(
                    f"AIV{core} released {released[core]} panels, expected "
                    f"{LAND_DEPTH} primed + {PER_CORE} per chunk"
                )

    print("ORDER IS NOT CHECKED HERE. A cross-core wait is logged at its ISSUE "
          "tick, so an issue-order comparison reports an artefact on every "
          "correct run. Run the release-tick audit as well, it is part of this "
          "gate:")
    print(f"    python3 scripts/audit_mode4_tick_order.py <rundir> "
          f"--block 0 --cross")
    # ---- 4. both Vector cores executed, and both egressed -------------
    census = {}
    for name, records in (("AIV0", v0_in), ("AIV1", v1_in), ("AIC", c_in)):
        if not records:
            raise SystemExit(f"{name} executed no instructions")
        census[name] = Counter(pipe for _, (_, pipe, _) in records.items())
    stores = {
        "AIV0": sum(1 for _, (_, p, o) in v0_in.items()
                    if p == "MTE3" and "Dst:OUT" in o),
        "AIV1": sum(1 for _, (_, p, o) in v1_in.items()
                    if p == "MTE3" and "Dst:OUT" in o),
    }
    for name in ("AIV0", "AIV1"):
        if stores[name] < 2 * PER_CORE * strips:
            raise SystemExit(
                f"{name} issued {stores[name]} Global Memory stores, expected "
                f"at least {2 * PER_CORE * strips}: one packed page and one "
                f"scale slice per owned chunk, over {strips} strip(s)"
            )

    # ---- 5. the Cube staged every page without a UB hop ---------------
    aic_stores = sum(1 for _, (_, _, o) in c_in.items() if "Dst:OUT" in o)
    if aic_stores:
        raise SystemExit(
            f"the Cube wrote Global Memory {aic_stores} times: the page "
            f"staging is not a pure GM -> L1 path"
        )
    # MOV_SPR_XN on the MTE2 pipe is a special-register write that configures
    # the next transfer, not a transfer, so counting every "MOV_" overstates
    # the data movement. The page copies are the ND -> NZ converter moves, and
    # there must be exactly one per chunk: the whole point of the layout is
    # that a 16 x 256 INT8 page is one burst, not a strip schedule.
    aic_pages = sum(1 for _, (_, p, o) in c_in.items()
                    if p == "MTE2" and "MOV_OUT_TO_L1" in o)
    aic_mte2_data = sum(1 for _, (_, p, o) in c_in.items()
                        if p == "MTE2" and "MOV_" in o and "MOV_SPR" not in o)
    # The rotation table reaches L1 ONCE per block, not once per strip: that it
    # stays resident across a strip boundary is part of what the stream claims,
    # and it is why TABLE_BLOCKS is not multiplied by `strips` here.
    want_moves = CHUNKS * strips + TABLE_BLOCKS
    if aic_pages != want_moves:
        raise SystemExit(
            f"the Cube issued {aic_pages} GM -> L1 ND -> Nz bursts, expected "
            f"exactly {want_moves}: {CHUNKS * strips} pages, one per chunk over "
            f"{strips} strip(s), plus {TABLE_BLOCKS} resident rotation blocks "
            f"staged ONCE for the whole stream"
        )
    if aic_mte2_data != want_moves:
        raise SystemExit(
            f"the Cube issued {aic_mte2_data} MTE2 data moves against "
            f"{aic_pages} ND -> Nz bursts: every MTE2 move on this core must be "
            f"one of those bursts"
        )
    ub_moves = sum(1 for _, (_, _, o) in c_in.items() if "MOV_UB_TO_L1" in o)
    if ub_moves:
        raise SystemExit(
            f"{ub_moves} UB -> L1 moves on the Cube: this kernel stages pages "
            f"straight from Global Memory"
        )

    # ---- report -------------------------------------------------------
    total_events = sum(sum(c.values()) for c in per_core_events.values())
    pairs = sum(len(c) for c in per_core_events.values())
    print(f"intra-core HardEvents: {total_events} sets / {total_events} waits "
          f"across {pairs} (route, ID) pairs on 3 cores; zero residual")
    print(f"cross-core calls: {cross_sets} sets / {cross_waits} waits over "
          f"{len(ids)} ids {sorted(ids)}")
    print(f"panel accounting: the Cube made {announced[0]} announcements each "
          f"AIV0 and AIV1 consumed; the cores released {released[0]} and "
          f"{released[1]} panels ({LAND_DEPTH} primed each)")
    for name in ("AIV0", "AIV1", "AIC"):
        total = sum(census[name].values())
        detail = ", ".join(f"{p} {n}" for p, n in census[name].most_common())
        print(f"  {name}: {total} instructions ({detail})")
    print(f"egress: AIV0 {stores['AIV0']} GM stores, AIV1 {stores['AIV1']} GM "
          f"stores, AIC {aic_stores}")
    print(f"page staging: {aic_pages} GM->L1 ND->NZ page moves and "
          f"{aic_mte2_data} MTE2 data moves in all on the Cube, {ub_moves} "
          f"UB->L1 moves, so every page reached L1 from Global Memory in one "
          f"burst with no Unified Buffer hop")

    # ---- 6. the Cube census matches the design ------------------------
    cube_ops = Counter(op.split()[0] for _, (_, pipe, op) in c_in.items()
                       if pipe == "CUBE" and "_FLAG" not in op)
    mmad_ops = [op for op, n in cube_ops.items()
                if n == CHUNKS * strips * MMADS_PER_CHUNK]
    if not mmad_ops:
        raise SystemExit(
            f"no CUBE-pipe mnemonic appears exactly "
            f"{CHUNKS * strips * MMADS_PER_CHUNK} "
            f"times ({CHUNKS} chunks x {MMADS_PER_CHUNK} Mmads, one per output "
            f"group and source group pair); saw {dict(cube_ops)}"
        )
    mmad = dispatch_ticks(
        c_in, lambda p, o: p == "CUBE" and o.split()[0] in mmad_ops)
    fixpipe = dispatch_ticks(c_in, lambda p, o: "FIX_L0C_TO_DST" in o)
    if len(fixpipe) != CHUNKS * strips * FIXPIPES_PER_CHUNK:
        raise SystemExit(
            f"the Cube issued {len(fixpipe)} L0C -> UB transfers, expected "
            f"{CHUNKS * strips * FIXPIPES_PER_CHUNK}: the whole rotated "
            f"panel leaves L0C "
            f"in one Fixpipe per chunk"
        )
    print(f"rotation on the Cube: {len(mmad)} Mmads ({CHUNKS} chunks x {GROUPS} "
          f"output groups x {GROUPS} source groups) and {len(fixpipe)} L0C->UB "
          f"transfers, one per chunk, so no part of the rotation runs on a "
          f"Vector core")

    # ---- per-chunk cadence --------------------------------------------
    print("per-chunk cadence. CUBE-pipe mnemonics: " +
          ", ".join(f"{op} x{n}" for op, n in cube_ops.most_common()))
    streams = [
        ("Mmad dispatch (per chunk)", mmad[::MMADS_PER_CHUNK] if mmad else [], None),
        ("Fixpipe dispatch (per chunk)", fixpipe[::FIXPIPES_PER_CHUNK],
         CHUNKS * strips),
        ("AIV0 egress", dispatch_ticks(
            v0_in, lambda p, o: p == "MTE3" and "Dst:OUT" in o)[::2], None),
        ("AIV1 egress", dispatch_ticks(
            v1_in, lambda p, o: p == "MTE3" and "Dst:OUT" in o)[::2], None),
    ]
    paced = []
    for name, ticks, expected in streams:
        fill = STAGE_DEPTH if "chunk" in name else LAND_DEPTH
        deltas = cadence(name, ticks, expected=expected, fill=fill)
        if deltas:
            paced.append((name, sorted(deltas)[len(deltas) // 2]))
    if paced:
        slowest = max(paced, key=lambda e: e[1])
        print(f"  steady-state interval: {slowest[1]} cycles per chunk, the "
              f"slowest of the measured streams ({slowest[0]})")

    # ---- cycle spans and engine overlap -------------------------------
    classes = {
        "MTE2": merge(busy(c_in, c_out, lambda p, o: p == "MTE2" and "MOV_" in o)),
        "Cube": merge(busy(c_in, c_out,
                           lambda p, o: p in ("CUBE", "FIXP", "MTE1"))),
        "Vector": merge(busy(v0_in, v0_out, lambda p, o: p in VECTOR_PIPES)
                        + busy(v1_in, v1_out, lambda p, o: p in VECTOR_PIPES)),
        "MTE3": merge(busy(v0_in, v0_out, lambda p, o: p == "MTE3" and "MOV_" in o)
                      + busy(v1_in, v1_out,
                             lambda p, o: p == "MTE3" and "MOV_" in o)),
    }
    missing = [name for name, spans in classes.items() if not spans]
    if missing:
        raise SystemExit(f"no in-flight intervals recorded for {missing}")

    all_in = list(v0_in.items()) + list(v1_in.items()) + list(c_in.items())
    all_out = list(v0_out.items()) + list(v1_out.items()) + list(c_out.items())
    dispatch = [tick for _, (tick, _, _) in all_in]
    complete = [tick for _, (tick, _, _) in all_out]
    print(f"dispatch span: cycles {min(dispatch)}..{max(dispatch)} "
          f"({max(dispatch) - min(dispatch)} cycles, {len(all_in)} instructions)")
    print(f"completion span: cycles {min(complete)}..{max(complete)} "
          f"({max(complete) - min(complete)} cycles, {len(all_out)} instructions)")

    occupancy = {n: sum(b - a for a, b in s) for n, s in classes.items()}
    print("engine busy cycles: " +
          ", ".join(f"{n} {occupancy[n]}" for n in ("MTE2", "Cube", "Vector", "MTE3")))
    for left, right in (("Cube", "Vector"), ("Cube", "MTE2"), ("Cube", "MTE3"),
                        ("Vector", "MTE2"), ("Vector", "MTE3"), ("MTE2", "MTE3")):
        common = intersect(classes[left], classes[right])
        widest = max(common, key=lambda s: s[1] - s[0], default=None)
        detail = f"widest {widest[0]}..{widest[1]}" if widest else "none"
        print(f"  overlap {left:<6} + {right:<6}: "
              f"{sum(b - a for a, b in common):>5} cycles ({detail})")

    # ---- Vector register traffic --------------------------------------
    # The factored revision's cost was register traffic, not arithmetic:
    # msprof put RV_VLD + RV_VST at 48.6% of the Vector cycles against 24.4%
    # for the whole execute pipe. msprof's denominator is accumulated
    # issue-to-complete latency per instruction site; this is the same question
    # asked of merged busy intervals, so the ratios are comparable across
    # revisions of this kernel but not against msprof's percentages.
    print("Vector register traffic (merged busy cycles, load+store vs execute):")
    for name, records, complete in (("AIV0", v0_in, v0_out), ("AIV1", v1_in, v1_out)):
        parts = {}
        for pipe in ("RVECLD", "RVECST", "RVECEX"):
            spans = busy(records, complete, lambda p, _o, want=pipe: p == want)
            parts[pipe] = sum(b - a for a, b in spans)
        traffic = parts["RVECLD"] + parts["RVECST"]
        total = traffic + parts["RVECEX"]
        counts = Counter(p for _, (_, p, _) in records.items())
        share = f"{100.0 * traffic / total:.1f}%" if total else "n/a"
        print(f"  {name}: RVECLD {parts['RVECLD']} ({counts['RVECLD']} instr), "
              f"RVECST {parts['RVECST']} ({counts['RVECST']}), "
              f"RVECEX {parts['RVECEX']} ({counts['RVECEX']}); "
              f"load+store {share} of load+store+execute")

    # ---- 7. four-engine concurrency -----------------------------------
    four = classes["Cube"]
    for name in ("Vector", "MTE2", "MTE3"):
        four = intersect(four, classes[name])
    if not four:
        raise SystemExit(
            "no four-engine concurrency: MTE2, Cube, Vector and MTE3 are never "
            "in flight together, so the four-stage schedule is serialised"
        )
    widest = max(four, key=lambda s: s[1] - s[0])
    print(f"four-engine overlap: {sum(b - a for a, b in four)} cycles, "
          f"widest CAModel cycles {widest[0]}..{widest[1]}")

    both_vectors = intersect(
        merge(busy(v0_in, v0_out, lambda p, o: True)),
        merge(busy(v1_in, v1_out, lambda p, o: True)),
    )
    if not both_vectors:
        raise SystemExit("AIV0 and AIV1 are never in flight at the same time")
    widest = max(both_vectors, key=lambda s: s[1] - s[0])
    print(f"AIV0 + AIV1 concurrent: {sum(b - a for a, b in both_vectors)} cycles, "
          f"widest CAModel cycles {widest[0]}..{widest[1]}")


if __name__ == "__main__":
    args = sys.argv[1:]
    # Mode 2 is the BF16 instance's default, so it is this script's too; a trace
    # analysed under the wrong discipline fails its census for the wrong reason,
    # which is worse than refusing to guess. --mode4-reuse named the recycled
    # ring, which is refuted and unreachable, so it now selects mode 2 -- the
    # discipline that replaced it -- rather than silently reading the fallback.
    policy = "mode4-single-shot" if "--mode4-single-shot" in args else "mode2"
    strips = 1
    rest = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("--mode2", "--mode4-reuse", "--mode4-single-shot"):
            index += 1
            continue
        if arg == "--strips-per-block" and index + 1 < len(args):
            strips = int(args[index + 1])
            index += 2
            continue
        rest.append(arg)
        index += 1
    if len(rest) != 6 or strips < 1:
        raise SystemExit(__doc__)
    main(*rest, policy=policy, strips=strips)
