#!/usr/bin/env python3
"""Audit the scalar fence in front of every mode-4 cross-core announcement.

MICROROUTINE_DESIGN_GUIDELINES.md section 6.2a: `CrossCoreSetFlag<4, PIPE_x>`
lowers to `set_intra_block`, which issues on the FLOWCTRL/scalar unit. The
`pipe` template argument does NOT order it behind the work on that pipe. An
announcement whose meaning is "the work I did on pipe x has finished" therefore
has to put pipe x into the scalar stream first:

    work_on_pipe_x();
    SetFlag<HardEvent::x_S>(evt);     # pipe x -> scalar
    WaitFlag<HardEvent::x_S>(evt);    # the scalar stream waits for pipe x
    CrossCoreSetFlag<4, PIPE_x>(flag);

Section 6.2c records what happens without it. Two CAModel runs of this operator
measured a *second* wait on a recycled mode-4 id retiring 1032 and 2179 cycles
before the announcement it was waiting for, and the diagnosis that survived
`tests/research/kernel_cluster_sync_probe.cpp` on silicon and on CAModel is that
the announcement was overtaking its own un-drained pipeline -- not that the flag
is a non-consuming latch. So this fence is the precondition the recycled
two-slot ring rests on, and the recycled ring is the production default.

A missing fence is invisible to every other gate here: set and wait counts
balance, the id census passes, the token/queue model works on the schedule
rather than the fence, and the numbers can come out bit-exact while a panel is
read one generation early. Hence a separate audit.

Two kinds of set are exempt, and both are checked rather than assumed:

  * a *priming* set, whose meaning is "this buffer starts empty". There is no
    earlier pipe work for it to report, so there is nothing to drain
    (section 6.2a). Recognised by `Prime` in the flag expression.
  * a set drained by a *sequence* of routes rather than one, which is the
    inter-wave epilogue: the last pair in the sequence must still match the
    announcement's pipe.

Run from the repository root:

    python3 scripts/audit_cross_core_fences.py [<source>...]

With no argument it audits both reshape-and-cache kernels. Exit status is 0
when every announcement is fenced, 1 otherwise; the self-test at the end proves
the audit can fail, so a clean run is evidence and not a no-op.
"""

import re
import sys

DEFAULT_SOURCES = (
    "kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp",
    "kernels/ops/turboquant/turboquant_reshape_and_cache_bf16.cpp",
)

# The scalar routes that exist. There is no MTE1_S: an MTE1 drain has to ride a
# later pipe that provably depends on it (section 6.2a).
PIPE_TO_ROUTE = {
    "PIPE_V": "V_S",
    "PIPE_MTE2": "MTE2_S",
    "PIPE_MTE3": "MTE3_S",
    "PIPE_FIX": "FIX_S",
    "PIPE_M": "M_S",
}

SET_RE = re.compile(
    r"CrossCoreSetFlag<\s*4\s*,\s*(?P<pipe>PIPE_\w+)\s*>\s*\(\s*(?P<flag>[^;]*?)\s*\)\s*;")
FENCE_SET_RE = re.compile(
    r"\bSetFlag<\s*(?:AscendC::)?HardEvent::(?P<route>\w+)\s*>\s*\(\s*(?P<evt>[^;]*?)\s*\)\s*;")
FENCE_WAIT_RE = re.compile(
    r"\bWaitFlag<\s*(?:AscendC::)?HardEvent::(?P<route>\w+)\s*>\s*\(\s*(?P<evt>[^;]*?)\s*\)\s*;")
PRIMING_RE = re.compile(r"Prime")

# Diagnostics are transparent between a fence and the announcement it orders.
# They issue no work on V, MTE2, MTE3, FIX or M, so they cannot be the work the
# fence was meant to put in front of the flag -- but they do appear there, in
# the research probe among others, and an audit that called that a missing
# fence would be reporting the print rather than the hazard.
TRANSPARENT_RE = re.compile(r"^(?:AscendC::)?(?:PRINTF|printf|ASSERT|assert|DumpTensor|DumpAccChkPoint)\b")


def code_lines(text):
    """(line number, code) for every line that carries non-diagnostic code.

    Comments are dropped rather than skipped over, so a fence mentioned in a
    comment cannot satisfy the audit and a comment between the fence and the
    announcement cannot break it. Diagnostic calls are dropped for the reason
    given at TRANSPARENT_RE.
    """
    out = []
    for number, raw in enumerate(text.splitlines(), start=1):
        code = re.sub(r"//.*$", "", raw).strip()
        if not code or TRANSPARENT_RE.match(code):
            continue
        out.append((number, code))
    return out


def preceding_fence(lines, index):
    """The route of the SET/WAIT pair immediately in front of lines[index].

    Returns (route, evt) for the innermost pair, or None when the two lines in
    front of the announcement are not a matched pair. "Immediately" is over
    code lines, so comments and blank lines are transparent and any instruction
    in between -- which could be the very work the fence was meant to order --
    is not.
    """
    if index < 2:
        return None
    wait = FENCE_WAIT_RE.search(lines[index - 1][1])
    put = FENCE_SET_RE.search(lines[index - 2][1])
    if not wait or not put:
        return None
    if wait.group("route") != put.group("route"):
        return None
    if wait.group("evt") != put.group("evt"):
        return None
    return wait.group("route"), wait.group("evt")


def audit_text(text, name="<text>"):
    """[(line, pipe, flag, verdict, detail)] for every cross-core set."""
    lines = code_lines(text)
    findings = []
    for index, (number, code) in enumerate(lines):
        match = SET_RE.search(code)
        if not match:
            continue
        pipe = match.group("pipe")
        flag = match.group("flag")
        want = PIPE_TO_ROUTE.get(pipe)
        if PRIMING_RE.search(flag):
            findings.append((number, pipe, flag, "exempt",
                             "priming set: no earlier pipe work to report"))
            continue
        if want is None:
            findings.append((number, pipe, flag, "FAIL",
                             "%s has no scalar route; see section 6.2a" % pipe))
            continue
        fence = preceding_fence(lines, index)
        if fence is None:
            # Allow a run of sets that share one fence: the fence in front of
            # the FIRST set of the run still orders the whole run, because a
            # mode-4 set is itself a scalar instruction and the scalar stream
            # is in order.
            back = index - 1
            shared = None
            while back > 0 and SET_RE.search(lines[back][1]):
                shared = preceding_fence(lines, back)
                if shared is not None:
                    break
                back -= 1
            if shared is None:
                findings.append((number, pipe, flag, "FAIL",
                                 "no %s drain in front of the announcement" % want))
                continue
            fence = shared
        route, evt = fence
        if route != want:
            findings.append((number, pipe, flag, "FAIL",
                             "drained %s but announces on %s; expected %s"
                             % (route, pipe, want)))
            continue
        findings.append((number, pipe, flag, "ok", "%s(%s)" % (route, evt)))
    if not findings:
        findings.append((0, "-", "-", "FAIL",
                         "%s contains no cross-core announcement at all" % name))
    return findings


def audit_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    return audit_text(text, path)


SELF_TEST_BAD = """
__aicore__ void bad() {
    FixpipeChunk(l0c);
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kFlagProductReady<Chunk>);
}
"""

SELF_TEST_WRONG_ROUTE = """
__aicore__ void wrong() {
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kFlagProductReady<Chunk>);
}
"""

SELF_TEST_SPLIT = """
__aicore__ void split() {
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
    FixpipeChunk(l0c);
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kFlagProductReady<Chunk>);
}
"""

SELF_TEST_GOOD = """
__aicore__ void good() {
    FixpipeChunk(l0c);
    // a comment between the fence and the announcement is transparent
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtL0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtL0);
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kFlagProductReady<Chunk>);
}
"""


SELF_TEST_PRINTF = """
__aicore__ void printing() {
    AscendC::SetFlag<AscendC::HardEvent::M_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::M_S>(kEvtDrain);
    AscendC::PRINTF("[AIC] iter %u: releasing AIV0 and AIV1\\n", i);
    AscendC::CrossCoreSetFlag<4, PIPE_M>(kFlagAicToAiv0);
    AscendC::CrossCoreSetFlag<4, PIPE_M>(kFlagAicToAiv1);
}
"""


def self_test():
    """The audit must reject what it exists to reject."""
    failures = 0
    controls = (
        ("unfenced set", SELF_TEST_BAD, True),
        ("wrong scalar route", SELF_TEST_WRONG_ROUTE, True),
        ("work between fence and set", SELF_TEST_SPLIT, True),
        ("correctly fenced set", SELF_TEST_GOOD, False),
        ("diagnostic between the two", SELF_TEST_PRINTF, False),
    )
    for label, source, want_fail in controls:
        verdicts = [row[3] for row in audit_text(source, label)]
        got_fail = "FAIL" in verdicts
        if got_fail != want_fail:
            print("  UNEXPECTED self-test %r: %s" % (label, verdicts))
            failures += 1
        else:
            print("  %-28s %s" % (label, "rejected" if got_fail else "accepted"))
    return failures


def main(argv):
    sources = argv[1:] or list(DEFAULT_SOURCES)
    print("cross-core announcement fence audit "
          "(MICROROUTINE_DESIGN_GUIDELINES.md section 6.2a)")
    failures = 0
    for path in sources:
        print("\n%s" % path)
        try:
            findings = audit_file(path)
        except OSError as error:
            print("  FAIL cannot read: %s" % error)
            failures += 1
            continue
        for number, pipe, flag, verdict, detail in findings:
            mark = {"ok": "ok    ", "exempt": "exempt", "FAIL": "FAIL  "}[verdict]
            print("  %s line %-5s %-9s %-44s %s"
                  % (mark, number, pipe, flag[:44], detail))
            if verdict == "FAIL":
                failures += 1

    print("\nself-test (the audit must fail on an unfenced announcement)")
    failures += self_test()

    print()
    if failures:
        print("FAIL cross-core fence audit: %d finding(s)" % failures)
        return 1
    print("PASS cross-core fence audit: every mode-4 announcement is behind a "
          "matching scalar drain, and every priming set is accounted for")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
