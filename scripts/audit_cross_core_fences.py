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

Both synchronization modes are audited. Mode 4 lowers to `set_intra_block`
and mode 2 to `ffts_cross_core_sync`, and BOTH issue on the scalar stream with
the `pipe` argument ordering neither, so the drain is the same precondition for
both. Auditing only mode 4 is how a mode-2 kernel earns a vacuous pass.

Two kinds of set are exempt, and both are checked rather than assumed:

  * a *priming* set, whose meaning is "this buffer starts empty". There is no
    earlier pipe work for it to report, so there is nothing to drain
    (section 6.2a). Recognised STRUCTURALLY: nothing between the start of the
    enclosing block and the set issues work on the pipe being announced, so
    there is demonstrably no earlier work on that pipe for the fence to order.
    The older `Prime`-in-the-name route is kept as a second, narrower test.
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
    "kernels/ops/turboquant/turboquant_paged_decode_attention.cpp",
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
    r"CrossCoreSetFlag<\s*(?P<mode>[0-9]+)\s*,\s*(?P<pipe>[A-Za-z_]\w*)\s*>"
    r"\s*\(\s*(?P<flag>[^;]*?)\s*\)\s*;")
# `constexpr pipe_t kSync2AicAnnounce = PIPE_FIX;` -- a named pipe, which is how
# a kernel can assert at compile time that each participant names a pipe its own
# core may legally use. The audit resolves the alias rather than failing to
# match: an announcement it cannot see is an announcement it cannot fail.
PIPE_ALIAS_RE = re.compile(
    r"\bconstexpr\s+pipe_t\s+(?P<name>\w+)\s*=\s*(?P<pipe>PIPE_\w+)\s*;")
# A cross-core WAIT is not audited for a fence -- `wait_intra_block` and
# `wait_flag_dev` both execute on the scalar stream, so everything issued after
# one is behind it by construction (section 6.2a). It is matched only so that a
# wait between a fence and an announcement is not mistaken for pipe work.
WAIT_RE = re.compile(
    r"CrossCoreWaitFlag<\s*[0-9]+\s*,\s*PIPE_\w+\s*>\s*\([^;]*\)\s*;")
FENCE_SET_RE = re.compile(
    r"\bSetFlag<\s*(?:AscendC::)?HardEvent::(?P<route>\w+)\s*>\s*\(\s*(?P<evt>[^;]*?)\s*\)\s*;")
FENCE_WAIT_RE = re.compile(
    r"\bWaitFlag<\s*(?:AscendC::)?HardEvent::(?P<route>\w+)\s*>\s*\(\s*(?P<evt>[^;]*?)\s*\)\s*;")
PRIMING_RE = re.compile(r"Prime")

# What counts as issuing work on a pipe, for the structural priming test. A
# DataCopy, a Mmad, a Fixpipe, a Duplicate, a leaf routine call -- anything that
# is not a flag, a wait, a declaration or control flow. Rather than enumerate
# every operator, enumerate what is TRANSPARENT and treat the rest as work:
# that way an unlisted operator makes the audit stricter, never weaker.
NON_WORK_RE = re.compile(
    r"^(?:"
    r"(?:AscendC::)?(?:Set|Wait)Flag<|"
    r"(?:AscendC::)?CrossCore(?:Set|Wait)Flag<|"
    r"(?:const\s+)?(?:std::)?(?:u?int(?:8|16|32|64)_t|uint\d+|auto|bool|float)\b|"
    r"(?:AscendC::)?(?:Local|Global)Tensor\b|"
    r"(?:ascend_micro::)?Native\w+Engine\b|"
    r"\w[\w:<>]*\s+\w+\s*(?:=|;|\{)|"       # a declaration
    r"\w[\w.]*\.\w+\s*=|"                     # a field assignment
    r"(?:if|else|for|while|return|constexpr|static_assert|\}|\{)\b"
    r")")


def issues_work(code):
    """Does this code line issue work on some pipe?"""
    return not NON_WORK_RE.match(code)


def block_start(lines, index):
    """Index of the first code line of the innermost block containing lines[index].

    Brace counting backwards: the priming test asks whether anything in THIS
    block ran on the announced pipe before the set, and a set at the top of a
    loop body or a function body is the shape a priming set actually takes.
    """
    depth = 0
    for back in range(index - 1, -1, -1):
        code = lines[back][1]
        depth += code.count("}") - code.count("{")
        if depth < 0:
            return back + 1
    return 0

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
    aliases = {m.group("name"): m.group("pipe")
               for m in PIPE_ALIAS_RE.finditer(text)}
    findings = []
    for index, (number, code) in enumerate(lines):
        match = SET_RE.search(code)
        if not match:
            continue
        named = match.group("pipe")
        pipe = aliases.get(named, named)
        flag = match.group("flag")
        mode = match.group("mode")
        shown = named if named == pipe else "%s=%s" % (named, pipe)
        want = PIPE_TO_ROUTE.get(pipe)
        if PRIMING_RE.search(flag):
            findings.append((number, shown, flag, "exempt",
                             "priming set, by name: no earlier pipe work"))
            continue
        if not pipe.startswith("PIPE_"):
            findings.append((number, shown, flag, "FAIL",
                             "pipe %s is neither a PIPE_x literal nor a "
                             "`constexpr pipe_t` declared in this file" % named))
            continue
        fence = preceding_fence(lines, index)
        if fence is None:
            # Allow a run of sets that share one fence: the fence in front of
            # the FIRST set of the run still orders the whole run, because a
            # cross-core set is itself a scalar instruction and the scalar
            # stream is in order.
            back = index - 1
            shared = None
            while back > 0 and SET_RE.search(lines[back][1]):
                shared = preceding_fence(lines, back)
                if shared is not None:
                    break
                back -= 1
            fence = shared
        if fence is None:
            # No fence was written at all. That is correct exactly when the set
            # is a PRIMING set -- "this buffer starts empty" -- and the
            # structural test for that is that nothing between the start of the
            # enclosing block and the set issues work on any pipe, so there is
            # certainly none on this one. Only cross-core sets and waits, which
            # are themselves scalar and in order, may intervene. This replaces
            # the old `Prime`-in-the-name test as the primary route; the name
            # test above stays for kernels that rely on it.
            start = block_start(lines, index)
            between = [code for _, code in lines[start:index]
                       if not SET_RE.search(code) and not WAIT_RE.search(code)]
            if not any(issues_work(code) for code in between):
                findings.append((number, shown, flag, "exempt",
                                 "priming set, structurally: no work on any "
                                 "pipe since the start of this block"))
                continue
            if want is None:
                findings.append((number, shown, flag, "FAIL",
                                 "%s has no scalar route; see section 6.2a" % pipe))
                continue
            findings.append((number, shown, flag, "FAIL",
                             "no %s drain in front of the announcement" % want))
            continue
        if want is None:
            findings.append((number, shown, flag, "FAIL",
                             "%s has no scalar route; see section 6.2a" % pipe))
            continue
        route, evt = fence
        if route != want:
            findings.append((number, shown, flag, "FAIL",
                             "drained %s but announces on %s; expected %s"
                             % (route, pipe, want)))
            continue
        findings.append((number, shown, flag, "ok",
                         "mode %s, %s(%s)" % (mode, route, evt)))
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


# A mode-2 announcement is held to the same standard as a mode-4 one.
SELF_TEST_MODE2_BAD = """
__aicore__ void bad_mode2() {
    FixpipeStrip(l0c, ub);
    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(kSync2StepRetired);
}
"""

SELF_TEST_MODE2_GOOD = """
__aicore__ void good_mode2() {
    FixpipeStrip(l0c, ub);
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtStreamDrain);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtStreamDrain);
    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(kSync2StepRetired);
}
"""

# The structural priming exemption, and the control that it is not a blanket
# amnesty: the second function does work on the announced pipe first.
SELF_TEST_STRUCTURAL_PRIME = """
__aicore__ void priming() {
    for (std::uint32_t i = 0; i < primes; ++i) {
        AscendC::CrossCoreSetFlag<2, PIPE_V>(kSync2StageReady);
    }
}
"""

SELF_TEST_STRUCTURAL_NOT_PRIME = """
__aicore__ void not_priming() {
    for (std::uint32_t i = 0; i < count; ++i) {
        RecurStrip(engine, i);
        AscendC::CrossCoreSetFlag<2, PIPE_V>(kSync2StageReady);
    }
}
"""

# A named pipe must be resolved, and an unresolvable one reported rather than
# skipped: an announcement the audit cannot see is one it cannot fail.
SELF_TEST_NAMED_PIPE_GOOD = """
constexpr pipe_t kAnnounce = PIPE_FIX;
__aicore__ void named() {
    FixpipeStrip(l0c, ub);
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtStreamDrain);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtStreamDrain);
    AscendC::CrossCoreSetFlag<2, kAnnounce>(kSync2StepRetired);
}
"""

SELF_TEST_NAMED_PIPE_WRONG = """
constexpr pipe_t kAnnounce = PIPE_FIX;
__aicore__ void named_wrong() {
    FixpipeStrip(l0c, ub);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtStreamDrain);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtStreamDrain);
    AscendC::CrossCoreSetFlag<2, kAnnounce>(kSync2StepRetired);
}
"""

SELF_TEST_UNKNOWN_PIPE = """
__aicore__ void unknown() {
    FixpipeStrip(l0c, ub);
    AscendC::CrossCoreSetFlag<2, kPipeDeclaredElsewhere>(kSync2StepRetired);
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
        ("unfenced mode-2 set", SELF_TEST_MODE2_BAD, True),
        ("correctly fenced mode-2 set", SELF_TEST_MODE2_GOOD, False),
        ("structural priming set", SELF_TEST_STRUCTURAL_PRIME, False),
        ("work on the announced pipe first", SELF_TEST_STRUCTURAL_NOT_PRIME, True),
        ("named pipe, fenced", SELF_TEST_NAMED_PIPE_GOOD, False),
        ("named pipe, wrong route", SELF_TEST_NAMED_PIPE_WRONG, True),
        ("pipe name declared elsewhere", SELF_TEST_UNKNOWN_PIPE, True),
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
    print("PASS cross-core fence audit: every mode-2 and mode-4 announcement is "
          "behind a matching scalar drain, and every priming set is accounted for")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
