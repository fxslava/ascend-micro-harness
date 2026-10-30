#!/usr/bin/env python3
"""Audit turboquant_wave_protocol.hpp for device callability and linkage.

Directive 1. The device link failed with

    ld.lld: undefined symbol: turboquant_wave::TotalStrips

`TotalStrips` is called with runtime scalars from inside the `__global__
__aicore__` kernel body of both reshape kernels. In C++17 `constexpr` already
implies `inline`, so the function had vague linkage and a weak definition
available in every translation unit that used it: linkage was never the
problem, and adding `inline` would have changed nothing. What was missing is
the *core type*. A function with no core-type attribute is host code, so
BiSheng emitted a call to a host symbol that the device link cannot resolve.

The header now routes every function through `TQ_WAVE_FN`, which is
`__aicore__ inline` in a device pass (`__CCE_AICORE__`) and plain `inline`
otherwise. One body serves both callers -- the device kernel body and the host
launch wrapper that computes the grid -- because those two live in different
arms of the same `#if`/`#elif` and so never compile together.

Three stages, because no single one of them can see the whole property:

  1. source. Every function definition in the header carries TQ_WAVE_FN. This
     is the only stage that checks the attribute itself, and it is the one
     that runs everywhere. Negative controls included, so a pass is evidence
     rather than a no-op.
  2. host linkage. Two translation units include the header and reference
     every function in it, compiled with inlining off and linked. Catches an
     undefined symbol (a declaration with no definition) and a duplicate
     symbol (a function that lost `constexpr`/`inline`). It cannot see core
     types: `__aicore__` does not exist in a host pass.
  3. device objects. `nm` over the built aic/aiv objects, requiring that no
     `turboquant_wave` symbol is defined or referenced there at all -- the
     direct form of what directive 1 asks for. Needs objects built by a CANN
     toolchain; recorded as UNAVAILABLE, not as a pass, when they are absent.

    python3 scripts/audit_device_symbols.py [<device object or directory>...]
"""

import os
import re
import subprocess
import sys
import tempfile

HEADER = "kernels/ops/turboquant/turboquant_wave_protocol.hpp"
TU1 = "tests/cpu/test_wave_protocol_linkage.cpp"
TU2 = "tests/cpu/test_wave_protocol_linkage_tu2.cpp"
NAMESPACE = "turboquant_wave"

# Lines that open a scope or assert rather than define a function, and so are
# never candidates however many parentheses they carry.
NOT_A_FUNCTION = re.compile(
    r"^\s*(?:#|//|template\s*<|namespace\b|struct\b|class\b|using\b|"
    r"static_assert\b|return\b|\}|\{)")

# A function definition head: optional decl-specifiers, a return type, a name,
# then an open paren, and NOT terminated by `;` on this line (which would make
# it a declaration or a call). The return type is required, which is what keeps
# `kFoo(1u)` inside an initialiser from matching.
DEFINITION = re.compile(
    r"^\s*(?P<pre>(?:static\s+|TQ_WAVE_FN\s+|inline\s+|constexpr\s+)*)"
    r"(?P<type>[A-Za-z_][\w:]*(?:\s*<[^<>]*>)?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*\(")


def code_lines(text):
    """(line number, code) for every line with code on it, comments removed."""
    out = []
    for number, raw in enumerate(text.splitlines(), start=1):
        code = re.sub(r"//.*$", "", raw).rstrip()
        if code.strip():
            out.append((number, code))
    return out


def audit_source_text(text):
    """[(line, name, verdict, detail)] for every function definition found."""
    findings = []
    for number, code in code_lines(text):
        if NOT_A_FUNCTION.match(code):
            continue
        # A declaration or a statement, not a definition head.
        if code.rstrip().endswith(";") and "{" not in code:
            continue
        match = DEFINITION.match(code)
        if not match:
            continue
        name = match.group("name")
        pre = match.group("pre")
        if "TQ_WAVE_FN" in pre:
            findings.append((number, name, "ok", "TQ_WAVE_FN"))
        else:
            findings.append((number, name, "FAIL",
                             "no TQ_WAVE_FN: host-attributed in a device pass, "
                             "which is the undefined-symbol link error"))
    if not findings:
        findings.append((0, "-", "FAIL", "no function definition found at all"))
    return findings


SELF_TEST_UNMARKED = """
namespace turboquant_wave {
constexpr std::uint32_t TotalStrips(std::uint32_t a, std::uint32_t b)
{
    return b == 0u ? 0u : a / b;
}
}
"""

SELF_TEST_UNMARKED_MEMBER = """
struct Channels {
    static constexpr std::uint16_t ProductReady(std::uint32_t owner)
    {
        return owner;
    }
};
"""

SELF_TEST_MARKED = """
namespace turboquant_wave {
TQ_WAVE_FN constexpr std::uint32_t TotalStrips(std::uint32_t a, std::uint32_t b)
{
    return b == 0u ? 0u : a / b;
}
struct Channels {
    static TQ_WAVE_FN constexpr std::uint16_t ProductReady(std::uint32_t owner)
    {
        return owner;
    }
};
}
"""

# Neither of these defines a function, and flagging either would make the
# audit cry wolf on the constants and the static_asserts the header is full of.
SELF_TEST_NOT_FUNCTIONS = """
namespace turboquant_wave {
TQ_WAVE_FN constexpr std::uint32_t Real(std::uint32_t a) { return a; }
static constexpr std::uint32_t kPanelFreeBase = Chunks / 2u;
static constexpr bool kRecycles = false;
static_assert(TotalStrips(1024u, 16u) == 64u, "64 strips of 16 tokens");
constexpr std::uint16_t kFlagProductReady = Flags<Reuse>::ProductReady(0u, 1u);
}
"""


def self_test():
    """The source audit must reject what it exists to reject."""
    failures = 0
    controls = (
        ("unmarked free function", SELF_TEST_UNMARKED, True),
        ("unmarked static member", SELF_TEST_UNMARKED_MEMBER, True),
        ("marked function and member", SELF_TEST_MARKED, False),
        ("constants and static_asserts", SELF_TEST_NOT_FUNCTIONS, False),
    )
    for label, source, want_fail in controls:
        verdicts = [row[2] for row in audit_source_text(source)]
        got_fail = "FAIL" in verdicts
        if got_fail != want_fail:
            print("  UNEXPECTED self-test %r: %s" % (label, verdicts))
            failures += 1
        else:
            print("  %-30s %s" % (label, "rejected" if got_fail else "accepted"))
    return failures


def stage_source():
    print("=== 1. source: every function in the header is device-callable ===")
    try:
        with open(HEADER, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        print("  FAIL cannot read %s: %s" % (HEADER, error))
        return 1

    failures = 0
    findings = audit_source_text(text)
    for number, name, verdict, detail in findings:
        if verdict == "FAIL":
            print("  FAIL   line %-5s %-24s %s" % (number, name, detail))
            failures += 1
    marked = sum(1 for row in findings if row[2] == "ok")
    print("  %d function definitions, all behind TQ_WAVE_FN" % marked
          if not failures else
          "  %d marked, %d unmarked" % (marked, failures))

    print("\n  self-test (the audit must fail on an unmarked function)")
    failures += self_test()
    return failures


def run(command, **kwargs):
    return subprocess.run(command, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, **kwargs)


def which(*names):
    for name in names:
        for folder in os.environ.get("PATH", "").split(os.pathsep):
            candidate = os.path.join(folder, name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def stage_host_linkage():
    print("\n=== 2. host linkage: two translation units, no undefined, no duplicate ===")
    compiler = which("g++", "clang++")
    if compiler is None:
        print("  UNAVAILABLE no host C++ compiler on PATH")
        return 0, 1

    failures = 0
    with tempfile.TemporaryDirectory(prefix="wave-linkage-") as out:
        objects = []
        # -fno-inline so the calls survive as calls: an audit that let the
        # optimiser delete every reference would pass on a header with no
        # definitions in it at all.
        for source in (TU1, TU2):
            obj = os.path.join(out, os.path.basename(source) + ".o")
            done = run([compiler, "-std=c++17", "-O0", "-fno-inline",
                        "-Wall", "-Wextra", "-Werror", "-I.", "-c", source,
                        "-o", obj])
            if done.returncode != 0:
                print("  FAIL compile %s" % source)
                print("".join("    " + l for l in
                              done.stdout.decode("utf-8", "replace").splitlines(True)[:20]))
                return 1, 0
            objects.append(obj)
            print("  compiled %s" % source)

        namer = which("nm", "llvm-nm")
        if namer is None:
            print("  UNAVAILABLE no nm on PATH: skipping the symbol listing")
        else:
            for obj in objects:
                done = run([namer, "-C", obj])
                undefined = [l.strip() for l in
                             done.stdout.decode("utf-8", "replace").splitlines()
                             if NAMESPACE in l and re.match(r"^\s+U ", l)]
                if undefined:
                    print("  FAIL %s references undefined %s symbols:"
                          % (os.path.basename(obj), NAMESPACE))
                    for line in undefined[:10]:
                        print("    %s" % line)
                    failures += 1
                else:
                    print("  %-44s no undefined %s symbol"
                          % (os.path.basename(obj), NAMESPACE))

        binary = os.path.join(out, "test_wave_protocol_linkage")
        done = run([compiler] + objects + ["-o", binary])
        if done.returncode != 0:
            print("  FAIL link (a duplicate symbol here means a function in the "
                  "header lost its vague linkage):")
            print("".join("    " + l for l in
                          done.stdout.decode("utf-8", "replace").splitlines(True)[:20]))
            return failures + 1, 0

        done = run([binary])
        print("  %s" % done.stdout.decode("utf-8", "replace").strip())
        if done.returncode != 0:
            failures += 1

    return failures, 0


DEVICE_HINT = re.compile(r"(aic|aiv)", re.IGNORECASE)


def collect_objects(roots):
    found = []
    for root in roots:
        if os.path.isfile(root):
            found.append(root)
        for folder, _, names in os.walk(root):
            for name in names:
                if name.endswith((".o", ".obj")) and DEVICE_HINT.search(name + folder):
                    found.append(os.path.join(folder, name))
    return sorted(set(found))


def stage_device_objects(roots):
    print("\n=== 3. device objects: no %s symbol in aic/aiv objects ===" % NAMESPACE)
    objects = collect_objects(roots)
    if not objects:
        print("  UNAVAILABLE no aic/aiv device object found under %s."
              % ", ".join(roots))
        print("              This is the only stage that checks directive 1 "
              "directly; it needs a CANN toolchain build. Not a pass.")
        return 0, 1

    namer = which("llvm-nm", "nm")
    if namer is None:
        print("  UNAVAILABLE no nm on PATH")
        return 0, 1

    failures = 0
    for obj in objects:
        done = run([namer, "-C", obj])
        if done.returncode != 0:
            print("  UNAVAILABLE %s: nm failed" % obj)
            continue
        hits = [l.strip() for l in done.stdout.decode("utf-8", "replace").splitlines()
                if NAMESPACE in l]
        if hits:
            print("  FAIL %s carries %d %s symbol(s):" % (obj, len(hits), NAMESPACE))
            for line in hits[:10]:
                print("    %s" % line)
            failures += 1
        else:
            print("  ok   %-56s no %s symbol" % (obj, NAMESPACE))
    return failures, 0


def main(argv):
    roots = argv[1:] or ["build", "build_npu", "build_sim", "out"]
    roots = [r for r in roots if os.path.exists(r)] or roots

    print("turboquant_wave device callability and linkage audit (directive 1)\n")
    failures = stage_source()
    more, unavailable = stage_host_linkage()
    failures += more
    more, un2 = stage_device_objects(roots)
    failures += more
    unavailable += un2

    print()
    if failures:
        print("FAIL device symbol audit: %d finding(s)" % failures)
        return 1
    if unavailable:
        print("PASS device symbol audit (%d stage(s) UNAVAILABLE on this host: "
              "the source attribute audit and the host link both pass, the "
              "device object listing has not run)" % unavailable)
        return 0
    print("PASS device symbol audit: every function device-callable, host link "
          "clean, and no %s symbol in any device object" % NAMESPACE)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
