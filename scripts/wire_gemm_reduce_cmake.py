#!/usr/bin/env python3
"""Add the TQ_GEMM_REDUCE option to cmake/SimTargets.cmake, byte-for-byte.

That file is CRLF and is not valid UTF-8 -- it carries a mojibake section sign in a comment --
so it is read and written as latin-1 to preserve every byte this edit does not touch, and the
inserted lines carry CRLF like their neighbours. Converting the file's encoding would be a
whole-file diff for a three-line change.

Idempotent: running it twice is a no-op.
"""
import pathlib
import sys

PATH = pathlib.Path(__file__).resolve().parent.parent / 'cmake' / 'SimTargets.cmake'
ANCHOR = '  ascendc_compile_options(camodel_fused_decode_opt_kernel PRIVATE\n'
SYSOPTS = '      -std=c++17 -Wall -Werror ${ASCEND_MICRO_CANN_SYSTEM_OPTIONS}\n'
DEFINES = '      ${TQ_FUSED_DECODE_REDUCE_DEFINES}\n'
OPTION = ''.join(line + '\n' for line in [
    '  # TQ_GEMM_REDUCE drives the operator\'s reduction phase with the Cube GEMM reduction',
    '  # (TurboQuantBatchedGemmReducer) instead of the shipping vector reducer. It is OFF by',
    '  # default and turning it ON is EXPECTED TO FAIL the build, on a static_assert naming the',
    '  # operator\'s 19648-byte UB deficit -- the batched form\'s two 8-row fp32 panels do not fit',
    '  # the 4256 bytes left under the 192 KB window. The arithmetic and the three ways out are in',
    '  # include/turboquant_gemm_reduce.hpp; scripts/gemm_reduce_switch_gate.sh is the negative',
    '  # control that checks the build fails for THAT reason and not another.',
    '  option(TQ_GEMM_REDUCE "Drive TurboQuantFusedDecodeOpt\'s reduction with the Cube GEMM" OFF)',
    '  set(TQ_FUSED_DECODE_REDUCE_DEFINES "")',
    '  if(TQ_GEMM_REDUCE)',
    '    set(TQ_FUSED_DECODE_REDUCE_DEFINES -DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1)',
    '  endif()',
])

# newline='' disables universal-newline translation, so the bytes this edit does not touch are
# the bytes that go back. The committed file is LF, so CRLF in the working tree is normalised
# away here too: it turns a twelve-line change into a whole-file diff, and a reviewer cannot see
# the twelve lines through it.
STALE = '''  # The Split-K final reduction as a Cube GEMV. PROTOTYPE: its kernel source and
  # its service header both live under tests/, so nothing in kernels/ can reach
  # them and the variant cannot link into the operator. It emits three entry
  # points over one workspace -- the GEMV stage, the same stage with Mmad's GEMV
  # mode left on, and the shipping TurboQuantPartialReducer as the numerical
  # reference -- so the comparison is against production arithmetic compiled in
  # the same build rather than against a host reimplementation of it.
'''
FRESH = '''  # The Split-K final reduction on the Cube. Its KERNEL source stays under tests/, so the
  # probe's entry points cannot reach a production launcher, but its SERVICE header now lives
  # in kernels/ops/turboquant/fused_decode/cube/ because the dense batched reducer is wired
  # into the operator's reduction phase behind TQ_GEMM_REDUCE above. The probe emits five entry
  # points over one workspace -- the M = 1 GEMV stage, that stage with Mmad's GEMV mode left
  # on, the dense batched form, the batched form with the same knob, and the shipping
  # TurboQuantPartialReducer as the numerical reference -- so every comparison is against
  # production arithmetic compiled in the same build rather than a host reimplementation of it.
'''

with PATH.open('r', encoding='latin-1', newline='') as source:
    text = source.read().replace('\r\n', '\n')
if 'TQ_GEMM_REDUCE' in text:
    print('already wired; nothing to do')
    sys.exit(0)
if ANCHOR not in text:
    sys.exit('anchor not found: %s' % ANCHOR.strip())

head, _, tail = text.partition(ANCHOR)
if not tail.startswith(SYSOPTS):
    sys.exit('the system-options line does not follow the anchor as expected')
tail = SYSOPTS + DEFINES + tail[len(SYSOPTS):]
# The probe's comment says the service header lives under tests/. It does not any more.
if STALE in tail:
    tail = tail.replace(STALE, FRESH)
else:
    print('note: the probe comment was not in its expected form; left as it is', file=sys.stderr)
with PATH.open('w', encoding='latin-1', newline='') as sink:
    sink.write(head + OPTION + ANCHOR + tail)
print('wired TQ_GEMM_REDUCE into %s' % PATH.name)
