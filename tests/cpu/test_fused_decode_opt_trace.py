"""Negative controls for the completion audit and producer-fence source gate."""
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from analyze_fused_decode_opt_trace import places
from audit_cross_core_fences import audit_text


def row(op, i, route='PIPE:MTE1, TRIGGER PIPE:CUBE, FLAG ID:0,'):
    return dict(mnemonic=op, id=i, tick=i, tail=route)


class AuditTests(unittest.TestCase):
    def test_source_fences(self):
        source = (Path(__file__).resolve().parents[2] /
                  'kernels/ops/turboquant/vllm_integration_layer/op_kernel/'
                  'turboquant_fused_decode.cpp').read_text()
        source = source.replace('<kSubBlockSyncMode,', '<2,')
        source = re.sub(r'AscendC::CrossCoreSetFlag<[^>]+>\([^;]+\);',
                        lambda m: ' '.join(m[0].split()), source)
        # Expand the exact helper definition (FetchEventID, SET, WAIT) so the
        # shared lexical audit sees these producer drains, rather than silently
        # skipping symbolic mode names or treating helper calls as proof.
        source = re.sub(r'vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::(\w+)>\(\);',
            lambda m: f'AscendC::SetFlag<AscendC::HardEvent::{m[1]}>(0);\n'
                      f'AscendC::WaitFlag<AscendC::HardEvent::{m[1]}>(0);', source)
        findings = audit_text(source)
        self.assertEqual(len(findings), 7)
        self.assertFalse([f for f in findings if f[3] == 'FAIL'], findings)
        broken = source.replace('AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);', '')
        self.assertTrue(any(f[3] == 'FAIL' for f in audit_text(broken)))

    def test_balanced(self):
        events = [row('SET_FLAG', 1), row('WAIT_FLAG', 2)]
        self.assertEqual(sum(places(events, lambda r: r['tick']).values()), 1)

    def test_lost_band_event(self):
        with self.assertRaisesRegex(ValueError, 'double SET'):
            places([row('SET_FLAG', 1), row('SET_FLAG', 2), row('WAIT_FLAG', 3),
                    row('WAIT_FLAG', 4)], lambda r: r['tick'])

    def test_wait_without_token(self):
        with self.assertRaisesRegex(ValueError, 'WAIT without token'):
            places([row('WAIT_FLAG', 1), row('SET_FLAG', 2)], lambda r: r['tick'])

    def test_compiler_reserved_route(self):
        route = 'PIPE:VEC, TRIGGER PIPE:SCALAR, FLAG ID:0x7,'
        places([row('SET_FLAG', 1, route), row('WAIT_FLAG', 2, route)], lambda r: r['tick'])
        with self.assertRaisesRegex(ValueError, 'reserved'):
            places([row('SET_FLAG', 1, route.replace('SCALAR', 'MTE2'))], lambda r: r['tick'])

    # BiSheng schedules either direction of its reserved VEC/SCALAR pair
    # depending on the body length, so both are permitted -- and only those
    # two. The widening must not become "any route may carry a reserved id",
    # and must not exempt an id-7 route from the capacity-one lifetime audit.
    def test_compiler_reserved_reverse_route(self):
        route = 'PIPE:SCALAR, TRIGGER PIPE:VEC, FLAG ID:0x7,'
        places([row('SET_FLAG', 1, route), row('WAIT_FLAG', 2, route)], lambda r: r['tick'])

    def test_compiler_reserved_reverse_route_other_pipe_rejected(self):
        route = 'PIPE:SCALAR, TRIGGER PIPE:MTE2, FLAG ID:0x7,'
        with self.assertRaisesRegex(ValueError, 'reserved'):
            places([row('SET_FLAG', 1, route), row('WAIT_FLAG', 2, route)], lambda r: r['tick'])

    def test_other_reserved_id_still_rejected(self):
        route = 'PIPE:VEC, TRIGGER PIPE:SCALAR, FLAG ID:0xb,'
        with self.assertRaisesRegex(ValueError, 'reserved'):
            places([row('SET_FLAG', 1, route), row('WAIT_FLAG', 2, route)], lambda r: r['tick'])

    def test_compiler_reserved_route_leak_rejected(self):
        for route in ('PIPE:VEC, TRIGGER PIPE:SCALAR, FLAG ID:0x7,',
                      'PIPE:SCALAR, TRIGGER PIPE:VEC, FLAG ID:0x7,'):
            with self.assertRaisesRegex(ValueError, 'nonempty final marking'):
                places([row('SET_FLAG', 1, route)], lambda r: r['tick'])
            with self.assertRaisesRegex(ValueError, 'double SET'):
                places([row('SET_FLAG', 1, route), row('SET_FLAG', 2, route)],
                       lambda r: r['tick'])
            with self.assertRaisesRegex(ValueError, 'WAIT without token'):
                places([row('WAIT_FLAG', 1, route)], lambda r: r['tick'])


if __name__ == '__main__': unittest.main()
