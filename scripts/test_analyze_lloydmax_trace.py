#!/usr/bin/env python3
"""Negative fixtures for the benchmark's evidence gates, no CANN required."""
from pathlib import Path
import tempfile
import unittest
from analyze_lloydmax_trace import audit

def line(ident,pc,engine,op,tick=None):
    return f'[info] [{100+ident if tick is None else tick}] (PC: {pc:#x}) {engine} : (Binary: 0x0) (ID: {ident}) {op}\n'

class TraceGate(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.p=self.root/'core0.veccore0.instr_popped_log.dump'
        self.d=self.root/'core0.veccore0.instr_log.dump'
        self.rows=[line(0,0,'PUSHQ','VF addr: 0x100,instr_num: 0x8,')]
        for i,op in enumerate(['RV_VLD']+['RV_VSELR']*4+['RV_VST']*2,1):
            self.rows.append(line(i,0x100+4*(i-1),'RVECEX',op))
        self.rows.extend([line(8,0,'VEC','SET_FLAG PIPE:VEC, TRIGGER PIPE:MTE3, FLAG ID:2'),
                          line(9,0,'MTE3','WAIT_FLAG PIPE:VEC, TRIGGER PIPE:MTE3, FLAG ID:2')])
        self.save()
        (self.root/'run.log').write_text('PASS bits=2 mode=4 n=128 rows=1 boundaries=0 nmse=0 max_abs=0 max_ulp=0\n')
    def tearDown(self): self.temp.cleanup()
    def save(self): self.p.write_text(''.join(self.rows));self.d.write_text(''.join(self.rows))
    def check(self): return audit(self.root,2,4,128,1)
    def test_valid(self): self.assertEqual(self.check()['R1'],'PASS')
    def test_inactive_core_placeholders(self):
        (self.root/'core1.veccore0.instr_popped_log.dump').write_text('')
        self.assertEqual(self.check()['R1'],'PASS')
    def test_missing_evidence(self):
        self.p.unlink()
        with self.assertRaises(ValueError): self.check()
    def test_missing_completion(self):
        self.d.write_text('')
        with self.assertRaisesRegex(ValueError,'completion'): self.check()
    def test_spill(self):
        self.rows.append(line(10,0x11c,'RVECLD','RV_VLD'));self.save()
        with self.assertRaisesRegex(ValueError,'R1'): self.check()
    def test_scalar_drain(self):
        self.rows.append(line(10,0,'VEC','SET_FLAG PIPE:VEC, TRIGGER PIPE:SCALAR, FLAG ID:0'));self.save()
        with self.assertRaisesRegex(ValueError,'scalar drain'): self.check()
    def test_unbalanced(self):
        self.rows.pop();self.save()
        with self.assertRaisesRegex(ValueError,'unbalanced'): self.check()
    def test_model_error(self):
        (self.root/'core.log').write_text('[ERROR] invalid ldst addr')
        with self.assertRaisesRegex(ValueError,'CAModel error'): self.check()
    def test_immediate_load_spill(self):
        self.rows.append(line(10,0x11c,'RVECLD','RV_VLDI'));self.save()
        with self.assertRaisesRegex(ValueError,'R1'): self.check()
    def test_dynamic_allocator(self):
        self.rows.append(line(10,0,'SCALAR','SFF'));self.save()
        with self.assertRaisesRegex(ValueError,'allocator'): self.check()
    def test_missing_numeric_verdict(self):
        (self.root/'run.log').write_text('exited 0\n')
        with self.assertRaisesRegex(ValueError,'numeric'): self.check()
    def test_wrong_case(self):
        (self.root/'run.log').write_text('PASS bits=2 mode=4 n=256 rows=1 boundaries=0 nmse=0 max_abs=0 max_ulp=0\n')
        with self.assertRaisesRegex(ValueError,'case identity'): self.check()

if __name__=='__main__': unittest.main()
