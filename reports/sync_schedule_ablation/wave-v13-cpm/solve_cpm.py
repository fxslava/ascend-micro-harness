"""Trace-calibrated local Max-Plus schedule search; never launches a kernel.

Run under WSL: python3 /mnt/d/Projects/ascend-micro-harness/reports/
sync_schedule_ablation/wave-v13-cpm/solve_cpm.py
This is a conditional counterfactual model, not a silicon timing oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from graphlib import TopologicalSorter, CycleError
from pathlib import Path
import hashlib
import heapq
import json
import sys

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
SCRIPTS = REPORTS.parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from analyze_single_token_decode_trace import parse

TRACE_ROOT = Path('/home/ladislav/ascend-sync-schedule-ablation/agent_02')
RUNS = [TRACE_ROOT / 'wave-v11/run/t4.u97cm4',
        TRACE_ROOT / 'wave-v12/run/t4.augkqw']
COMPARISON = REPORTS / 'wave-v12-slot02-vs-v11.json'
UB_LIMIT = 216 * 1024
UB_BYTES = 192384
MTE2_STARTUP = 200


@dataclass
class Edge:
    source: str
    lag: int
    kind: str


class Graph:
    def __init__(self):
        self.edges: dict[str, list[Edge]] = {'origin': []}
        self.labels: dict[str, str] = {}

    def add(self, name, *edges, label=''):
        assert name not in self.edges
        self.edges[name] = [Edge(*edge) for edge in edges]
        assert all(edge.lag >= 0 for edge in self.edges[name])
        self.labels[name] = label
        return name

    def fixed(self, name, tick, label='retained measured boundary'):
        return self.add(name, ('origin', tick, 'retained upstream work'), label=label)

    def solve(self):
        order = list(TopologicalSorter({n: [e.source for e in es]
                                       for n, es in self.edges.items()}).static_order())
        time, critical = {}, {}
        for node in order:
            arrivals = [(time[e.source] + e.lag, e.source) for e in self.edges[node]]
            time[node], critical[node] = max(arrivals, default=(0, None))
        return time, critical

    def independent_check(self):
        """Event-driven readiness queue, independent of longest-path traversal."""
        pending = {n: len(es) for n, es in self.edges.items()}
        out = {n: [] for n in self.edges}
        for node, edges in self.edges.items():
            for edge in edges:
                out[edge.source].append((node, edge.lag))
        ready = [(0, n) for n, count in pending.items() if count == 0]
        heapq.heapify(ready)
        time, arrival = {}, dict.fromkeys(self.edges, 0)
        while ready:
            tick, node = heapq.heappop(ready)
            time[node] = tick
            for child, lag in out[node]:
                arrival[child] = max(arrival[child], tick + lag)
                pending[child] -= 1
                if pending[child] == 0:
                    heapq.heappush(ready, (arrival[child], child))
        assert len(time) == len(self.edges), 'cycle or missing producer'
        return time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(case, run):
    roles = []
    for role, aiv in enumerate(case['aivs']):
        pop_path = run / f'core0.veccore{role}.instr_popped_log.dump'
        retire_path = run / f'core0.veccore{role}.instr_log.dump'
        pops, retired = parse(pop_path), parse(retire_path)
        wait = [w for w in aiv['waits'] if w[0] == 7][-1]
        first_receive = aiv['score_receive'][2]
        vf_start = aiv['vf'][1][0]
        previous_end = max(r['tick'] for r in retired
                           if first_receive <= r['tick'] < wait[2]
                           and r['engine'].startswith('RV'))
        first_vector = min(r['tick'] for r in pops
                           if wait[2] <= r['tick'] < vf_start
                           and r['engine'].startswith('RV'))
        # PairKeyScale<1> uses UB 0x5fe0 in both signed native traces.
        key_writes = [r for r in retired
                      if wait[2] <= r['tick'] < vf_start
                      and r['mnemonic'] == 'RV_VST' and '=0x5fe0,' in r['tail']]
        assert len(key_writes) == 2, key_writes
        key_done = max(r['tick'] for r in key_writes)
        roles.append(dict(role=role, wait_start=wait[1], score_ready=wait[2],
                          previous_vector_end=previous_end,
                          first_vector=first_vector, key_done=key_done,
                          scalar_prefix=first_vector-wait[2],
                          key_vector=key_done-first_vector,
                          value_and_vf_admission=vf_start-key_done,
                          total_prefix=vf_start-wait[2],
                          key_write_evidence=key_writes,
                          popped_sha256=digest(pop_path), retired_sha256=digest(retire_path)))
    return roles


def build(case, weights, variant='parent', stress=0, end_stress=0):
    g = Graph()
    # All upstream scalar/cold, DMA, dequant, first pair and ring work is retained.
    # This search changes only the final pair; it cannot spend upstream gaps.
    qk3 = g.fixed('AIC.QK3.post', case['products']['QK3']['post'])
    for role, (aiv, w) in enumerate(zip(case['aivs'], weights)):
        p = f'AIV{role}'
        g.fixed(f'{p}.score0.received', aiv['score_receive'][2])
        g.fixed(f'{p}.key0.done', w['previous_vector_end'])
        g.fixed(f'{p}.prefix.admission', w['wait_start'])
        g.add(f'{p}.score1.ready', (qk3, w['score_ready']-case['products']['QK3']['post'], 'score data'))
        if variant == 'parent':
            start = f'{p}.score1.ready'
        elif variant == 'serialized_prefix':
            start = g.add(f'{p}.prefix.start',
                          (f'{p}.prefix.admission', 0, 'source issue'),
                          (f'{p}.key0.done', 0, 'conservative whole-prefix serialization'))
        else:
            start = f'{p}.prefix.admission'
        g.add(f'{p}.scale.scalar.end', (start, w['scalar_prefix'], 'retained descriptor cost'))
        g.add(f'{p}.scale.vector.start',
              (f'{p}.scale.scalar.end', 0, 'descriptor ready'),
              (f'{p}.key0.done', 0, 'single AIV Vector queue'))
        g.add(f'{p}.key1.done', (f'{p}.scale.vector.start', w['key_vector'], 'key scale VF service'))
        if variant in ('key1_hoist', 'serialized_prefix'):
            value_start = g.add(f'{p}.value.start',
                                (f'{p}.key1.done', 0, 'Vector order'),
                                (f'{p}.score1.ready', 0, 'retained score wait'))
        else:
            value_start = f'{p}.key1.done'
        g.add(f'{p}.prefix.end', (value_start, w['value_and_vf_admission'], 'retained value preparation/admission'))
        g.add(f'{p}.pairedVF.start',
              (f'{p}.prefix.end', stress, 'retained/added pre-VF residual'),
              (f'{p}.score1.ready', 0, 'score1 data'),
              (f'{p}.score0.received', 0, 'score0 data'))
        g.add(f'{p}.pairedVF.end',
              (f'{p}.pairedVF.start', aiv['vf'][1][1]-aiv['vf'][1][0], 'paired VF unchanged'))
        g.add(f'{p}.P2.post', (f'{p}.pairedVF.end', aiv['p'][2]-aiv['vf'][1][1], 'P2 copy/fence'))
    p2_ready = max(a['p'][2] for a in case['aivs'])
    pv2 = case['products']['PV2']['post']
    g.add('AIC.PV2.post', ('AIV0.P2.post', pv2-p2_ready, 'quorum + unchanged GEMM/Fixpipe'),
          ('AIV1.P2.post', pv2-p2_ready, 'quorum + unchanged GEMM/Fixpipe'),
          (qk3, 1, 'single AIC product issue queue'))
    for role, aiv in enumerate(case['aivs']):
        p = f'AIV{role}'
        # Replay P3 is after receiving/accumulating PV2 in source order.
        g.add(f'{p}.P3.post', ('AIC.PV2.post', aiv['p'][3]-pv2, 'ordered recurrence and P3 publication'))
    p3_ready = max(a['p'][3] for a in case['aivs'])
    pv3 = case['products']['PV3']['post']
    g.add('AIC.PV3.post', ('AIV0.P3.post', pv3-p3_ready, 'quorum + unchanged GEMM/Fixpipe'),
          ('AIV1.P3.post', pv3-p3_ready, 'quorum + unchanged GEMM/Fixpipe'),
          ('AIC.PV2.post', 1, 'single AIC product issue queue'))
    for role, aiv in enumerate(case['aivs']):
        p = f'AIV{role}'
        for i, (begin, end) in enumerate(aiv['output']):
            g.add(f'{p}.output{i}.start', ('AIC.PV3.post', begin-pv3, 'retained normalize/output scalar work'))
            g.add(f'{p}.output{i}.end', (f'{p}.output{i}.start', end-begin, 'measured shared-HBM envelope'))
        last = max(b for a, b in aiv['output'])
        g.add(f'{p}.END', *[(f'{p}.output{i}.end', aiv['last']-last, 'role terminal work')
                           for i in range(len(aiv['output']))])
    last_output = max(b for aiv in case['aivs'] for a,b in aiv['output'])
    joins = [w for w in case['aic_waits'] if w[0] == 10]
    g.add('AIC.join1', *[(f'AIV{r}.output{i}.end', joins[0][2]-last_output, 'task output join')
                        for r in range(2) for i in range(len(case['aivs'][r]['output']))])
    g.add('AIC.join2', ('AIC.join1', joins[1][2]-joins[0][2], 'retained task join/scalar tail'))
    g.add('AIC.branch', ('AIC.join2', case['aic_final_jump']['tick']-joins[1][2], 'retained scalar tail'))
    g.add('AIC.END', ('AIC.branch', case['aic_end']['tick']-case['aic_final_jump']['tick']+end_stress,
                      'measured unexplained branch-to-END gap'))
    last_core = max(case['aic_end']['tick'], *(a['last'] for a in case['aivs']))
    g.add('STOP', *[(n, case['ticks']-last_core, 'model stop latency')
                   for n in ('AIC.END', 'AIV0.END', 'AIV1.END')])
    time, critical = g.solve()
    assert time == g.independent_check()
    return g, time, critical


def ring_check(tiles, illegal=False):
    """Finite two-slot ownership graph; not an exhaustive full-kernel verifier."""
    deps = {}
    for t in range(tiles):
        deps[f'packed{t}'] = [] if t < 2 else [f'stage{t-2}']
        deps[f'scale{t}'] = [] if t < 2 else [f'pair{(t-2)//2}']
        deps[f'stage{t}'] = [f'packed{t}', f'scale{t}'] + ([] if t < 2 else [f'pv{t-2}'])
        deps[f'qk{t}'] = [f'stage{t}']
        deps[f'pv{t}'] = [f'pair{t//2}']
        if illegal and t >= 2:
            deps[f'packed{t}'].append(f'pv{t-2}')
    for first in range(0, tiles, 2):
        deps[f'pair{first//2}'] = [f'qk{t}' for t in range(first, min(first+2, tiles))]
        deps[f'pair{first//2}'] += [f'packed{t}' for t in range(first+2, min(first+4, tiles))]
    return list(TopologicalSorter(deps).static_order())


def hbm_audit(case, times):
    """Keep measured multi-core contention; reject newly competing DMA windows.

    Not a bandwidth fit: the traces do not identify separate bus service times.
    Startup=200 plus a measured residual is retained for every MTE2 transfer.
    Candidates cannot reorder/advance ingress, or change bytes or overlap peers.
    """
    dma = []
    for role, aiv in enumerate(case['aivs']):
        for kind in ('query', 'packed', 'scales'):
            spans = [aiv[kind]] if kind == 'query' else aiv[kind]
            for i,(start,end) in enumerate(spans):
                assert end-start >= MTE2_STARTUP
                dma.append(dict(core=f'AIV{role}', kind=kind, index=i, start=start,
                                end=end, initiation=MTE2_STARTUP,
                                measured_service_contention_residual=end-start-MTE2_STARTUP))
    latest_ingress = max(d['end'] for d in dma)
    earliest_output = min(times[f'AIV{r}.output0.start'] for r in range(2))
    assert earliest_output > latest_ingress, 'new input/output HBM contention: reject'
    # Both roles' output move by the same downstream quorum amount.
    shift = {times[f'AIV{r}.output{i}.start']-span[0]
             for r,a in enumerate(case['aivs']) for i,span in enumerate(a['output'])}
    assert len(shift) == 1, 'output contention changed: reject'
    return dict(policy='fixed measured shared-bus envelopes; new overlap rejected',
                mte2=dma, output_common_shift=shift.pop(), latest_ingress=latest_ingress,
                earliest_output=earliest_output,
                aic_policy='unchanged AIC memory traffic and upstream scalar cold costs')


def render(case, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'scalar':'#ba7c37','vector':'#367aa7','vf':'#5362a8','wait':'#cdd3da','cube':'#795ca4','tail':'#4b8a70'}
    fig, axes = plt.subplots(len(runs), 2, figsize=(15, 3.0*len(runs)),
                             gridspec_kw={'width_ratios':[2,1]})
    for row,(name,time,variant) in enumerate(runs):
        ax, tail = axes[row]
        for role in range(2):
            p=f'AIV{role}'; y=1-role
            w=WEIGHTS[0][role]
            scalar_end=time[f'{p}.scale.scalar.end']
            scalar_start=scalar_end-w['scalar_prefix']
            vector_start=time[f'{p}.scale.vector.start']
            key_end=time[f'{p}.key1.done']
            prefix_end=time[f'{p}.prefix.end']
            vfstart=time[f'{p}.pairedVF.start']; vfend=time[f'{p}.pairedVF.end']
            segments=[(scalar_start,scalar_end,'scalar'), (vector_start,key_end,'vector'),
                      (max(key_end, w['score_ready']) if variant=='key1_hoist' else key_end,prefix_end,'vector'),
                      (vfstart,vfend,'vf')]
            ax.broken_barh([(w['wait_start'],w['score_ready']-w['wait_start'])],(y-.32,.64),facecolors=colors['wait'])
            for start,end,kind in segments:
                ax.broken_barh([(start,end-start)],(y-.23,.46),facecolors=colors[kind])
            ax.plot([w['score_ready']]*2,[y-.4,y+.4],color='#9a3544',lw=1)
            ax.text(vfstart+8,y,'pair VF',fontsize=8,va='center',color='white')
        ax.set(yticks=[0,1],yticklabels=['AIV1','AIV0'],xlim=(24650,25850),ylim=(-.6,1.7))
        ax.set_title(name,loc='left',weight='bold',fontsize=11)
        ax.grid(axis='x',alpha=.2); ax.set_xlabel('Absolute model ticks — final-pair window')
        for j,role in enumerate(('AIC','AIV0','AIV1')):
            end=time[f'{role}.END']
            start=time['AIC.PV3.post'] if role=='AIC' else time[f'{role}.output0.start']
            tail.broken_barh([(start,end-start)],(2-j-.22,.44),facecolors=colors['tail'])
        tail.axvline(time['STOP'],color='#273a4a',linestyle='--')
        tail.text(time['STOP']-50,2.5,f"{time['STOP']:,} ticks",ha='right',weight='bold')
        tail.set(xlim=(27000,35400),ylim=(-.6,2.9),yticks=[0,1,2],yticklabels=['AIV1','AIV0','AIC'])
        tail.grid(axis='x',alpha=.2); tail.set_xlabel('Output, joins and independent core sinks')
        for a in (ax,tail):
            a.spines[['top','right']].set_visible(False)
    fig.suptitle('Wave-v13 conditional Max-Plus packing — measured weights retained',fontsize=16,weight='bold')
    fig.text(.5,.012,'Gray: original score wait · Ochre: scalar descriptors · Blue: scale work · Purple: paired VF · Red line: score1 ready\nPredictions assume unchanged compiler costs; no CAModel or hardware win is claimed.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.96))
    fig.savefig(HERE/'predicted-gantt.png',dpi=180)
    fig.savefig(HERE/'predicted-gantt.svg')
    plt.close(fig)


def main():
    global WEIGHTS
    cases=json.loads(COMPARISON.read_text())
    assert [c['ticks'] for c in cases] == [34935,35271]
    assert UB_BYTES <= UB_LIMIT
    WEIGHTS=[extract(c,r) for c,r in zip(cases,RUNS)]
    calibration=[]
    for case,w in zip(cases,WEIGHTS):
        g,t,c=build(case,w)
        assert t['STOP']==case['ticks']
        for role,a in enumerate(case['aivs']):
            assert t[f'AIV{role}.pairedVF.start']==a['vf'][1][0]
            assert t[f'AIV{role}.END']==a['last']
        calibration.append(dict(measured=case['ticks'],replayed=t['STOP'],max_error=0))
    for n in (1,2,4,5,8,16):
        ring_check(n)
    try:
        ring_check(4,illegal=True)
    except CycleError:
        pass
    else:
        raise AssertionError('illegal packed/L1 reuse not rejected')
    frontier=[]; charts=[]
    for name,variant in [('Wave-v11 measured parent','parent'),
                         ('K: hoist final key1 scale helper','key1_hoist'),
                         ('KV: hoist final key/value prefix','full_prefix_hoist'),
                         ('Conservative no scalar overlap','serialized_prefix')]:
        g,t,c=build(cases[0],WEIGHTS[0],variant)
        bus=hbm_audit(cases[0],t)
        stress=build(cases[0],WEIGHTS[0],variant,457,191)[1]['STOP']
        chain=[]; node='STOP'
        while node:
            chain.append(node);node=c[node]
        result=dict(name=name,variant=variant,makespan=t['STOP'],savings=34935-t['STOP'],
                    stress_457_plus_191=stress,critical_chain=list(reversed(chain)),
                    times=t,shared_hbm=bus,
                    edges={n:[asdict(e) for e in es] for n,es in g.edges.items()})
        frontier.append(result)
        if variant!='serialized_prefix':charts.append((name,t,variant))
    # The broader prefix already contains K. Combining them is idempotent.
    assert frontier[1]['makespan']==frontier[2]['makespan']
    output=dict(calibration=calibration,weights=WEIGHTS,ub_bytes=UB_BYTES,ub_limit=UB_LIMIT,
                mte2_startup=MTE2_STARTUP,comparison_sha256=digest(COMPARISON),
                frontier=frontier,combination='K+KV = KV; no additive saving',
                held_out_slot01=dict(measured=36494,status='not predicted: local final-pair model cannot forecast new prologue'),
                uncertainty='457+191 sensitivity is an observed-scale stress scenario, not a probabilistic bound',
                validation='Independent longest-path and event-readiness solvers match all nodes; two baselines replay exactly; finite ring horizons and cycle negative pass')
    (HERE/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    render(cases[0],charts)
    print(json.dumps(dict(calibration=calibration,candidates=[{k:r[k] for k in ('name','makespan','savings','stress_457_plus_191')} for r in frontier]),indent=2))


if __name__=='__main__':
    main()
