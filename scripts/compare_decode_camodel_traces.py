"""Compare native device-kernel spans; extrapolate without repeating setup.

Old endpoints export partials and include query rotation. New endpoints take
rotated queries and emit normalized context. Report those boundaries explicitly;
never label their raw span ratio a measured full-operator/hardware speedup.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path


def point(path, context, old=False):
    raw = path.read_bytes()
    document = json.loads(raw)
    events = document if isinstance(document, list) else document['traceEvents']
    work = [e for e in events if e.get('ph') == 'X']
    start = min(e['ts'] for e in work)
    end = max(e['ts'] + e['dur'] for e in work)
    mmads = sorted((e for e in work if e.get('name') == 'MMAD'), key=lambda e: e['ts'])
    expected = 64 + 8 * (context // 16) if old else 2 * (context // 64)
    if len(mmads) != expected:
        raise ValueError(f'{path}: {len(mmads)} MMADs, expected {expected}')
    result = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), context=context,
                  start_us=start, end_us=end, span_us=end-start, events=len(events), mmads=len(mmads))
    if old:
        result['query_rotation_mmads'] = 64
        result['pre_score_prefix_us'] = mmads[64]['ts'] - start
        result['rotation_mmads_elapsed_us'] = max(e['ts']+e['dur'] for e in mmads[:64])-mmads[0]['ts']
        result['rotation_mmads_pcs'] = [e.get('args', {}).get('pc_addr') for e in mmads[:64]]
    return result


def fit(a, b):
    slope = (b['span_us'] - a['span_us']) / (b['context'] - a['context'])
    intercept = a['span_us'] - slope * a['context']
    if slope <= 0 or intercept < 0:
        raise ValueError('nonphysical two-point affine fit')
    return dict(fixed_us=intercept, us_per_token=slope,
                fitted_contexts=[a['context'], b['context']])


def predict(model, n):
    return model['fixed_us'] + model['us_per_token'] * n


def overlay(old, new, destination):
    events = []
    for number, (label, datum) in enumerate((('OLD BF16 partial export', old), ('NEW FP8 normalized output', new))):
        document = json.loads(Path(datum['path']).read_bytes())
        source = document if isinstance(document, list) else document['traceEvents']
        for event in source:
            e = copy.deepcopy(event)
            if 'pid' in e: e['pid'] = f"{label} / {e['pid']}"
            if 'ts' in e: e['ts'] -= datum['start_us']
            if e.get('name') == 'process_sort_index':
                e['args']['sort_index'] += number * 3
            events.append(e)
    destination.write_text(json.dumps(dict(traceEvents=events, displayTimeUnit='us',
        otherData=dict(context=128, alignment='Each independent launch shifted to its first traced instruction.',
                       note='Comparison overlay; not one concurrent execution. Durations are unchanged.',
                       sources=[old, new])), separators=(',', ':')), encoding='utf-8')


def make_plot(out, points, old_model, new_model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ('#b65118', '#176ba0')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for ax, maximum in zip(axes, (1024, 16384)):
        for color, label, model, series in zip(colors,
                ('BF16 AIC: raw Q → partials', 'FP8 fused: rotated Q → context'),
                (old_model, new_model), (points[:2], points[2:])):
            ns = [128 + (maximum-128)*i/200 for i in range(201)]
            ax.plot(ns, [predict(model, n) for n in ns], '--', color=color, label=label)
            lo, hi = model['fitted_contexts']
            ax.plot([lo, hi], [predict(model, lo), predict(model, hi)], color=color, linewidth=2.5)
            ax.scatter([p['context'] for p in series], [p['span_us'] for p in series],
                       color=color, s=32, zorder=3)
        ax.set_xlim(0, maximum)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Context tokens')
        ax.set_ylabel('Observed / projected kernel span (CAModel µs)')
        ax.grid(alpha=.2)
        ax.spines[['right', 'top']].set_visible(False)
    axes[0].legend(loc='upper left', frameon=False, fontsize=8)
    fig.suptitle('Decode trace extrapolation: one KV head, 16 shared query rows, D=256, K=1', fontsize=12)
    axes[0].set_title('Short contexts; dots = native captures', fontsize=10)
    axes[1].set_title('Long-context projection; dashed = extrapolation', fontsize=10)
    fig.supxlabel('Different endpoint work: old still needs normalization; new still needs query rotation. No hardware timing claim.', fontsize=8)
    fig.savefig(out / 'comparison.png', dpi=170)
    fig.savefig(out / 'comparison.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-short', 'old-long', 'new-short', 'new-medium', 'new-long', 'out'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    points = [point(args.old_short, 16, True), point(args.old_long, 128, True),
              point(args.new_short, 128), point(args.new_medium, 384), point(args.new_long, 768)]
    old, new = fit(*points[:2]), fit(*points[3:])
    prefix = max(p['pre_score_prefix_us'] for p in points[:2])
    rows = []
    for n in (128, 384, 768, 1024, 4096, 8192, 16384):
        o = next((p['span_us'] for p in points[:2] if p['context'] == n), predict(old, n))
        f = next((p['span_us'] for p in points[2:] if p['context'] == n), predict(new, n))
        rows.append(dict(context=n, old_partial_span_us=o, new_context_span_us=f,
                         old_source='observed' if n == 128 else 'extrapolated',
                         new_source='observed' if n in (128,384,768) else 'extrapolated',
                         span_ratio=o/f, new_missing_stage_budget_us=o-f,
                         new_with_old_prefix_proxy_us=f+prefix, proxy_ratio=o/(f+prefix)))
    result = dict(points=points, old_model=old, new_model=new, comparison=rows,
        rotation_pc_fingerprint_equal=points[0]['rotation_mmads_pcs']==points[1]['rotation_mmads_pcs'],
        short_new_holdout_error_us=predict(new,128)-points[2]['span_us'],
        old_pre_score_proxy_us=prefix,
        boundaries=dict(old='raw BF16 query -> (m,l,acc) partial export; includes query rotation',
                        new='pre-rotated FP32 query -> normalized FP16 rotated-basis context',
                        full_old='old_span + old_normalize_and_output',
                        full_new='new_span + new_query_rotation',
                        conditional_win='new_query_rotation - old_normalize_and_output < old_span - new_span'),
        assumptions=['Fixed 16 shared query rows, D=256, one KV head/request, K=1, one MIX block.',
                     'Native span is max(ts+dur)-min(ts), across all core/engine events; busy times are never summed.',
                     'Old fit uses N=16 and N=128; page_tokens differ (16 versus 128). Rotation MMAD PC sequence is checked.',
                     'New fit uses full modulo periods N=384 and N=768; N=128 is a short-context holdout.',
                     'Extrapolation assumes constant per-token work and no new memory/cache/scheduling regime.',
                     'Full end-to-end also needs the missing boundary stages. Host/ACL launch overhead is not in the trace.',
                     'The proxy scenario charges the new path the entire observed old pre-score prefix and charges old normalization zero; this is an assumption, not a measured end-to-end time.',
                     'FP8 versus BF16, Cube PV versus Vector PV, and different fixtures prevent attributing ratios solely to scheduling.'])
    out=args.out; out.mkdir(parents=True, exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    overlay(points[1],points[2],out/'matched_context_128_overlay.json')
    make_plot(out,points,old,new)
    md = ['# BF16 AIC versus fused FP8: trace-based context extrapolation','',
          'One KV head, 16 shared query rows, D=256, K=1. All values below are **CAModel microseconds**, not silicon latency.', '',
          '**Boundary difference:** the BF16 trace includes raw-query rotation and exports partials; the FP8 trace accepts a rotated query and writes normalized context. The span ratio is not a measured full-operator speedup.', '',
          '## Native observations', '', '| Path | Context | MMADs | First instruction → last completion (µs) |', '| --- | ---: | ---: | ---: |']
    for i,p in enumerate(points):
        md.append(f"| {'BF16 AIC' if i<2 else 'FP8 fused'} | {p['context']} | {p['mmads']} | {p['span_us']:.6f} |")
    md += ['', '## Affine models', '',
           f"`T_BF16(N) = {old['fixed_us']:.6f} + {old['us_per_token']:.9f} × N` µs.", '',
           f"`T_FP8(N) = {new['fixed_us']:.6f} + {new['us_per_token']:.9f} × N` µs.", '',
           f"The BF16 body costs {old['us_per_token']*16:.6f} µs per 16-token strip; the FP8 body costs {new['us_per_token']*64:.6f} µs per 64-token tile. Fixed overhead is charged once, not once per strip.", '',
           f"The new fit uses two complete modulo-period points (6 and 12 tiles). Its N=128 holdout residual is {result['short_new_holdout_error_us']:+.6f} µs. The old rotation-PC fingerprint matches: {result['rotation_pc_fingerprint_equal']}.", '',
           '## Same-context spans', '',
           '| Context | BF16 partial-export span | FP8 normalized-output span | Span ratio | New missing-stage budget |',
           '| ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        md.append(f"| {r['context']} | {r['old_partial_span_us']:.3f} ({r['old_source']}) | {r['new_context_span_us']:.3f} ({r['new_source']}) | {r['span_ratio']:.2f}× | {r['new_missing_stage_budget_us']:.3f} µs |")
    md += ['', '## End-to-end interpretation', '',
           '`T_old_full = T_BF16 + C_old_normalize/output`; `T_new_full = T_FP8 + C_new_query_rotation` for a common raw-query → normalized rotated-basis context boundary.', '',
           'The new path wins that boundary if `C_new_query_rotation − C_old_normalize/output` is below the budget column. Those costs are not present in both captures, so a unique full end-to-end number cannot be measured from these traces.', '',
           f"For an explicit **same-transform-cost proxy**, charge the new path the entire old pre-score prefix ({prefix:.6f} µs) and charge the old path zero normalization cost. This prefix includes setup/ingress as well as rotation; it is not an isolated measurement of the new prologue.", '',
           '| Context | BF16, normalization free (µs) | FP8 + old-prefix proxy (µs) | Conditional ratio |', '| ---: | ---: | ---: | ---: |']
    for r in rows:
        md.append(f"| {r['context']} | {r['old_partial_span_us']:.3f} | {r['new_with_old_prefix_proxy_us']:.3f} | {r['proxy_ratio']:.2f}× |")
    md += ['', 'Inverse basis/gating and host dispatch are additional costs if the target boundary requires them. Their equality between paths would be another assumption.', '',
           '## Assumptions and limits', ''] + ['* '+x for x in result['assumptions']] + ['',
           '## Artifacts', '',
           '* `comparison.json` / `comparison.csv`: exact inputs, hashes, model coefficients and projections.',
           '* `comparison.png` / `comparison.svg`: observed points and dashed extrapolations.',
           '* `matched_context_128_overlay.json`: two independent real N=128 traces, each shifted to start at zero and given distinct process names. Durations are unchanged; the overlay does not imply concurrent execution.', '',
           'Sources:', ''] + [f"* N={p['context']}: `{p['path']}`; SHA-256 `{p['sha256']}`." for p in points]
    (out/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(dict(old_model=old,new_model=new,comparison=rows),indent=2))


if __name__ == '__main__': main()
