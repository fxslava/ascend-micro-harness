"""Render selected measured engine spans and the conditional final-pair shift."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
cases = json.loads((root.parent / 'wave-v12-slot02-vs-v11.json').read_text())
result = json.loads((root / 'results.json').read_text())
candidate = next(row for row in result['frontier'] if row['variant'] == 'key1_hoist')
panels = [('Wave-v11 measured', cases[0], None),
          ('Wave-v12 measured split prologue', cases[1], None),
          ('K candidate: conditional prediction', cases[0], candidate['times'])]
fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
for ax, (title, case, predicted) in zip(axes, panels):
    def spans(values, lane, color):
        ax.broken_barh([(a, b-a) for a,b in values], (lane-.31,.62), facecolors=color)
    for name, product in case['products'].items():
        shift = (predicted[f'AIC.{name}.post']-product['post']) if predicted and name in ('PV2','PV3') else 0
        spans([(product['mmad_pop']+shift, product['mmad_retire']+shift)],8,'#795ca4')
    for role, aiv in enumerate(case['aivs']):
        lane = 7-role*4
        spans([aiv['query']]+aiv['packed']+aiv['scales'], lane, '#ba7c37')
        spans(aiv['nz'], lane-1, '#4b8a70')
        vf = list(aiv['vf'])
        if predicted:
            vf[1] = (predicted[f'AIV{role}.pairedVF.start'], predicted[f'AIV{role}.pairedVF.end'])
        spans(vf, lane-2, '#5362a8')
        outputs = aiv['output'] if not predicted else [
            (predicted[f'AIV{role}.output{i}.start'],predicted[f'AIV{role}.output{i}.end'])
            for i in range(len(aiv['output']))]
        spans(outputs, lane-3, '#367aa7')
    stop = predicted['STOP'] if predicted else case['ticks']
    ax.axvline(stop, color='#253949', linestyle='--')
    ax.text(stop-80,8.6,f'{stop:,} ticks',ha='right',weight='bold')
    ax.axvspan(24650,25850,color='#cdd3da',alpha=.22)
    ax.set_title(title,loc='left',weight='bold')
    ax.set(ylim=(-.6,9),yticks=list(range(9)),yticklabels=[
        'AIV1 output','AIV1 paired VF','AIV1 NZ → L1','AIV1 MTE2',
        'AIV0 output','AIV0 paired VF','AIV0 NZ → L1','AIV0 MTE2','AIC MMAD'])
    ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axes[-1].set(xlim=(0,36000),xlabel='Absolute model ticks')
fig.suptitle('Prologue, ring and epilogue: measured spans with conditional final-pair packing',weight='bold',fontsize=15)
fig.text(.5,.012,'Selected traced engine intervals; blank space includes unplotted scalar, vector, synchronization and tail work.\nUpstream timing is retained. Prediction is a local counterfactual, not a new CAModel result.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.055,1,.96))
fig.savefig(root/'predicted-overview.png',dpi=160)
fig.savefig(root/'predicted-overview.svg')
