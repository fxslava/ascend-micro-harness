"""Clone the signed native parent and change only the final-pair score wait."""
from pathlib import Path
import shutil

root = Path('/home/ladislav/ascend-sync-schedule-ablation')
parent = root / 'agent_02/wave-v11/src'
candidate = root / 'wave-v13-k/src'
kernel = Path('src/decode/turboquant_fused_decode.cpp')
old = b'''        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagScoresReady + 1));
        vector_.template PreparePairLogits<1>();
'''
new = b'''        if (task.numTiles - first <= 3) {
            // The final pair's resident key scales do not depend on score 1.
            vector_.template PreparePairLogits<1>();
            AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagScoresReady + 1));
        } else {
            AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagScoresReady + 1));
            vector_.template PreparePairLogits<1>();
        }
'''
source = (parent / kernel).read_bytes()
assert source.count(old) == 1
assert b'PreparePairLogits<0>();\n' + old in source
candidate.parent.mkdir(exist_ok=False)
shutil.copytree(parent, candidate)
(candidate / kernel).write_bytes(source.replace(old, new))
assert [
    file.relative_to(parent)
    for file in parent.rglob('*')
    if file.is_file() and file.read_bytes() != (candidate / file.relative_to(parent)).read_bytes()
] == [kernel]
print('PASS only final-pair slot-1 key-scale scheduling differs from the signed parent.')
