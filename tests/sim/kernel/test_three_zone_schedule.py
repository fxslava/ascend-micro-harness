"""Compile the actual production scheduling methods with recording services.

This checks template phase selection and operation coverage, not arithmetic or
hardware timing. The independent capacity-one model and CAModel remain required.
"""
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'kernels/ops/turboquant/turboquant_fused_decode_opt.cpp'


def method(source, name):
    match = re.search(r'(?:    template <[^\n]+>\n)?'
                      r'    __aicore__ inline void ' + name + r'\(', source)
    assert match, name
    start = source.index('{', match.end())
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[match.start():end].replace('__aicore__ ', '')


def main():
    source = SOURCE.read_text()
    names = [f'Run{role}{part}' for role in ('Cube', 'Vector')
             for part in ('Pipeline', 'ShortPipeline', 'SteadyPhase', 'SteadyTail', 'Drain')]
    methods = '\n'.join(method(source, name) for name in names)
    for role in ('Cube', 'Vector'):
        body = method(source, f'Run{role}SteadyPhase')
        assert not re.search(r'\b(if|while|switch|for)\s*\(|\?', body), body
        assert 'numTiles' not in body
    cpp = r'''
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
std::vector<std::string> events;
void add(const char* name, unsigned slot=0, unsigned window=0) {
    events.push_back(std::string(name)+std::to_string(slot)+":"+std::to_string(window));
}
namespace AscendC { void CrossCoreWaitFlag(uint16_t slot) { add("W", slot); } }
constexpr unsigned kOptFlagSlotFree=0;
void SyncVectorToMte2() {}
struct TurboQuantTaskHeads {};
struct PipelineTask { unsigned numTiles, tile=0; };
struct Schedule {
    static constexpr unsigned kSlots=3, kWindows=2, kPhases=6;
    struct Vector { void ComputeAccumulatorDecay(const TurboQuantTaskHeads&) { add("D"); } } vector_;
    template<unsigned P> void CubeScores(unsigned) { add("S",P%3,P%2); }
    template<unsigned P> void CubeContext(unsigned) { add("C",P%3); }
    template<unsigned P> void ReleaseCubeSlot() { add("F",P%3); }
    template<unsigned P> void ReadPipelineTile(PipelineTask&) { add("R",P%3); }
    template<unsigned P> void VectorSoftmax(PipelineTask&,const TurboQuantTaskHeads&) { add("M",P%3,P%2); }
    void StageToL1(PipelineTask&,unsigned slot) { add("G",slot); }
    void VectorAccumulate(const TurboQuantTaskHeads&) { add("A"); }
''' + methods + r'''
};
int main() {
    for(unsigned n=1;n<=130;++n) for(unsigned cube=0;cube<2;++cube) {
        events.clear(); Schedule s; PipelineTask t{n};
        if(cube) s.RunCubePipeline(t,16); else s.RunVectorPipeline(t,{});
        std::cout<<n<<" "<<cube;
        for(const auto& e: events) std::cout<<" "<<e;
        std::cout<<"\n";
    }
}
'''
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        path = Path(directory)
        (path / 'schedule.cpp').write_text(cpp)
        subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                        str(path / 'schedule.cpp'), '-o', str(path / 'schedule')], check=True)
        lines = subprocess.check_output([str(path / 'schedule')], text=True).splitlines()
    def op(name, slot=0, window=0): return f'{name}{slot}:{window}'
    for line in lines:
        count, cube, *actual = line.split()
        n, cube = int(count), int(cube)
        expected = []
        def emit(name, slot=0, window=0): expected.append(op(name, slot, window))
        if cube:
            if n >= 4: emit('F', 2)
            emit('S')
            for k in range(n):
                if k + 1 < n: emit('S', (k+1)%3, (k+1)%2)
                emit('C', k%3)
                if k + 3 < n: emit('F', k%3)
        else:
            emit('R')
            if n > 1: emit('R', 1)
            emit('G')
            if n > 2: emit('R', 2)
            if n > 1: emit('G', 1)
            for k in range(n):
                emit('M', k%3, k%2)
                if k + 3 < n: emit('R', k%3)
                emit('D')
                if k + 2 < n:
                    if n >= 4: emit('W', (k+2)%3)
                    emit('G', (k+2)%3)
                emit('A')
        assert actual == expected, (n, cube, actual, expected)
    print('PASS actual C++ scheduling methods: N=1..130, all six residues, no steady boundary branches')


if __name__ == '__main__':
    main()
