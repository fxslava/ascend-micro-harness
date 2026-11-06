#include "../../include/turboquant_fused_decode_opt.hpp"
#include "../../include/turboquant_fused_decode.hpp"
#include "../../routines/routine_vector_online_softmax.hpp"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace fd = ascend_micro::fused_decode_opt;
namespace tq = vllm_ascend::turboquant;
static void require(bool ok) { if (!ok) throw std::runtime_error("fused decode contract"); }
struct Phases {
    unsigned phase = 0;
    void update_max_and_alpha() { require(phase++ == 0); }
    void exponentiate_scores() { require(phase++ == 1); }
    void update_denominator() { require(phase++ == 2); }
    void update_accumulator() { require(phase++ == 3); }
};

#include "../../kernels/ops/turboquant/vllm_integration_layer/tests/common/decode_tiling_gate.hpp"

int main()
{
    tiling_gate::run();
    // valid_launch refuses the same non-power-of-two page the tiling builder does. Checked here
    // rather than in the shared gate header, which must not include an operator contract header.
    require(!fd::valid_launch(1, 1, 16, 1, 256, 192, 8, 1, 16, 1, 16));
    require(fd::valid_launch(1, 1, 16, 1, 256, 128, 8, 1, 16, 1, 16));
    require(fd::valid_launch(1, 1, 17, 1, 256, 128, 8, 1, 16, 2, 17));
    require(!fd::valid_launch(1, 1, 17, 1, 256, 128, 8, 1, 16, 1, 17));
    require(!fd::valid_launch(1, 1, 16, 0, 256, 128, 8, 1, 16, 1, 16));
    require(!fd::valid_launch(1, 1, 16, 1, 256, 127, 8, 1, 16, 1, 16));
    require(!fd::valid_launch(1, 1, 16, 1, 512, 128, 8, 1, 16, 1, 16));
    for (unsigned heads = 1; heads <= 4; ++heads) {
        std::vector<bool> seen(128 * heads * 128);
        for (unsigned row = 0; row < 128; ++row)
            for (unsigned h = 0; h < heads; ++h)
                for (unsigned c = 0; c < 128; ++c) {
                    const auto i = tq::NzTiledPackedByte(row, h, c, heads, 128);
                    require(i < seen.size() && !seen[i]); seen[i] = true;
                }
        for (bool bit : seen) require(bit);
    }
    Phases cube, vector;
    ascend_micro::vector_online_softmax_probabilities(cube);
    ascend_micro::vector_online_softmax(vector);
    require(cube.phase == 3 && vector.phase == 4);
    std::printf("PASS host tiling (dispatch odometer == the six divisions it replaced, descriptors "
                "== the page walk, five refusals), launch guards, NZ bijection (1..4 heads), "
                "shared recurrence phases; UB=%u+256 bytes\n", fd::kUbEnd);
}
