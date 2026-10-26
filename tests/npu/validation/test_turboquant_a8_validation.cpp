#include "../turboquant_npu_common.hpp"

#include <cstdio>
#include <exception>

int main() {
    try {
        tq_npu::Session session;
        for (const int tokens : {16, 512, 1024}) {
            tq_npu::Fixture fixture(tokens);
            tq_npu::A8Device device(fixture);
            device.launch(session.stream, fixture);
            tq_npu::check(aclrtSynchronizeStream(session.stream), "aclrtSynchronizeStream");
            tq_npu::verify(fixture, device);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL turboquant A8 NPU validation: %s\n", e.what());
        return 1;
    }
}
