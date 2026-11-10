#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/qcs_gate_timing.hpp"
#include "../src/qcs_gate.hpp"
#include <cassert>
#include <string>

int main() {
    FILE* output = std::tmpfile();
    assert(output);
    unsigned long long sequence = 0;
    for (const char* value : {"", "0", "true"}) {
        assert(setenv("QCS_GATE_TIMING", value, 1) == 0);
        qcs_gate::Timer timer;
        timer.finish(0, sequence, "h", false, output);
    }
    assert(unsetenv("QCS_GATE_TIMING") == 0);
    qcs_gate::Timer{}.finish(0, sequence, "h", false, output);
    assert(std::ftell(output) == 0 && sequence == 0);
    assert(setenv("QCS_GATE_TIMING", "1", 1) == 0);
    qcs_gate::Timer{}.finish(2, sequence, qcs_gate::name(qcs_gate::Kind::h), false, output);
    qcs_gate::Timer{}.finish(2, sequence, "swap", true, output);
    assert(sequence == 2);
    std::rewind(output);
    char line[512];
    for (unsigned long long index = 0; index < 2; ++index) {
        assert(std::fgets(line, sizeof(line), output));
        int rank = -1;
        unsigned long long actual = 99;
        assert(std::sscanf(line, "{\"type\":\"gate_timing\",\"rank\":%d,\"gate_index\":%llu", &rank, &actual) == 2);
        assert(rank == 2 && actual == index);
        std::string text(line);
        assert(text.find(index ? "\"skipped\":true" : "\"skipped\":false") != std::string::npos);
        auto start = text.find("\"elapsed_time\":");
        assert(start != std::string::npos && std::stod(text.substr(start + 15)) >= 0);
    }
    assert(!std::fgets(line, sizeof(line), output));
    std::fclose(output);
}
