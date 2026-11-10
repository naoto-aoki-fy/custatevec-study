#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace qcs_gate {
// Explicit finish reports successful calls only; failed gates emit no timing.
class Timer {
    using Clock = std::chrono::steady_clock;
    bool enabled_;
    Clock::time_point start_{};
public:
    Timer() : enabled_(false) {
        const char* value = std::getenv("QCS_GATE_TIMING");
        enabled_ = value && std::strcmp(value, "1") == 0;
        if (enabled_) start_ = Clock::now();
    }
    void finish(int rank, unsigned long long& sequence, const char* gate,
                bool skipped, FILE* output = stderr) const {
        if (!enabled_) return;
        const double seconds = std::chrono::duration<double>(Clock::now() - start_).count();
        std::fprintf(output,
            "{\"type\":\"gate_timing\",\"rank\":%d,\"gate_index\":%llu,"
            "\"gate\":\"%s\",\"skipped\":%s,\"elapsed_time\":%.18g}\n",
            rank, sequence++, gate, skipped ? "true" : "false", seconds);
        std::fflush(output);
    }
};
}
