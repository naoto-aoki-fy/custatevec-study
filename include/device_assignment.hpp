#pragma once

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace bench {

struct DeviceAssignment {
    int visible = 0;
    int device = -1;
    int local_rank = -1;
    std::string source = "single-visible-device";
};

namespace device_assignment {
inline int parse_rank(const char* text, const char* source) {
    const std::string error = std::string(source) + " must be a nonnegative decimal integer <= INT_MAX";
    if (!text || !*text) throw std::runtime_error(error);
    int value = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') throw std::runtime_error(error);
        const int digit = *p - '0';
        if (value > (std::numeric_limits<int>::max() - digit) / 10)
            throw std::runtime_error(error);
        value = value * 10 + digit;
    }
    return value;
}

inline int select_device(int visible, int local_rank) {
    if (visible < 1) throw std::runtime_error("No visible CUDA GPU");
    if (visible == 1) return 0;
    if (local_rank < 0)
        throw std::runtime_error("Multiple CUDA GPUs are visible but no supported launcher local rank is available; "
                                 "use Open MPI, a direct Slurm launch, or expose one distinct GPU per process");
    if (local_rank >= visible)
        throw std::runtime_error("GPU oversubscription: node-local rank exceeds visible GPU count");
    return local_rank;
}

// Open MPI's process rank takes precedence over an enclosing Slurm task's rank.
// getenv is injected so parsing and policy can be exercised without CUDA/MPI.
template<class Getenv>
DeviceAssignment discover(int visible, Getenv getenv) {
    DeviceAssignment result;
    result.visible = visible;
    for (const char* source : {"OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"}) {
        if (const char* value = getenv(source)) {
            result.source = source;
            result.local_rank = parse_rank(value, source);
            break;
        }
    }
    result.device = select_device(visible, result.local_rank);
    return result;
}

inline void validate(const DeviceAssignment& assignment, int actual_local_rank) {
    if (assignment.local_rank >= 0 && assignment.local_rank != actual_local_rank)
        throw std::runtime_error("Pre-MPI local rank from " + assignment.source + " (" +
            std::to_string(assignment.local_rank) + ") disagrees with MPI shared-node rank (" +
            std::to_string(actual_local_rank) + ")");
    if (assignment.device != select_device(assignment.visible, actual_local_rank))
        throw std::runtime_error("Pre-MPI CUDA device disagrees with MPI shared-node assignment policy");
}
} // namespace device_assignment
} // namespace bench
