#pragma once

namespace bench {

enum class HadamardRange { all, low, high };

inline int hadamard_begin(int nqubits, HadamardRange range) {
    return range == HadamardRange::high ? (nqubits < 5 ? nqubits : 5) : 0;
}

inline int hadamard_end(int nqubits, HadamardRange range) {
    return range == HadamardRange::low && nqubits > 5 ? 5 : nqubits;
}

// Logical indices only: physical positions may change after every gate.
template<class Apply>
void for_each_hadamard_target(int nqubits, bool reverse, Apply apply,
                              HadamardRange range = HadamardRange::all) {
    const int begin = hadamard_begin(nqubits, range);
    const int end = hadamard_end(nqubits, range);
    for (int step = 0; step < end - begin; ++step)
        apply(reverse ? end - 1 - step : begin + step);
}

} // namespace bench
