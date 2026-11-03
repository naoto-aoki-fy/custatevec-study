#pragma once

namespace bench {

// Logical indices only: physical positions may change after every gate.
template<class Apply>
void for_each_hadamard_target(int nqubits, bool reverse, Apply apply) {
    for (int step = 0; step < nqubits; ++step)
        apply(reverse ? nqubits - 1 - step : step);
}

} // namespace bench
