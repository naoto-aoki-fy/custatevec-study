#include "hadamard_order.hpp"

#include <cassert>
#include <vector>

int main() {
    for (int nqubits : {1, 3, 4, 5, 6, 8, 31, 62}) {
        for (bool reverse : {false, true}) {
            std::vector<int> targets;
            bench::for_each_hadamard_target(nqubits, reverse,
                [&](int logical) { targets.push_back(logical); });
            assert(targets.size() == static_cast<std::size_t>(nqubits));
            for (int i = 0; i < nqubits; ++i)
                assert(targets[i] == (reverse ? nqubits - 1 - i : i));
        }
    }
    for (int nqubits : {1, 3, 4, 5, 6, 8, 31, 62}) {
        for (auto range : {bench::HadamardRange::low, bench::HadamardRange::high}) {
            std::vector<int> targets, expected;
            bench::for_each_hadamard_target(nqubits, false,
                [&](int logical) { targets.push_back(logical); }, range);
            for (int q = 0; q < nqubits; ++q)
                if ((range == bench::HadamardRange::low && q < 5) ||
                    (range == bench::HadamardRange::high && q >= 5))
                    expected.push_back(q);
            assert(targets == expected);
        }
    }

}
