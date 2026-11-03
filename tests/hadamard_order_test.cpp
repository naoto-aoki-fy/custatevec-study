#include "hadamard_order.hpp"

#include <cassert>
#include <vector>

int main() {
    for (int nqubits : {1, 3, 8, 31, 62}) {
        for (bool reverse : {false, true}) {
            std::vector<int> targets;
            bench::for_each_hadamard_target(nqubits, reverse,
                [&](int logical) { targets.push_back(logical); });
            assert(targets.size() == static_cast<std::size_t>(nqubits));
            for (int i = 0; i < nqubits; ++i)
                assert(targets[i] == (reverse ? nqubits - 1 - i : i));
        }
    }
}
