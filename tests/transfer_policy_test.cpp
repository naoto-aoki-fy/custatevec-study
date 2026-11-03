#include "transfer_policy.hpp"

#include <cassert>
#include <limits>
#include <vector>

using namespace bench::transfer_policy;
template<class F> void rejects(F f) {
    bool threw = false;
    try { f(); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
}
int main() {
    assert(round_down(0) == 0);
    assert(round_down(1) == 1);
    assert(round_down(7) == 4);
    assert(round_down(8) == 8);
    assert(round_down(9) == 8);
    for (unsigned bit = 1; bit < 64; ++bit) {
        const Bytes power = Bytes{1} << bit;
        assert(round_down(power - 1) == power / 2);
        assert(round_down(power) == power);
        assert(round_down(power + 1) == power);
    }
    assert(budget(100, 0.95) == 95);
    assert(budget(1, 0.95) == 0);
    assert(budget(100, 0.5) == 50);
    for (Bytes free = 0; free < 2048; ++free) {
        assert(budget(free, 0.95) == free * 19 / 20);
        assert(budget(free, 0.9) <= free * 19 / 20);
    }
    for (double f : {0.0, -0.1, 0.951, 1.0,
                     std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()})
        rejects([&] { budget(100, f); });
    assert(budget(100, std::numeric_limits<double>::denorm_min()) == 0);
    assert(valid_fraction(std::nextafter(0.95, 0.0)));
    assert(!valid_fraction(std::nextafter(0.95, 1.0)));
    rejects([] { select(0, 0, true); });
    rejects([] { select(127, 65, true); });
    rejects([] { select(127, 65, false); });
    assert(select(128, 65, false) == 128);
    assert(select(1024, 65, false) == 128);
    assert(select(1024, 65, true) == 1024);
    // Simulate MPI_MIN budgets and MPI_MAX worker requirements.
    const std::vector<Bytes> free = {4096, 2048, 8192, 3072};
    const std::vector<Bytes> minima = {16, 65, 32, 64};
    Bytes common = std::numeric_limits<Bytes>::max(), required = 0;
    for (unsigned r = 0; r < free.size(); ++r) {
        common = std::min(common, budget(free[r], 0.95));
        required = std::max(required, minima[r]);
    }
    assert(select(common, required, true) == 1024);
    assert(select(common, required, false) == 128);
    const Bytes maximum = std::numeric_limits<Bytes>::max();
    assert(round_down(maximum) == (Bytes{1} << 63));
    assert(budget(maximum, 0.95) == static_cast<Bytes>(static_cast<unsigned __int128>(maximum) * 19 / 20));
    assert(budget(maximum, 0.5) == maximum / 2);
    rejects([&] { select(maximum, maximum, false); });

    // Rank 1 fails the first collective attempt; successful rank 0 must free
    // before the refreshed query. The shrinking refresh can skip powers of two.
    std::vector<Bytes> allocated(2, 0), attempts;
    unsigned releases = 0, retries = 0;
    const auto final = allocate_common(1024, 128,
        [&](Bytes size) {
            attempts.push_back(size);
            allocated[0] = size;
            allocated[1] = attempts.size() == 1 ? 0 : size;
            return allocated[1] != 0;
        }, [&] { allocated.assign(2, 0); ++releases; },
        [&] { assert(allocated[0] == 0 && allocated[1] == 0); return Bytes{300}; },
        [&](Bytes old, Bytes next) { assert(next < old); ++retries; });
    assert((attempts == std::vector<Bytes>{1024, 256}));
    assert(final == 256 && releases == 1 && retries == 1);
    assert(allocated[0] == final && allocated[1] == final);
    rejects([] {
        allocate_common(1024, 128, [](Bytes) { return false; }, [] {},
            [] { return Bytes{127}; }, [](Bytes, Bytes) {});
    });
    unsigned failures = 0;
    rejects([&] {
        allocate_common(Bytes{1} << 63, 1, [](Bytes) { return false; }, [] {},
            [&] { return maximum; }, [&](Bytes, Bytes) { ++failures; });
    });
    assert(failures == 64);
    rejects([] {
        allocate_common(128, 65, [](Bytes) { return false; }, [] {},
            [] { return Bytes{4096}; }, [](Bytes, Bytes) {});
    });
}
