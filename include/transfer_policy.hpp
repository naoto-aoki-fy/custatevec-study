#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace bench::transfer_policy {
using Bytes = std::uint64_t;
inline bool valid_fraction(double fraction) {
    return std::isfinite(fraction) && fraction > 0 && fraction <= 0.95;
}
inline Bytes budget(Bytes free, double fraction) {
    if (!valid_fraction(fraction))
        throw std::runtime_error("--transfer-buffer-fraction must be finite and in (0,0.95]");
    // Exact integer arithmetic prevents a rounded floating product from
    // exceeding the cap at large sizes. Treat the default 0.95 as exactly 19/20.
    using Wide = unsigned __int128;
    const Bytes cap = static_cast<Bytes>(Wide(free) * 19 / 20);
    if (fraction == 0.95) return cap;
    int exponent = 0;
    const double mantissa = std::frexp(fraction, &exponent);
    const auto significand = static_cast<Bytes>(std::ldexp(mantissa, 53));
    const int shift = 53 - exponent;
    if (shift >= 128) return 0;
    return std::min(cap, static_cast<Bytes>((Wide(free) * significand) >> shift));
}
inline Bytes round_down(Bytes value) {
    if (!value) return 0;
    Bytes result = 1;
    while (value >>= 1) result <<= 1;
    return result;
}
inline Bytes select(Bytes common_budget, Bytes minimum, bool automatic) {
    const Bytes largest = round_down(common_budget);
    if (!largest || minimum > largest)
        throw std::runtime_error("Transfer memory budget insufficient: common_budget=" +
            std::to_string(common_budget) + " required_minimum=" + std::to_string(minimum));
    if (automatic) return largest;
    Bytes smallest = 1;
    while (smallest < minimum) smallest <<= 1; // bounded by largest
    return smallest;
}
// attempt returns world-wide success. release frees every successful local
// allocation after a failed attempt, before refresh queries the new budget.
// At most 64 attempts, strictly decreasing powers of two, including min mode.
template<class Attempt, class Release, class Refresh, class Report>
Bytes allocate_common(Bytes initial, Bytes minimum, Attempt attempt, Release release,
                      Refresh refresh, Report report) {
    Bytes candidate = initial;
    for (;;) {
        if (attempt(candidate)) return candidate;
        release();
        const Bytes next = std::min(candidate >> 1, round_down(refresh()));
        report(candidate, next);
        if (!next || next < minimum)
            throw std::runtime_error("Transfer memory budget exhausted after collective allocation failure: "
                "next_bytes=" + std::to_string(next) + " required_minimum=" + std::to_string(minimum));
        candidate = next;
    }
}
} // namespace bench::transfer_policy
