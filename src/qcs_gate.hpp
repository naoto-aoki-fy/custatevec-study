#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qcs_gate {
using Complex = std::complex<double>;
enum class Kind { global_phase, h, x, y, z, s, sdg, t, tdg, sx, sxdg,
    rx, ry, rz, u4, u3, u2, u1, u, p, swap, iswap, id, r,
    rxx, ryy, rzz, rzx, dcx, ecr, xx_plus_yy, xx_minus_yy, rccx, rcccx };
inline int arity(Kind kind) {
    switch (kind) {
    case Kind::global_phase: return 0;
    case Kind::swap: case Kind::iswap: case Kind::rxx: case Kind::ryy:
    case Kind::rzz: case Kind::rzx: case Kind::dcx: case Kind::ecr:
    case Kind::xx_plus_yy: case Kind::xx_minus_yy: return 2;
    case Kind::rccx: return 3;
    case Kind::rcccx: return 4;
    default: return 1;
    }
}
// Row-major M[row,column], target list element zero is the low basis bit.
// SWAP uses this two-bit matrix repeatedly after localizing ALL requested targets.
inline std::vector<Complex> matrix(Kind kind, std::array<double, 4> a = {}) {
    const Complex i(0, 1);
    const auto phase = [](double theta) { return Complex(std::cos(theta), std::sin(theta)); };
    const int d = 1 << arity(kind);
    std::vector<Complex> m(d*d);
    auto at = [&](int row, int col) -> Complex& { return m[row*d+col]; };
    for (int j = 0; j < d; ++j) at(j,j) = 1;
    double c = std::cos(a[0]/2), s = std::sin(a[0]/2), h = std::sqrt(0.5);
    switch (kind) {
    case Kind::global_phase: at(0,0) = phase(a[0]); break;
    case Kind::h: m = {h,h,h,-h}; break;
    case Kind::x: m = {0,1,1,0}; break;
    case Kind::y: m = {0,-i,i,0}; break;
    case Kind::z: at(1,1) = -1; break;
    case Kind::s: at(1,1) = i; break;
    case Kind::sdg: at(1,1) = -i; break;
    case Kind::t: at(1,1) = Complex(h,h); break;
    case Kind::tdg: at(1,1) = Complex(h,-h); break;
    case Kind::sx: m = {(1.0+i)*0.5,(1.0-i)*0.5,(1.0-i)*0.5,(1.0+i)*0.5}; break;
    case Kind::sxdg: m = {(1.0-i)*0.5,(1.0+i)*0.5,(1.0+i)*0.5,(1.0-i)*0.5}; break;
    case Kind::rx: m = {c,-i*s,-i*s,c}; break;
    case Kind::ry: m = {c,-s,s,c}; break;
    case Kind::rz: m = {phase(-a[0]/2),0,0,phase(a[0]/2)}; break;
    case Kind::u2:
        m = {h,-h*phase(a[1]),h*phase(a[0]),h*phase(a[0]+a[1])}; break;
    case Kind::u: case Kind::u3: case Kind::u4:
        m = {c,-s*phase(a[2]),s*phase(a[1]),c*phase(a[1]+a[2])};
        if (kind == Kind::u4) for (auto& v : m) v *= phase(a[3]);
        break;
    case Kind::u1: case Kind::p: at(1,1) = phase(a[0]); break;
    case Kind::id: break;
    case Kind::r: m = {c,-i*s*phase(-a[1]),-i*s*phase(a[1]),c}; break;
    case Kind::swap: case Kind::iswap:
        at(1,1)=at(2,2)=0;
        at(1,2)=at(2,1)=kind == Kind::swap ? Complex(1) : i; break;
    case Kind::rzz:
        at(0,0)=at(3,3)=phase(-a[0]/2); at(1,1)=at(2,2)=phase(a[0]/2); break;
    case Kind::rxx: case Kind::ryy:
        for (int j=0;j<4;++j) { at(j,j)=c; at(j,j^3)=-i*s; }
        if (kind==Kind::ryy) at(0,3)=at(3,0)=i*s;
        break;
    case Kind::rzx:
        for (int j=0;j<4;++j) { at(j,j)=c; at(j,j^2)=(j&1 ? i : -i)*s; } break;
    case Kind::dcx:
        at(1,1)=at(2,2)=at(3,3)=0; at(1,3)=at(2,1)=at(3,2)=1; break;
    case Kind::ecr:
        std::fill(m.begin(),m.end(),Complex{});
        at(0,1)=at(1,0)=at(2,3)=at(3,2)=h;
        at(0,3)=at(2,1)=i*h; at(1,2)=at(3,0)=-i*h; break;
    case Kind::xx_plus_yy:
        at(1,1)=at(2,2)=c; at(1,2)=-i*s*phase(-a[1]); at(2,1)=-i*s*phase(a[1]); break;
    case Kind::xx_minus_yy:
        at(0,0)=at(3,3)=c; at(0,3)=-i*s*phase(-a[1]); at(3,0)=-i*s*phase(a[1]); break;
    case Kind::rccx:
        at(3,3)=at(7,7)=0; at(3,7)=-i; at(7,3)=i; at(5,5)=-1; break;
    case Kind::rcccx:
        at(3,3)=i; at(11,11)=-i; at(7,7)=at(15,15)=0; at(7,15)=1; at(15,7)=-1; break;
    }
    return m;
}
struct Localization {
    std::vector<int> p2l, l2p;
    std::vector<std::pair<int,int>> swaps; // local/global physical bits
};
inline Localization localize(const std::vector<int>& p2l, const std::vector<int>& targets, int local_bits) {
    Localization plan{p2l, std::vector<int>(p2l.size()), {}};
    for (std::size_t p=0;p<p2l.size();++p) plan.l2p[p2l[p]]=int(p);
    if (targets.size()>std::size_t(local_bits)) throw std::runtime_error("Number of targets exceeds local qubits");
    std::vector<int> physical;
    for (int q:targets) physical.push_back(plan.l2p[q]);
    int victim=local_bits-1;
    for (int p:physical) if (p>=local_bits) {
        while (victim>=0 && std::find(physical.begin(),physical.end(),victim)!=physical.end()) --victim;
        if (victim<0) throw std::runtime_error("No non-target local victim");
        plan.swaps.emplace_back(victim,p);
        std::swap(plan.p2l[victim],plan.p2l[p]); --victim;
    }
    for (std::size_t p=0;p<p2l.size();++p) plan.l2p[plan.p2l[p]]=int(p);
    return plan;
}
} // namespace qcs_gate
