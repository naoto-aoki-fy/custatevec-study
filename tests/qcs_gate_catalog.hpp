#pragma once
#include "qcs.h"
#include "../src/qcs_gate.hpp"
#include "qcs_gate_reference.hpp"
#include <array>
#include <vector>
#include <stdexcept>
using GateKind=qcs_gate::Kind;
inline std::vector<qcs_gate::Complex> reference_column(GateKind kind,int column,std::array<double,4> a) {
    int d=1<<qcs_gate::arity(kind);
    std::vector<qcs_gate::Complex> in(d),out(d);
    in[column]=1; out=in;
    std::vector<const qcs_gate::Complex*> input(d);
    std::vector<qcs_gate::Complex*> output(d);
    for(int j=0;j<d;++j) { input[j]=&in[j]; output[j]=&out[j]; }
    switch(kind) {
    case GateKind::global_phase: reference_gate::gate::global_phase(a[0]).apply(input.data(),output.data()); break;
    case GateKind::h: reference_gate::gate::hadamard().apply(input.data(),output.data()); break;
    case GateKind::x: reference_gate::gate::x().apply(input.data(),output.data()); break;
    case GateKind::y: reference_gate::gate::y().apply(input.data(),output.data()); break;
    case GateKind::z: reference_gate::gate::z().apply(input.data(),output.data()); break;
    case GateKind::s: reference_gate::gate::s().apply(input.data(),output.data()); break;
    case GateKind::sdg: reference_gate::gate::sdg().apply(input.data(),output.data()); break;
    case GateKind::t: reference_gate::gate::t().apply(input.data(),output.data()); break;
    case GateKind::tdg: reference_gate::gate::tdg().apply(input.data(),output.data()); break;
    case GateKind::sx: reference_gate::gate::sx().apply(input.data(),output.data()); break;
    case GateKind::rx: reference_gate::gate::rx(a[0]).apply(input.data(),output.data()); break;
    case GateKind::ry: reference_gate::gate::ry(a[0]).apply(input.data(),output.data()); break;
    case GateKind::rz: reference_gate::gate::rz(a[0]).apply(input.data(),output.data()); break;
    case GateKind::u4: reference_gate::gate::u4(a[0], a[1], a[2], a[3]).apply(input.data(),output.data()); break;
    case GateKind::u3: reference_gate::gate::u3(a[0], a[1], a[2]).apply(input.data(),output.data()); break;
    case GateKind::u2: reference_gate::gate::u2(a[0], a[1]).apply(input.data(),output.data()); break;
    case GateKind::u1: reference_gate::gate::u1(a[0]).apply(input.data(),output.data()); break;
    case GateKind::u: reference_gate::gate::u(a[0], a[1], a[2]).apply(input.data(),output.data()); break;
    case GateKind::p: reference_gate::gate::p(a[0]).apply(input.data(),output.data()); break;
    case GateKind::swap: reference_gate::gate::swap<1>().apply(input.data(),output.data()); break;
    case GateKind::iswap: reference_gate::gate::iswap().apply(input.data(),output.data()); break;
    case GateKind::id: reference_gate::gate::id().apply(input.data(),output.data()); break;
    case GateKind::sxdg: reference_gate::gate::sxdg().apply(input.data(),output.data()); break;
    case GateKind::r: reference_gate::gate::r(a[0], a[1]).apply(input.data(),output.data()); break;
    case GateKind::rxx: reference_gate::gate::rxx(a[0]).apply(input.data(),output.data()); break;
    case GateKind::ryy: reference_gate::gate::ryy(a[0]).apply(input.data(),output.data()); break;
    case GateKind::rzz: reference_gate::gate::rzz(a[0]).apply(input.data(),output.data()); break;
    case GateKind::rzx: reference_gate::gate::rzx(a[0]).apply(input.data(),output.data()); break;
    case GateKind::dcx: reference_gate::gate::dcx().apply(input.data(),output.data()); break;
    case GateKind::ecr: reference_gate::gate::ecr().apply(input.data(),output.data()); break;
    case GateKind::xx_plus_yy: reference_gate::gate::xx_plus_yy(a[0], a[1]).apply(input.data(),output.data()); break;
    case GateKind::xx_minus_yy: reference_gate::gate::xx_minus_yy(a[0], a[1]).apply(input.data(),output.data()); break;
    case GateKind::rccx: reference_gate::gate::rccx().apply(input.data(),output.data()); break;
    case GateKind::rcccx: reference_gate::gate::rcccx().apply(input.data(),output.data()); break;
    }
    return out;
}
inline const std::vector<GateKind> all_gates = {
    GateKind::global_phase, GateKind::h, GateKind::x, GateKind::y, GateKind::z, GateKind::s, GateKind::sdg, GateKind::t, GateKind::tdg, GateKind::sx, GateKind::rx, GateKind::ry, GateKind::rz, GateKind::u4, GateKind::u3, GateKind::u2, GateKind::u1, GateKind::u, GateKind::p, GateKind::swap, GateKind::iswap, GateKind::id, GateKind::sxdg, GateKind::r, GateKind::rxx, GateKind::ryy, GateKind::rzz, GateKind::rzx, GateKind::dcx, GateKind::ecr, GateKind::xx_plus_yy, GateKind::xx_minus_yy, GateKind::rccx, GateKind::rcccx
};
#ifdef QCS_GATE_API_TEST
inline void call_gate(qcs_simulator* sim,GateKind kind,std::array<double,4> a,
                      const std::vector<bit_num_t>& t,const std::vector<bit_num_t>& n,
                      const std::vector<bit_num_t>& p,bool c_api) {
    int ok=0;
    switch(kind) {
    case GateKind::global_phase: if(c_api) ok=qcs_simulator_gate_global_phase(sim, a[0], n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_global_phase_cxx(sim, a[0], n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::h: if(c_api) ok=qcs_simulator_gate_h(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_h_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::x: if(c_api) ok=qcs_simulator_gate_x(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_x_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::y: if(c_api) ok=qcs_simulator_gate_y(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_y_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::z: if(c_api) ok=qcs_simulator_gate_z(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_z_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::s: if(c_api) ok=qcs_simulator_gate_s(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_s_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::sdg: if(c_api) ok=qcs_simulator_gate_sdg(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_sdg_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::t: if(c_api) ok=qcs_simulator_gate_t(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_t_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::tdg: if(c_api) ok=qcs_simulator_gate_tdg(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_tdg_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::sx: if(c_api) ok=qcs_simulator_gate_sx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_sx_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::rx: if(c_api) ok=qcs_simulator_gate_rx(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_rx_cxx(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::ry: if(c_api) ok=qcs_simulator_gate_ry(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_ry_cxx(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::rz: if(c_api) ok=qcs_simulator_gate_rz(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_rz_cxx(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::u4: if(c_api) ok=qcs_simulator_gate_u4(sim, a[0], a[1], a[2], a[3], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_u4_cxx(sim, a[0], a[1], a[2], a[3], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::u3: if(c_api) ok=qcs_simulator_gate_u3(sim, a[0], a[1], a[2], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_u3_cxx(sim, a[0], a[1], a[2], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::u2: if(c_api) ok=qcs_simulator_gate_u2(sim, a[0], a[1], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_u2_cxx(sim, a[0], a[1], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::u1: if(c_api) ok=qcs_simulator_gate_u1(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_u1_cxx(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::u: if(c_api) ok=qcs_simulator_gate_u(sim, a[0], a[1], a[2], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_u_cxx(sim, a[0], a[1], a[2], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::p: if(c_api) ok=qcs_simulator_gate_p(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_p_cxx(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::swap: if(c_api) ok=qcs_simulator_gate_swap(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_swap_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::iswap: if(c_api) ok=qcs_simulator_gate_iswap(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_iswap_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::id: if(c_api) ok=qcs_simulator_gate_id(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_id_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::sxdg: if(c_api) ok=qcs_simulator_gate_sxdg(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_sxdg_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::r: if(c_api) ok=qcs_simulator_gate_r(sim, a[0], a[1], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_r_cxx(sim, a[0], a[1], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::rxx: if(c_api) ok=qcs_simulator_gate_rxx(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_rxx_cxx(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::ryy: if(c_api) ok=qcs_simulator_gate_ryy(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_ryy_cxx(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::rzz: if(c_api) ok=qcs_simulator_gate_rzz(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_rzz_cxx(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::rzx: if(c_api) ok=qcs_simulator_gate_rzx(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_rzx_cxx(sim, a[0], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::dcx: if(c_api) ok=qcs_simulator_gate_dcx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_dcx_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::ecr: if(c_api) ok=qcs_simulator_gate_ecr(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_ecr_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::xx_plus_yy: if(c_api) ok=qcs_simulator_gate_xx_plus_yy(sim, a[0], a[1], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_xx_plus_yy_cxx(sim, a[0], a[1], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::xx_minus_yy: if(c_api) ok=qcs_simulator_gate_xx_minus_yy(sim, a[0], a[1], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_xx_minus_yy_cxx(sim, a[0], a[1], t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::rccx: if(c_api) ok=qcs_simulator_gate_rccx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_rccx_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    case GateKind::rcccx: if(c_api) ok=qcs_simulator_gate_rcccx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); else { qcs_simulator_gate_rcccx_cxx(sim, t.data(), bit_num_t(t.size()), n.data(), bit_num_t(n.size()), p.data(), bit_num_t(p.size())); ok=1; } break;
    }
    if(!ok) throw std::runtime_error("C gate failed");
}
#endif
