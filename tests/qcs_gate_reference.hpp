#pragma once
// Host oracle adapted from jhpc-quantum/rqs-svg src/qcs.cu, Apache-2.0,
// revision 10a4da5b29a65f3eec0f556344ce06c3ff24072a. Changes: host execution,
// std::complex, namespace isolation. Kept independent of the backend matrices.
#include <cmath>
#include <complex>
#include <cstdint>
namespace reference_gate {
using complex_t = std::complex<double>;
using float_t = double;
inline complex_t multiply_i(complex_t z) { return {-z.imag(),z.real()}; }
inline complex_t multiply_i_m(complex_t z) { return {z.imag(),-z.real()}; }
inline complex_t multiply_i_real(complex_t z,double x) { return {-x*z.imag(),x*z.real()}; }
namespace gate {
constexpr unsigned max_num_swaps=4;
        struct u4
        {
            static constexpr unsigned int num_target_qubits = 1;
            reference_gate::float_t cos_theta_2;
            reference_gate::float_t sin_theta_2;
            reference_gate::complex_t exp_i_phi;
            reference_gate::complex_t exp_i_lambda;
            reference_gate::complex_t exp_i_gamma;
            u4(double theta, double phi, double lambda, double gamma = 0.0)
            {
                reference_gate::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                sin_theta_2 = sin(theta_2);
                exp_i_phi = reference_gate::complex_t(cos(phi), sin(phi));
                exp_i_lambda = reference_gate::complex_t(cos(lambda), sin(lambda));
                exp_i_gamma = reference_gate::complex_t(cos(gamma), sin(gamma));
            }
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                reference_gate::complex_t const s0_in_copy = *s_in[0];
                reference_gate::complex_t const s1_in_copy = *s_in[1];
                *s_out[0] = exp_i_gamma * (cos_theta_2 * s0_in_copy - sin_theta_2 * exp_i_lambda * s1_in_copy);
                *s_out[1] = exp_i_gamma * (exp_i_phi * (sin_theta_2 * s0_in_copy + exp_i_lambda * cos_theta_2 * s1_in_copy));
            }
        };

        struct u3
        {
            static constexpr unsigned int num_target_qubits = 1;
            reference_gate::float_t cos_theta_2;
            reference_gate::float_t sin_theta_2;
            reference_gate::complex_t exp_i_phi;
            reference_gate::complex_t exp_i_lambda;
            reference_gate::complex_t exp_i_phi_plus_lambda;
            u3(double theta, double phi, double lambda)
            {
                reference_gate::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                sin_theta_2 = sin(theta_2);
                exp_i_phi = reference_gate::complex_t(cos(phi), sin(phi));
                exp_i_lambda = reference_gate::complex_t(cos(lambda), sin(lambda));
                reference_gate::float_t const phi_plus_lambda = phi + lambda;
                exp_i_phi_plus_lambda = reference_gate::complex_t(cos(phi_plus_lambda), sin(phi_plus_lambda));
            }
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                reference_gate::complex_t const s0_in_copy = *s_in[0];
                reference_gate::complex_t const s1_in_copy = *s_in[1];
                *s_out[0] = cos_theta_2 * s0_in_copy - sin_theta_2 * exp_i_lambda * s1_in_copy;
                *s_out[1] = exp_i_phi * sin_theta_2 * s0_in_copy + exp_i_phi_plus_lambda * cos_theta_2 * s1_in_copy;
            }
        };

        struct u2
        {
            static constexpr unsigned int num_target_qubits = 1;
            reference_gate::complex_t exp_i_phi;
            reference_gate::complex_t exp_i_lambda;
            reference_gate::complex_t exp_i_phi_plus_lambda;
            u2(double phi, double lambda)
            {
                exp_i_phi = reference_gate::complex_t(cos(phi), sin(phi));
                exp_i_lambda = reference_gate::complex_t(cos(lambda), sin(lambda));
                reference_gate::float_t const phi_plus_lambda = phi + lambda;
                exp_i_phi_plus_lambda = reference_gate::complex_t(cos(phi_plus_lambda), sin(phi_plus_lambda));
            }
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                reference_gate::complex_t const s0_in_copy = *s_in[0];
                reference_gate::complex_t const s1_in_copy = *s_in[1];
                *s_out[0] = M_SQRT1_2 * (s0_in_copy - exp_i_lambda * s1_in_copy);
                *s_out[1] = M_SQRT1_2 * (exp_i_phi * s0_in_copy + exp_i_phi_plus_lambda * s1_in_copy);
            }
        };

        struct u1
        {
            static constexpr unsigned int num_target_qubits = 1;
            reference_gate::complex_t exp_i_lambda;
            u1(double lambda)
            {
                exp_i_lambda = reference_gate::complex_t(cos(lambda), sin(lambda));
            }
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                *s_out[0] = *s_in[0];
                *s_out[1] = exp_i_lambda * *s_in[1];
            }
        };

        typedef u3 u;
        typedef u1 p;

        struct global_phase
        {
            static constexpr unsigned int num_target_qubits = 0;
            reference_gate::complex_t exp_i_theta;
            global_phase(double theta)
            {
                exp_i_theta.real(cos(theta));
                exp_i_theta.imag(sin(theta));
            }
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                *s_out[0] = exp_i_theta * *s_in[0];
            }
        };

        struct hadamard
        {
            static constexpr unsigned int num_target_qubits = 1;
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s0_in_copy = *s_in[0];
                auto const s1_in_copy = *s_in[1];
                *s_out[0] = M_SQRT1_2 * (s0_in_copy + s1_in_copy);
                *s_out[1] = M_SQRT1_2 * (s0_in_copy - s1_in_copy);
            }
        };

        struct identity
        {
            static constexpr unsigned int num_target_qubits = 0;
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
            }
        };

        struct x
        {
            static constexpr unsigned int num_target_qubits = 1;
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s0_in_copy = *s_in[0];
                auto const s1_in_copy = *s_in[1];
                *s_out[0] = s1_in_copy;
                *s_out[1] = s0_in_copy;
            }
        };

        struct y
        {
            static constexpr unsigned int num_target_qubits = 1;
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s0_in_copy = *s_in[0];
                auto const s1_in_copy = *s_in[1];
                s_out[0]->real(s1_in_copy.imag());
                s_out[0]->imag(-s1_in_copy.real());
                s_out[1]->real(-s0_in_copy.imag());
                s_out[1]->imag(s0_in_copy.real());
            }
        };

        struct z
        {
            static constexpr unsigned int num_target_qubits = 1;
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                // auto const s0_in_copy = *s_in[0];
                auto const s1_in_copy = *s_in[1];
                // *s_out[0] = s0_in_copy;
                *s_out[1] = -s1_in_copy;
            }
        };

        struct s
        {
            static constexpr unsigned int num_target_qubits = 1;
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s1_in_copy = *s_in[1];
                s_out[1]->real(-s1_in_copy.imag());
                s_out[1]->imag(s1_in_copy.real());
            }
        };

        struct sdg
        {
            static constexpr unsigned int num_target_qubits = 1;
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s1_in_copy = *s_in[1];
                s_out[1]->real(s1_in_copy.imag());
                s_out[1]->imag(-s1_in_copy.real());
            }
        };

        struct t
        {
            static constexpr unsigned int num_target_qubits = 1;
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s1_in_copy = *s_in[1];
                *s_out[1] = reference_gate::complex_t(M_SQRT1_2, M_SQRT1_2) * s1_in_copy;
            }
        };

        struct tdg
        {
            static constexpr unsigned int num_target_qubits = 1;
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s1_in_copy = *s_in[1];
                *s_out[1] = reference_gate::complex_t(M_SQRT1_2, -M_SQRT1_2) * s1_in_copy;
            }
        };

        struct sx
        {
            static constexpr unsigned int num_target_qubits = 1;
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                // auto const s0_in_copy = *s_in[0];
                // auto const s1_in_copy = *s_in[1];
                // *s_out[0] = reference_gate::complex_t(0.5, 0.5) * s0_in_copy + reference_gate::complex_t(0.5, - 0.5) * s1_in_copy;
                // *s_out[1] = reference_gate::complex_t(0.5, - 0.5) * s0_in_copy + reference_gate::complex_t(0.5, 0.5) * s1_in_copy;

                auto const a = 0.5 * (*s_in[0] + *s_in[1]);
                auto const b = 0.5 * multiply_i(*s_in[0] - *s_in[1]);
                *s_out[0] = a + b;
                *s_out[1] = a - b;
            }
        };

        struct sxdg
        {
            static constexpr unsigned int num_target_qubits = 1;

             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const a = 0.5 * (*s_in[0] + *s_in[1]);
                auto const b = -0.5 * multiply_i(*s_in[0] - *s_in[1]);
                *s_out[0] = a + b;
                *s_out[1] = a - b;
            }
        };

        struct rx
        {
            static constexpr unsigned int num_target_qubits = 1;
            reference_gate::float_t cos_theta_2;
            reference_gate::float_t sin_theta_2;
            rx(double theta)
            {
                cos_theta_2 = cos(0.5 * theta);
                sin_theta_2 = sin(0.5 * theta);
            }
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s0_in_copy = *s_in[0];
                auto const s1_in_copy = *s_in[1];
                *s_out[0] = cos_theta_2 * s0_in_copy + multiply_i_real(s1_in_copy, -sin_theta_2);
                *s_out[1] = multiply_i_real(s0_in_copy, -sin_theta_2) + cos_theta_2 * s1_in_copy;
            }
        };

        struct ry
        {
            static constexpr unsigned int num_target_qubits = 1;
            reference_gate::float_t cos_theta_2;
            reference_gate::float_t sin_theta_2;
            ry(double theta)
            {
                cos_theta_2 = cos(0.5 * theta);
                sin_theta_2 = sin(0.5 * theta);
            }
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s0_in_copy = *s_in[0];
                auto const s1_in_copy = *s_in[1];
                *s_out[0] = cos_theta_2 * s0_in_copy - sin_theta_2 * s1_in_copy;
                *s_out[1] = sin_theta_2 * s0_in_copy + cos_theta_2 * s1_in_copy;
            }
        };

        struct rz
        {
            static constexpr unsigned int num_target_qubits = 1;
            reference_gate::float_t cos_theta_2;
            reference_gate::float_t sin_theta_2;
            rz(double theta)
            {
                cos_theta_2 = cos(0.5 * theta);
                sin_theta_2 = sin(0.5 * theta);
            }
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                *s_out[0] = reference_gate::complex_t(cos_theta_2, -sin_theta_2) * *s_in[0];
                *s_out[1] = reference_gate::complex_t(cos_theta_2, sin_theta_2) * *s_in[1];
            }
        };

        template <unsigned int num_swaps>
        struct swap
        {
            static_assert(num_swaps > 0 && num_swaps <= max_num_swaps,
                          "num_swaps must be between 1 and max_num_swaps");
            static constexpr unsigned int num_target_qubits = 2 * num_swaps;

             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                constexpr uint64_t num_basis_indices = UINT64_C(1) << num_target_qubits;
                constexpr uint64_t even_bits = UINT64_C(0x5555555555555555);
                // Exchange target bits 0/1, 2/3, ... in each basis index.
                #pragma unroll
                for (uint64_t i = 0; i < num_basis_indices; ++i)
                {
                    uint64_t const j = ((i & even_bits) << 1) | ((i >> 1) & even_bits);
                    // Each nontrivial cycle has two entries. Read both before
                    // writing so inputs and outputs may alias; skip fixed points.
                    if (i < j)
                    {
                        auto const si_in_copy = *s_in[i];
                        auto const sj_in_copy = *s_in[j];
                        *s_out[i] = sj_in_copy;
                        *s_out[j] = si_in_copy;
                    }
                }
            }
        };

        struct iswap
        {
            static constexpr unsigned int num_target_qubits = 2;
             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s01_in_copy = *s_in[1];
                auto const s10_in_copy = *s_in[2];
                // *s_out[0] = *s_in[0];
                s_out[1]->real(-s10_in_copy.imag());
                s_out[1]->imag(s10_in_copy.real());
                s_out[2]->real(-s01_in_copy.imag());
                s_out[2]->imag(s01_in_copy.real());
                // *s_out[3] = *s_in[3];
            }
        };

        struct id
        {
            static constexpr unsigned int num_target_qubits = 1;

             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                // *s_out[0] = *s_in[0];
                // *s_out[1] = *s_in[1];
            }
        };

        struct r
        {
            static constexpr unsigned int num_target_qubits = 1;

            reference_gate::float_t cos_theta_2;
            reference_gate::float_t sin_theta_2;
            reference_gate::complex_t exp_i_phi;
            reference_gate::complex_t exp_minus_i_phi;

            r(double theta, double phi)
            {
                reference_gate::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                sin_theta_2 = sin(theta_2);
                reference_gate::float_t cos_phi = cos(phi);
                reference_gate::float_t sin_phi = sin(phi);
                exp_i_phi = reference_gate::complex_t(cos_phi, sin_phi);
                exp_minus_i_phi = reference_gate::complex_t(cos_phi, -sin_phi);
            }

             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s0_in_copy = *s_in[0];
                auto const s1_in_copy = *s_in[1];
                *s_out[0] = cos_theta_2 * s0_in_copy + multiply_i_real(exp_minus_i_phi * s1_in_copy, -sin_theta_2);
                *s_out[1] = multiply_i_real(exp_i_phi * s0_in_copy, -sin_theta_2) + cos_theta_2 * s1_in_copy;
            }
        };

        struct rzz
        {
            static constexpr unsigned int num_target_qubits = 2;

            reference_gate::complex_t phase_minus;
            reference_gate::complex_t phase_plus;

            rzz(double theta)
            {
                reference_gate::float_t const theta_2 = 0.5 * theta;
                phase_minus = reference_gate::complex_t(cos(theta_2), -sin(theta_2));
                phase_plus = reference_gate::complex_t(cos(theta_2), sin(theta_2));
            }

             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                *s_out[0] = phase_minus * *s_in[0];
                *s_out[1] = phase_plus * *s_in[1];
                *s_out[2] = phase_plus * *s_in[2];
                *s_out[3] = phase_minus * *s_in[3];
            }
        };

        struct rxx
        {
            static constexpr unsigned int num_target_qubits = 2;

            reference_gate::float_t cos_theta_2;
            reference_gate::float_t m_sin_theta_2;

            rxx(double theta)
            {
                reference_gate::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                m_sin_theta_2 = -sin(theta_2);
            }

             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s00_in_copy = *s_in[0];
                auto const s01_in_copy = *s_in[1];
                auto const s10_in_copy = *s_in[2];
                auto const s11_in_copy = *s_in[3];

                *s_out[0] = cos_theta_2 * s00_in_copy + multiply_i_real(s11_in_copy, m_sin_theta_2);
                *s_out[1] = cos_theta_2 * s01_in_copy + multiply_i_real(s10_in_copy, m_sin_theta_2);
                *s_out[2] = cos_theta_2 * s10_in_copy + multiply_i_real(s01_in_copy, m_sin_theta_2);
                *s_out[3] = cos_theta_2 * s11_in_copy + multiply_i_real(s00_in_copy, m_sin_theta_2);
            }
        };

        struct ryy
        {
            static constexpr unsigned int num_target_qubits = 2;

            reference_gate::float_t cos_theta_2;
            reference_gate::float_t sin_theta_2;

            ryy(double theta)
            {
                reference_gate::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                sin_theta_2 = sin(theta_2);
            }

             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s00_in_copy = *s_in[0];
                auto const s01_in_copy = *s_in[1];
                auto const s10_in_copy = *s_in[2];
                auto const s11_in_copy = *s_in[3];

                *s_out[0] = cos_theta_2 * s00_in_copy + multiply_i_real(s11_in_copy, sin_theta_2);
                *s_out[1] = cos_theta_2 * s01_in_copy + multiply_i_real(s10_in_copy, -sin_theta_2);
                *s_out[2] = cos_theta_2 * s10_in_copy + multiply_i_real(s01_in_copy, -sin_theta_2);
                *s_out[3] = cos_theta_2 * s11_in_copy + multiply_i_real(s00_in_copy, sin_theta_2);
            }
        };

        struct rzx
        {
            static constexpr unsigned int num_target_qubits = 2;

            reference_gate::float_t cos_theta_2;
            reference_gate::float_t sin_theta_2;

            rzx(double theta)
            {
                reference_gate::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                sin_theta_2 = sin(theta_2);
            }

             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s00_in_copy = *s_in[0];
                auto const s01_in_copy = *s_in[1];
                auto const s10_in_copy = *s_in[2];
                auto const s11_in_copy = *s_in[3];

                *s_out[0] = cos_theta_2 * s00_in_copy + multiply_i_real(s10_in_copy, -sin_theta_2);
                *s_out[1] = cos_theta_2 * s01_in_copy + multiply_i(s11_in_copy) * sin_theta_2;
                *s_out[2] = cos_theta_2 * s10_in_copy + multiply_i_real(s00_in_copy, -sin_theta_2);
                *s_out[3] = cos_theta_2 * s11_in_copy + multiply_i(s01_in_copy) * sin_theta_2;
            }
        };

        struct xx_plus_yy
        {
            static constexpr unsigned int num_target_qubits = 2;

            reference_gate::float_t cos_theta_2;
            reference_gate::float_t sin_theta_2;
            reference_gate::complex_t exp_i_beta;
            reference_gate::complex_t exp_minus_i_beta;

            xx_plus_yy(double theta, double beta = 0.0)
            {
                reference_gate::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                sin_theta_2 = sin(theta_2);
                exp_i_beta = reference_gate::complex_t(cos(beta), sin(beta));
                exp_minus_i_beta = reference_gate::complex_t(cos(beta), -sin(beta));
            }

             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s01_in_copy = *s_in[1];
                auto const s10_in_copy = *s_in[2];

                // *s_out[0] = *s_in[0];
                *s_out[1] = cos_theta_2 * s01_in_copy + multiply_i_real(exp_minus_i_beta * s10_in_copy, -sin_theta_2);
                *s_out[2] = multiply_i_real(exp_i_beta * s01_in_copy, -sin_theta_2) + cos_theta_2 * s10_in_copy;
                // *s_out[3] = *s_in[3];
            }
        };

        struct xx_minus_yy
        {
            static constexpr unsigned int num_target_qubits = 2;

            reference_gate::float_t cos_theta_2;
            reference_gate::float_t sin_theta_2;
            reference_gate::complex_t exp_i_beta;
            reference_gate::complex_t exp_minus_i_beta;

            xx_minus_yy(double theta, double beta = 0.0)
            {
                reference_gate::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                sin_theta_2 = sin(theta_2);
                exp_i_beta = reference_gate::complex_t(cos(beta), sin(beta));
                exp_minus_i_beta = reference_gate::complex_t(cos(beta), -sin(beta));
            }

             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s00_in_copy = *s_in[0];
                auto const s11_in_copy = *s_in[3];

                *s_out[0] = cos_theta_2 * s00_in_copy + multiply_i_real(exp_minus_i_beta * s11_in_copy, -sin_theta_2);
                // *s_out[1] = *s_in[1];
                // *s_out[2] = *s_in[2];
                *s_out[3] = multiply_i_real(exp_i_beta * s00_in_copy, -sin_theta_2) + cos_theta_2 * s11_in_copy;
            }
        };

        struct dcx
        {
            static constexpr unsigned int num_target_qubits = 2;

             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s10_in_copy = *s_in[2];
                auto const s11_in_copy = *s_in[3];
                auto const s01_in_copy = *s_in[1];
                // *s_out[0] = *s_in[0];
                *s_out[1] = s11_in_copy;
                *s_out[2] = s01_in_copy;
                *s_out[3] = s10_in_copy;
            }
        };

        struct ecr
        {
            static constexpr unsigned int num_target_qubits = 2;

             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s00_in_copy = *s_in[0];
                auto const s01_in_copy = *s_in[1];
                auto const s10_in_copy = *s_in[2];
                auto const s11_in_copy = *s_in[3];
                *s_out[0] = M_SQRT1_2 * (s01_in_copy + multiply_i(s11_in_copy));
                *s_out[1] = M_SQRT1_2 * (s00_in_copy + multiply_i_m(s10_in_copy));
                *s_out[2] = M_SQRT1_2 * (multiply_i(s01_in_copy) + s11_in_copy);
                *s_out[3] = M_SQRT1_2 * (multiply_i_m(s00_in_copy) + s10_in_copy);
            }
        };

        struct rccx
        {
            static constexpr unsigned int num_target_qubits = 3;

             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s111_in_copy = *s_in[7];
                auto const s101_in_copy = *s_in[5];
                auto const s011_in_copy = *s_in[3];
                // *s_out[0] = *s_in[0];
                // *s_out[1] = *s_in[1];
                // *s_out[2] = *s_in[2];
                *s_out[3] = multiply_i_m(s111_in_copy);
                // *s_out[4] = *s_in[4];
                *s_out[5] = -s101_in_copy;
                // *s_out[6] = *s_in[6];
                *s_out[7] = multiply_i(s011_in_copy);
            }
        };

        struct rcccx
        {
            static constexpr unsigned int num_target_qubits = 4;

             void apply(reference_gate::complex_t const *const s_in[] [[maybe_unused]], reference_gate::complex_t *const s_out[] [[maybe_unused]]) const
            {
                auto const s0011_in_copy = *s_in[3];
                auto const s1111_in_copy = *s_in[15];
                auto const s1011_in_copy = *s_in[11];
                auto const s0111_in_copy = *s_in[7];
                // *s_out[0] = *s_in[0];
                // *s_out[1] = *s_in[1];
                // *s_out[2] = *s_in[2];
                *s_out[3] = multiply_i(s0011_in_copy);
                // *s_out[4] = *s_in[4];
                // *s_out[5] = *s_in[5];
                // *s_out[6] = *s_in[6];
                *s_out[7] = s1111_in_copy;
                // *s_out[8] = *s_in[8];
                // *s_out[9] = *s_in[9];
                // *s_out[10] = *s_in[10];
                *s_out[11] = multiply_i_m(s1011_in_copy);
                // *s_out[12] = *s_in[12];
                // *s_out[13] = *s_in[13];
                // *s_out[14] = *s_in[14];
                *s_out[15] = -s0111_in_copy;
            }
        };

}
}
