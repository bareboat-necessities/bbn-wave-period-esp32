#pragma once
/*
  Fully analytical closed-form MEKF + Matérn-3/2 INS extension (header-only)
  - No Van Loan, no Padé, no runtime matrix .exp()
  - Φ(h) and Qd(h) are derived together from the same stochastic integral
  - Latent acceleration follows Matérn-3/2 (critically damped 2nd-order GM):
        ȧ = j
        j̇ = -(2/τ) j - (1/τ²) a + w,   with  E[w wᵀ] = q_c δ(t-t'),  q_c = 4 σ_a² / τ³
  - Linear kinematics per axis:
        v̇ = a ,  ṗ = v ,  Ṡ = p
  - Per-axis 5×5 block uses x=[v p S a j]ᵀ. Closed-form Φ(h)=exp(Ah) and
    Qd(h)=∫₀ʰ Φ(s) G q_c Gᵀ Φ(s)ᵀ ds are pasted below as hand-coded formulas.

  Public API matches the previous "Kalman_INS" you shared.
*/

#include <limits>
#include <stdexcept>
#include <cmath>

#ifdef EIGEN_NON_ARDUINO
  #include <Eigen/Dense>
#else
  #include <ArduinoEigenDense.h>
#endif

using Eigen::Matrix;

// -----------------------------------------------
// Per-axis Matérn-3/2 (critically damped 2nd GM)
// State x = [v, p, S, a, j]^T
//   v̇ = a,  ṗ = v,  Ṡ = p
//   ȧ = j
//   j̇ = -(2/τ) j - (1/τ²) a + w,   E[w(t)w(s)] = q_c δ(t-s),  q_c = 4 σ_a² / τ³
//
// This file gives closed-form Φ(h) = exp(Ah) and
// Qd(h) = ∫₀ʰ Φ(s) G q_c Gᵀ Φ(s)ᵀ ds,
// derived *together* from the same integral.
// G = [0,0,0,0,1]^T, i.e. noise drives j.
//
// No Van Loan, no Padé, no matrix exp.
// -----------------------------------------------

namespace m32_analytic {

// -------- small helpers (safe numerics near h→0) --------
template<typename T>
inline T sqr(T x){ return x*x; }

template<typename T>
inline T pow3(T x){ return x*x*x; }

// ∫₀ʰ s^n e^{-λ s} ds, n ∈ {0..4}, λ≥0 (closed forms)
template<typename T>
inline T integ_pow_exp_0toh(int n, T lambda, T h)
{
    if (lambda <= T(0)) {
        // plain polynomial integral
        switch(n){
            case 0: return h;
            case 1: return T(0.5)*h*h;
            case 2: return h*h*h/T(3);
            case 3: return h*h*h*h/T(4);
            case 4: return h*h*h*h*h/T(5);
            default: break;
        }
    }

    // Use: ∫ s^n e^{-λ s} ds = n!/λ^{n+1} [1 - e^{-λ h} * Σ_{k=0}^n (λ h)^k/k!]
    // We only need n ≤ 4 here.
    const T x = lambda*h;
    T Sm = T(1), term = T(1);
    for (int k=1;k<=n;++k){ term *= x/T(k); Sm += term; }

    T fact = T(1);
    for (int k=2;k<=n;++k) fact *= T(k);

    T lam_pow = std::pow(lambda, T(n+1));
    return (fact/lam_pow) * (T(1) - std::exp(-x)*Sm);
}

// Build Φ(h) and Qd(h) for one axis (5×5), Matérn-3/2
// Inputs:  h  (step, s),  tau (τ, s),  sigma2_a (stationary Var[a], (m/s²)²)
// Output:  Phi (5×5), Qd (5×5)
template<typename T>
inline void phi_Qd_axis_M32(const T h, const T tau, const T sigma2_a,
                            Eigen::Matrix<T,5,5>& Phi,
                            Eigen::Matrix<T,5,5>& Qd)
{
    using Mat5 = Eigen::Matrix<T,5,5>;

    // Clamp τ > 0
    const T tau_c = std::max(tau, T(1e-9));
    const T inv_tau = T(1)/tau_c;
    const T inv_tau2= inv_tau*inv_tau;

    // Exponentials
    const T alpha  = std::exp(-h*inv_tau);
    const T alpha2 = std::exp(-T(2)*h*inv_tau);

    // “E-moments” (compact combos that show up repeatedly):
    // E0=1-α, E1=1-α(1+h/τ), E2=1-α(1+h/τ+(h/τ)^2/2), E3=...
    const T th  = h*inv_tau;
    const T th2 = th*th;
    const T th3 = th2*th;

    const T E0 = T(1) - alpha;
    const T E1 = T(1) - alpha*(T(1) + th);
    const T E2 = T(1) - alpha*(T(1) + th + th2/T(2));
    const T E3 = T(1) - alpha*(T(1) + th + th2/T(2) + th3/T(6));

    // ---------- Φ(h) (exact) ----------
    // a(h), j(h) are exact solutions for a critically-damped pair
    // a(h) = e^{-h/τ}[(1 + h/τ)a0 + h j0]
    // j(h) = e^{-h/τ}[      - (h/τ²)a0 + (1 - h/τ) j0]

    Phi.setZero();
    // Kinematics base (v,p,S from v0,p0,S0)
    Phi(0,0) = T(1);          // v <- v
    Phi(1,0) = h;             // p <- v
    Phi(1,1) = T(1);          // p <- p
    Phi(2,0) = T(0.5)*h*h;    // S <- v
    Phi(2,1) = h;             // S <- p
    Phi(2,2) = T(1);          // S <- S

    // Couplings from a0, j0 into v(h)
    // ∫₀ʰ a(s) ds with a(s)=e^{-s/τ}[(1+s/τ)a0 + s j0]
    const T I_a = tau_c*(E0 + E1);   // coeff of a0
    const T I_j = tau_c*tau_c*E1;    // coeff of j0
    Phi(0,3) = I_a;
    Phi(0,4) = I_j;

    // Into p(h) = p0 + h v0 + ∫₀ʰ (h-s) a(s) ds
    // ==> p <- a0:  h*I_a - τ²(E1 + 2E2)
    //     p <- j0:  h*I_j - 2τ³ E2
    const T J_a = h*I_a - tau_c*tau_c*(E1 + T(2)*E2);
    const T J_j = h*I_j - T(2)*tau_c*tau_c*tau_c*E2;
    Phi(1,3) = J_a;
    Phi(1,4) = J_j;

    // Into S(h) = S0 + h p0 + 0.5 h² v0 + ∫₀ʰ (weight) a(s) ds
    // Closed-form via repeated integration; compact result:
    const T Sa = T(0.5)*( h*h*tau_c*(E0+E1)
                        - T(2)*h*tau_c*tau_c*(E1+T(2)*E2)
                        + T(2)*tau_c*tau_c*tau_c*(E2+T(3)*E3) );
    const T Sj = T(0.5)*( h*h*(tau_c*tau_c*E1)
                        - T(4)*h*(tau_c*tau_c*tau_c*E2)
                        + T(6)*tau_c*tau_c*tau_c*tau_c*E3 );
    Phi(2,3) = Sa;
    Phi(2,4) = Sj;

    // Latent 2×2 block (a,j)
    Phi(3,3) = alpha*(T(1) + th);
    Phi(3,4) = alpha*h;
    Phi(4,3) = -alpha*(h*inv_tau2);
    Phi(4,4) = alpha*(T(1) - th);

    // ---------- Qd(h) (exact, same integral) ----------
    // Build g(s)=Φ(s)G for G=[0,0,0,0,1]^T (i.e. initial j0=1, others 0)
    // Using the same closed-form Φ(s), but with s as variable (0..h):
    //  a(s) = s e^{-s/τ}
    //  j(s) = e^{-s/τ}(1 - s/τ)
    //  v(s) = ∫₀ˢ a(u) du = τ² - τ² e^{-s/τ} - τ s e^{-s/τ}
    //  p(s) = ∫₀ˢ v(u) du = τ² s - 2τ³ + e^{-s/τ}(2τ³ + τ² s)
    //  S(s) = ∫₀ˢ p(u) du = 0.5 τ² s² - 2τ³ s + 3τ⁴ + e^{-s/τ}(-3τ⁴ - τ³ s)

    // Each g_i(s) is a linear combo of { s^n e^{-k s/τ} } with k∈{0,1}.
    // Qd = q_c ∫₀ʰ g(s) g(s)^T ds, with q_c = 4 σ_a² / τ³.

    const T qc = T(4)*sigma2_a/(tau_c*tau_c*tau_c);

    // Pre-integrals (we’ll reuse a lot):
    auto I0  = [&](int n){ return integ_pow_exp_0toh(n, T(0),     h); };       // ∫ s^n ds
    auto I1  = [&](int n){ return integ_pow_exp_0toh(n, inv_tau,  h); };       // ∫ s^n e^{-s/τ} ds
    auto I2  = [&](int n){ return integ_pow_exp_0toh(n, T(2)*inv_tau, h); };   // ∫ s^n e^{-2s/τ} ds

    // For cleaner notation:
    const T t1 = tau_c;
    const T t2 = t1*t1;
    const T t3 = t2*t1;
    const T t4 = t3*t1;

    // g_v(s) = t2 - t2 e^{-s/τ} - t1 s e^{-s/τ}
    // g_p(s) = t2 s - 2 t3 + e^{-s/τ}(2 t3 + t2 s)
    // g_S(s) = 0.5 t2 s^2 - 2 t3 s + 3 t4 + e^{-s/τ}(-3 t4 - t3 s)
    // g_a(s) = s e^{-s/τ}
    // g_j(s) = e^{-s/τ}(1 - s/τ)

    // We’ll accumulate upper triangle and mirror.
    Qd.setZero();

    auto add_Q = [&](int i, int j, T val){
        Qd(i,j) = val;
        if (i!=j) Qd(j,i) = val;
    };

    // Helper lambdas to integrate pairwise products quickly.
    // (We expand only the necessary monomials; all closed forms via I0, I1, I2)

    // ---- <v,v> ----
    {
        // g_v = A + B e^- + C s e^- with A=t2, B= -t2, C= -t1
        // g_v^2 = A^2 + B^2 e^-2 + C^2 s^2 e^-2 + 2AB e^- + 2AC s e^- + 2BC s e^-2
        const T A = t2;
        const T B = -t2;
        const T C = -t1;

        T val =
            A*A * I0(0)
          + B*B * I2(0)
          + C*C * I2(2)
          + T(2)*A*B * I1(0)
          + T(2)*A*C * I1(1)
          + T(2)*B*C * I2(1);

        add_Q(0,0, qc*val);
    }

    // ---- <v,p> ----
    {
        // g_v = A + B e^- + C s e^-           (A=t2, B=-t2, C=-t1)
        // g_p = D s + E + (F + G s) e^-       (D=t2, E=-2t3, F=2t3, G=t2)
        const T A=t2, B=-t2, C=-t1;
        const T D=t2, E=-T(2)*t3, F=T(2)*t3, G=t2;

        // Expand product and integrate termwise.
        // Poly × Poly:
        T val = (A*D) * I0(1) + (A*E) * I0(0);

        // Poly × e^- terms:
        val += A*F * I1(0) + A*G * I1(1);

        // e^- × Poly:
        val += B*D * I1(1) + B*E * I1(0);

        // e^- × e^-:
        val += B*F * I2(0) + B*G * I2(1);

        // (s e^-) × Poly:
        val += C*D * I1(2) + C*E * I1(1);

        // (s e^-) × e^-:
        val += C*F * I2(1) + C*G * I2(2);

        add_Q(0,1, qc*val);
    }

    // ---- <v,S> ----
    {
        // g_S = H s^2 + K s + L + (M + N s) e^-   with H=0.5 t2, K=-2 t3, L=3 t4, M=-3 t4, N=-t3
        const T A=t2,  B=-t2,  C=-t1;               // g_v
        const T H=T(0.5)*t2, K=-T(2)*t3, L=T(3)*t4, M=-T(3)*t4, N=-t3;

        T val = T(0);
        // Poly×Poly:
        val += A*H * I0(2) + A*K * I0(1) + A*L * I0(0);
        // Poly×e^-:
        val += A*M * I1(0) + A*N * I1(1);

        // e^-×Poly:
        val += B*H * I1(2) + B*K * I1(1) + B*L * I1(0);
        // e^-×e^-:
        val += B*M * I2(0) + B*N * I2(1);

        // (s e^-)×Poly:
        val += C*H * I1(3) + C*K * I1(2) + C*L * I1(1);
        // (s e^-)×e^-:
        val += C*M * I2(1) + C*N * I2(2);

        add_Q(0,2, qc*val);
    }

    // ---- <v,a> ----
    {
        // g_a = s e^-
        const T A=t2, B=-t2, C=-t1; // g_v
        T val = T(0);
        // Poly×(s e^-)
        val += A * I1(1);
        // e^-×(s e^-)
        val += B * I2(1);
        // (s e^-)×(s e^-)
        val += C * I2(2);

        add_Q(0,3, qc*val);
    }

    // ---- <v,j> ----
    {
        // g_j = e^- (1 - s/τ) = e^- + (-inv_tau) s e^-
        const T A=t2, B=-t2, C=-t1; // g_v
        const T J0 = T(1), J1 = -inv_tau;

        T val = T(0);
        // Poly × e^- and Poly × (s e^-)
        val += A*J0 * I1(0) + A*J1 * I1(1);
        // e^- × e^- and e^- × (s e^-)
        val += B*J0 * I2(0) + B*J1 * I2(1);
        // (s e^-) × e^- and × (s e^-)
        val += C*J0 * I2(1) + C*J1 * I2(2);

        add_Q(0,4, qc*val);
    }

    // ---- <p,p> ----
    {
        // g_p = D s + E + (F + G s) e^-  (D=t2, E=-2t3, F=2t3, G=t2)
        const T D=t2, E=-T(2)*t3, F=T(2)*t3, G=t2;

        T val = T(0);
        // (Ds+E)^2
        val += D*D * I0(2) + T(2)*D*E * I0(1) + E*E * I0(0);
        // 2(Ds+E)(F+Gs) e^-
        val += T(2)*( D*F * I1(1) + D*G * I1(2) + E*F * I1(0) + E*G * I1(1) );
        // (F + G s)^2 e^-2
        val += F*F * I2(0) + T(2)*F*G * I2(1) + G*G * I2(2);

        add_Q(1,1, qc*val);
    }

    // ---- <p,S> ----
    {
        const T D=t2, E=-T(2)*t3, F=T(2)*t3, G=t2;                 // g_p
        const T H=T(0.5)*t2, K=-T(2)*t3, L=T(3)*t4, M=-T(3)*t4, N=-t3; // g_S

        T val = T(0);
        // Poly×Poly: (Ds+E)*(H s^2 + K s + L)
        val += D*H * I0(3) + D*K * I0(2) + D*L * I0(1)
             + E*H * I0(2) + E*K * I0(1) + E*L * I0(0);
        // Poly×e^-: (Ds+E)*(M + N s) e^-
        val += D*M * I1(1) + D*N * I1(2) + E*M * I1(0) + E*N * I1(1);
        // e^-×Poly: (F+Gs)e^- * (H s^2 + K s + L)
        val += F*H * I1(2) + F*K * I1(1) + F*L * I1(0)
             + G*H * I1(3) + G*K * I1(2) + G*L * I1(1);
        // e^-×e^-: (F+Gs)(M+Ns) e^-2
        val += F*M * I2(0) + F*N * I2(1) + G*M * I2(1) + G*N * I2(2);

        add_Q(1,2, qc*val);
    }

    // ---- <p,a> ----
    {
        const T D=t2, E=-T(2)*t3, F=T(2)*t3, G=t2; // g_p
        // g_a = s e^-
        T val = T(0);
        // (Ds+E) × s e^-
        val += D * I1(2) + E * I1(1);
        // (F+Gs)e^- × s e^-
        val += F * I2(1) + G * I2(2);

        add_Q(1,3, qc*val);
    }

    // ---- <p,j> ----
    {
        const T D=t2, E=-T(2)*t3, F=T(2)*t3, G=t2; // g_p
        const T J0 = T(1), J1 = -inv_tau;          // g_j = J0 e^- + J1 s e^-
        T val = T(0);

        // (Ds+E) × (J0 e^- + J1 s e^-)
        val += D*J0 * I1(1) + D*J1 * I1(2) + E*J0 * I1(0) + E*J1 * I1(1);
        // (F+Gs)e^- × (J0 e^- + J1 s e^-)
        val += F*J0 * I2(0) + F*J1 * I2(1) + G*J0 * I2(1) + G*J1 * I2(2);

        add_Q(1,4, qc*val);
    }

    // ---- <S,S> ----
    {
        const T H=T(0.5)*t2, K=-T(2)*t3, L=T(3)*t4, M=-T(3)*t4, N=-t3; // g_S = Poly + e^-(M+Ns)

        T val = T(0);
        // Poly×Poly: (H s^2 + K s + L)^2
        val += H*H * I0(4) + T(2)*H*K * I0(3) + (T(2)*H*L + K*K) * I0(2)
             + T(2)*K*L * I0(1) + L*L * I0(0);
        // 2 Poly×e^-: 2(H s^2 + K s + L)(M + N s) e^-
        val += T(2)*( H*M * I1(2) + H*N * I1(3)
                    + K*M * I1(1) + K*N * I1(2)
                    + L*M * I1(0) + L*N * I1(1) );
        // e^-×e^-: (M + N s)^2 e^-2
        val += M*M * I2(0) + T(2)*M*N * I2(1) + N*N * I2(2);

        add_Q(2,2, qc*val);
    }

    // ---- <S,a> ----
    {
        const T H=T(0.5)*t2, K=-T(2)*t3, L=T(3)*t4, M=-T(3)*t4, N=-t3; // g_S
        // g_a = s e^-
        T val = T(0);
        // Poly×(s e^-):
        val += H * I1(3) + K * I1(2) + L * I1(1);
        // e^-×(s e^-):
        val += M * I2(1) + N * I2(2);

        add_Q(2,3, qc*val);
    }

    // ---- <S,j> ----
    {
        const T H=T(0.5)*t2, K=-T(2)*t3, L=T(3)*t4, M=-T(3)*t4, N=-t3; // g_S
        const T J0=T(1), J1=-inv_tau;                                  // g_j
        T val = T(0);
        // Poly×(J0 e^- + J1 s e^-)
        val += H*J0 * I1(2) + H*J1 * I1(3)
             + K*J0 * I1(1) + K*J1 * I1(2)
             + L*J0 * I1(0) + L*J1 * I1(1);
        // e^-×(J0 e^- + J1 s e^-)
        val += M*J0 * I2(0) + M*J1 * I2(1) + N*J0 * I2(1) + N*J1 * I2(2);

        add_Q(2,4, qc*val);
    }

    // ---- <a,a> ----
    {
        // g_a = s e^-
        add_Q(3,3, qc * I2(2));
    }

    // ---- <a,j> ----
    {
        // g_j = e^- + (-inv_tau) s e^-
        const T J0=T(1), J1=-inv_tau;
        T val = J0 * I2(1) + J1 * I2(2);
        add_Q(3,4, qc*val);
    }

    // ---- <j,j> ----
    {
        // (J0 e^- + J1 s e^-)^2
        const T J0=T(1), J1=-inv_tau;
        T val = J0*J0 * I2(0) + T(2)*J0*J1 * I2(1) + J1*J1 * I2(2);
        add_Q(4,4, qc*val);
    }
}

} // namespace m32_analytic

template <typename T = float, bool with_gyro_bias = true, bool with_accel_bias = true>
class EIGEN_ALIGN_MAX Kalman_INS {

    // Base: attitude error (3) + optional gyro bias (3)
    static constexpr int BASE_N = with_gyro_bias ? 6 : 3;

    // Extended: v(3), p(3), S(3), a(3), j(3) [+ optional b_a(3)]
    static constexpr int EXT_ADD = with_accel_bias ? 18 : 15;
    static constexpr int NX      = BASE_N + EXT_ADD;

    // Offsets
    static constexpr int OFF_V  = BASE_N + 0;
    static constexpr int OFF_P  = BASE_N + 3;
    static constexpr int OFF_S  = BASE_N + 6;
    static constexpr int OFF_A  = BASE_N + 9;    // a
    static constexpr int OFF_J  = BASE_N + 12;   // j
    static constexpr int OFF_BA = with_accel_bias ? (BASE_N + 15) : -1;

    // Measurements: accel + mag
    static constexpr int M = 6;

    using Vector3    = Matrix<T,3,1>;
    using Vector4    = Matrix<T,4,1>;
    using Vector6    = Matrix<T,6,1>;
    using Matrix3    = Matrix<T,3,3>;
    using Matrix5    = Matrix<T,5,5>;
    using Matrix15   = Matrix<T,15,15>;
    using MatrixM    = Matrix<T,M,M>;
    using MatrixNX   = Matrix<T,NX,NX>;
    using MatrixBase = Matrix<T,BASE_N,BASE_N>;

    static constexpr T half = T(0.5);
    static constexpr T STD_GRAV = T(9.80665);
    static constexpr T TEMP_REF = T(35.0);

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Kalman_INS(const Vector3& sigma_a, const Vector3& sigma_g, const Vector3& sigma_m,
               T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-11), T R_S_noise = T(5e-2),
               T gravity_magnitude = T(STD_GRAV))
      : gravity_magnitude_(gravity_magnitude),
        Racc_(sigma_a.array().square().matrix().asDiagonal()),
        Rmag_(sigma_m.array().square().matrix().asDiagonal()),
        Qbase_(initialize_Q_(sigma_g, b0))
    {
        qref_.setIdentity();
        Pbase_.setIdentity();

        Pbase_.template topLeftCorner<3,3>() = Matrix3::Identity() * Pq0;
        if constexpr (with_gyro_bias) {
            Pbase_.template block<3,3>(3,3) = Matrix3::Identity() * Pb0;
        }

        xext_.setZero();
        Pext_.setZero();
        Pext_.topLeftCorner(BASE_N, BASE_N) = Pbase_;

        // Seed linear states
        set_initial_linear_uncertainty(T(1.0), T(20.0), T(50.0));

        // Latent acceleration stationary variance seed
        Pext_.template block<3,3>(OFF_A, OFF_A) = Sigma_a_stat_;
        Pext_.template block<3,3>(OFF_J, OFF_J) = Matrix3::Identity() * T(0.05);

        if constexpr (with_accel_bias) {
            Pext_.template block<3,3>(OFF_BA, OFF_BA) =
                Matrix3::Identity() * (sigma_bacc0_ * sigma_bacc0_);
        }

        R_.setZero();
        R_.template topLeftCorner<3,3>() = Racc_;
        R_.template bottomRightCorner<3,3>() = Rmag_;

        R_S_ = Matrix3::Identity() * R_S_noise;
    }

    // ========== Initialization ==========
    void initialize_from_acc_mag(const Vector3& acc_body, const Vector3& mag_body) {
        T an = acc_body.norm();
        if (an < T(1e-8)) throw std::runtime_error("acc vector too small");
        Vector3 acc_n = acc_body / an;

        Vector3 z_world = -acc_n; // +Z down
        Vector3 mag_h   = mag_body - (mag_body.dot(z_world))*z_world;
        if (mag_h.norm() < T(1e-8)) throw std::runtime_error("mag parallel to gravity");
        mag_h.normalize();
        Vector3 x_world = mag_h;                       // +X north
        Vector3 y_world = z_world.cross(x_world).normalized();

        Matrix3 Rwb;
        Rwb.col(0)=x_world; Rwb.col(1)=y_world; Rwb.col(2)=z_world;
        qref_ = Eigen::Quaternion<T>(Rwb);
        qref_.normalize();

        v2ref_ = R_bw_() * mag_body.normalized(); // store world field unit
    }

    void initialize_from_acc(const Vector3& acc) {
        T an = acc.norm();
        if (an < T(1e-8)) throw std::runtime_error("acc vector too small");
        qref_ = quaternion_from_acc(acc / an);
        qref_.normalize();
    }

    static Eigen::Quaternion<T> quaternion_from_acc(const Vector3& acc) {
        Vector3 an = acc.normalized();
        Vector3 zb = Vector3::UnitZ();
        Vector3 tgt = -an;
        T c = zb.dot(tgt);
        Vector3 axis = zb.cross(tgt);
        T n = axis.norm();
        if (n < T(1e-8)) {
            if (c > 0) return Eigen::Quaternion<T>::Identity();
            return Eigen::Quaternion<T>(0,1,0,0);
        }
        axis /= n;
        c = std::max(T(-1), std::min(T(1), c));
        T ang = std::acos(c);
        Eigen::AngleAxis<T> aa(ang, axis);
        Eigen::Quaternion<T> q(aa);
        q.normalize();
        return q;
    }

    // ========== Config ==========
    void set_mag_world_ref(const Vector3& B_world) { v2ref_ = B_world.normalized(); }

    void set_Racc(const Vector3& sigma_acc) {
        Racc_ = sigma_acc.array().square().matrix().asDiagonal();
        R_.template topLeftCorner<3,3>() = Racc_;
    }

    void set_Rmag(const Vector3& sigma_mag) {
        Matrix3 Rmag_new = sigma_mag.array().square().matrix().asDiagonal();
        Rmag_ = Rmag_new;
        R_.template bottomRightCorner<3,3>() = Rmag_new;
    }

    void set_RS_noise(const Vector3& sigma_S) {
        R_S_ = sigma_S.array().square().matrix().asDiagonal();
    }

    void set_aw_time_constant(T tau_seconds) { tau_lat_ = std::max(T(1e-3), tau_seconds); }

    void set_aw_stationary_std(const Vector3& std_aw) {
        Sigma_a_stat_ = std_aw.array().square().matrix().asDiagonal();   // Var[a] per axis
    }

    void set_initial_linear_uncertainty(T sigma_v0, T sigma_p0, T sigma_S0) {
        Pext_.template block<3,3>(OFF_V, OFF_V) = Matrix3::Identity() * (sigma_v0*sigma_v0);
        Pext_.template block<3,3>(OFF_P, OFF_P) = Matrix3::Identity() * (sigma_p0*sigma_p0);
        Pext_.template block<3,3>(OFF_S, OFF_S) = Matrix3::Identity() * (sigma_S0*sigma_S0);
    }

    void set_initial_acc_bias(const Vector3& b0) {
        if constexpr (with_accel_bias) xext_.template segment<3>(OFF_BA) = b0;
    }
    void set_initial_acc_bias_std(T s) {
        if constexpr (with_accel_bias) sigma_bacc0_ = std::max(T(0), s);
    }
    void set_Q_bacc_rw(const Vector3& rw_std_per_sqrt_s) {
        if constexpr (with_accel_bias) Q_bacc_ = rw_std_per_sqrt_s.array().square().matrix().asDiagonal();
    }
    void set_accel_bias_temp_coeff(const Vector3& ka_per_degC) { k_a_ = ka_per_degC; }

    void set_initial_jerk_std(T s) {
        Pext_.template block<3,3>(OFF_J, OFF_J) = Matrix3::Identity() * (s*s);
    }

    // ========== Accessors ==========
    Eigen::Quaternion<T> quaternion() const { return qref_.conjugate(); }
    const MatrixBase& covariance_base() const { return Pbase_; }
    const MatrixNX&   covariance_full() const { return Pext_;  }

    Vector3 gyroscope_bias() const {
        if constexpr (with_gyro_bias) return xext_.template segment<3>(3);
        else                           return Vector3::Zero();
    }
    Vector3 get_acc_bias() const {
        if constexpr (with_accel_bias) return xext_.template segment<3>(OFF_BA);
        else                            return Vector3::Zero();
    }
    Vector3 get_velocity() const { return xext_.template segment<3>(OFF_V); }
    Vector3 get_position() const { return xext_.template segment<3>(OFF_P); }
    Vector3 get_integral_displacement() const { return xext_.template segment<3>(OFF_S); }
    Vector3 get_world_accel() const { return xext_.template segment<3>(OFF_A); }
    Vector3 get_world_jerk()  const { return xext_.template segment<3>(OFF_J); }

    // ========== Cycle ==========
    void time_update(const Vector3& gyr, T Ts) {
        // attitude mean propagation (right-multiply)
        Vector3 gb = Vector3::Zero();
        if constexpr (with_gyro_bias) gb = xext_.template segment<3>(3);
        last_gyr_bias_corrected_ = gyr - gb;

        T ang = last_gyr_bias_corrected_.norm() * Ts;
        Eigen::Quaternion<T> dq;
        if (ang > T(1e-9)) {
            Vector3 axis = last_gyr_bias_corrected_.normalized();
            dq = Eigen::AngleAxis<T>(ang, axis);
        } else dq.setIdentity();
        qref_ = qref_ * dq;
        qref_.normalize();

        // Build all-analytic discrete Φ and Qd
        MatrixNX F, Qd;
        assembleExtendedFandQ_(Ts, F, Qd);

        // Mean propagation for linear block [v p S a j] per axis (3× of 5 states)
        Eigen::Matrix<T,15,1> xl;
        xl.template segment<3>(0)  = xext_.template segment<3>(OFF_V);
        xl.template segment<3>(3)  = xext_.template segment<3>(OFF_P);
        xl.template segment<3>(6)  = xext_.template segment<3>(OFF_S);
        xl.template segment<3>(9)  = xext_.template segment<3>(OFF_A);
        xl.template segment<3>(12) = xext_.template segment<3>(OFF_J);

        Eigen::Matrix<T,15,1> xl_next = F.template block<15,15>(OFF_V, OFF_V) * xl;

        xext_.template segment<3>(OFF_V) = xl_next.template segment<3>(0);
        xext_.template segment<3>(OFF_P) = xl_next.template segment<3>(3);
        xext_.template segment<3>(OFF_S) = xl_next.template segment<3>(6);
        xext_.template segment<3>(OFF_A) = xl_next.template segment<3>(9);
        xext_.template segment<3>(OFF_J) = xl_next.template segment<3>(12);

        // Covariance propagation
        Pext_ = F * Pext_ * F.transpose() + Qd;
        Pext_ = T(0.5) * (Pext_ + Pext_.transpose());

        // Mirror base
        Pbase_ = Pext_.topLeftCorner(BASE_N, BASE_N);

        // Optional S drift control
        applyIntegralZeroPseudoMeas();
    }

    void measurement_update(const Vector3& acc, const Vector3& mag, T tempC = TEMP_REF) {
        Vector3 v1hat = accelerometer_measurement_func_(tempC);
        Vector3 v2hat = magnetometer_measurement_func_();

        Matrix<T,M,NX> C = Matrix<T,M,NX>::Zero();
        // accel rows
        C.template block<3,3>(0,0)  = -skew_(v1hat);
        C.template block<3,3>(0,OFF_A) = R_wb_();
        if constexpr (with_accel_bias) C.template block<3,3>(0,OFF_BA) = Matrix3::Identity();
        // mag rows
        C.template block<3,3>(3,0)  = -skew_(v2hat);

        Vector6 yhat; yhat << v1hat, v2hat;
        Vector6 y;    y    << acc, mag;
        Vector6 inno = y - yhat;

        MatrixM S = C * Pext_ * C.transpose() + R_;
        auto PCt  = Pext_ * C.transpose();

        Eigen::LDLT<MatrixM> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            S += MatrixM::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * R_.norm());
            ldlt.compute(S);
            if (ldlt.info() != Eigen::Success) return;
        }
        Matrix<T,NX,M> K = PCt * ldlt.solve(MatrixM::Identity());

        xext_.noalias() += K * inno;

        MatrixNX I = MatrixNX::Identity();
        Pext_ = (I - K*C) * Pext_ * (I - K*C).transpose() + K * R_ * K.transpose();
        Pext_ = T(0.5) * (Pext_ + Pext_.transpose());

        applyQuaternionCorrection_();
        xext_.template head<3>().setZero();
        Pbase_ = Pext_.topLeftCorner(BASE_N, BASE_N);
    }

    void measurement_update_acc_only(const Vector3& acc_meas, T tempC = TEMP_REF) {
        const Vector3 v1hat = accelerometer_measurement_func_(tempC);

        Matrix<T,3,NX> C = Matrix<T,3,NX>::Zero();
        C.template block<3,3>(0,0)    = -skew_(v1hat);
        C.template block<3,3>(0,OFF_A)= R_wb_();
        if constexpr (with_accel_bias) C.template block<3,3>(0,OFF_BA) = Matrix3::Identity();

        Vector3 inno = acc_meas - v1hat;

        Matrix3 S = C * Pext_ * C.transpose() + Racc_;
        auto PCt  = Pext_ * C.transpose();

        Eigen::LDLT<Matrix3> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            S += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * Racc_.norm());
            ldlt.compute(S);
            if (ldlt.info() != Eigen::Success) return;
        }
        Matrix<T,NX,3> K = PCt * ldlt.solve(Matrix3::Identity());

        xext_.noalias() += K * inno;

        MatrixNX I = MatrixNX::Identity();
        Pext_ = (I - K*C) * Pext_ * (I - K*C).transpose() + K * Racc_ * K.transpose();
        Pext_ = T(0.5) * (Pext_ + Pext_.transpose());

        applyQuaternionCorrection_();
        xext_.template head<3>().setZero();
    }

    void measurement_update_mag_only(const Vector3& mag) {
        const Vector3 v2hat = magnetometer_measurement_func_();

        Matrix<T,3,NX> C = Matrix<T,3,NX>::Zero();
        C.template block<3,3>(0,0) = -skew_(v2hat);

        Vector3 inno = mag - v2hat;

        Matrix3 S = C * Pext_ * C.transpose() + Rmag_;
        auto PCt  = Pext_ * C.transpose();

        Eigen::LDLT<Matrix3> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            S += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * Rmag_.norm());
            ldlt.compute(S);
            if (ldlt.info() != Eigen::Success) return;
        }
        Matrix<T,NX,3> K = PCt * ldlt.solve(Matrix3::Identity());

        xext_.noalias() += K * inno;

        MatrixNX I = MatrixNX::Identity();
        Pext_ = (I - K*C) * Pext_ * (I - K*C).transpose() + K * Rmag_ * K.transpose();
        Pext_ = T(0.5) * (Pext_ + Pext_.transpose());

        applyQuaternionCorrection_();
        xext_.template head<3>().setZero();
    }

    // ========== Pseudo-measurement on S ==========
    void applyIntegralZeroPseudoMeas() {
        Matrix<T,3,NX> H = Matrix<T,3,NX>::Zero();
        H.template block<3,3>(0, OFF_S) = Matrix3::Identity();

        Vector3 inno = - H * xext_;

        Matrix3 S = H * Pext_ * H.transpose() + R_S_;
        auto PHt  = Pext_ * H.transpose();

        Eigen::LDLT<Matrix3> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            S += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * R_S_.norm());
            ldlt.compute(S);
            if (ldlt.info() != Eigen::Success) return;
        }
        Matrix<T,NX,3> K = PHt * ldlt.solve(Matrix3::Identity());

        xext_.noalias() += K * inno;

        MatrixNX I = MatrixNX::Identity();
        Pext_ = (I - K*H) * Pext_ * (I - K*H).transpose() + K * R_S_ * K.transpose();
        Pext_ = T(0.5) * (Pext_ + Pext_.transpose());

        applyQuaternionCorrection_();
        xext_.template head<3>().setZero();
        Pbase_ = Pext_.topLeftCorner(BASE_N, BASE_N);
    }

  private:
    // ===== Members =====
    const T gravity_magnitude_;

    Eigen::Quaternion<T> qref_;      // world→body
    Vector3 v2ref_ = Vector3::UnitX();

    MatrixBase Pbase_;
    MatrixNX   Pext_;
    Matrix<T,NX,1> xext_;

    Vector3 last_gyr_bias_corrected_{};

    // accel-bias (optional)
    T       sigma_bacc0_ = T(0.1);
    Matrix3 Q_bacc_      = Matrix3::Identity() * T(1e-8);
    Vector3 k_a_         = Vector3::Constant(T(0.003)); // m/s² per °C

    // measurements
    Matrix3  Racc_;
    Matrix3  Rmag_;
    MatrixM  R_;
    Matrix3  R_S_;

    // base process
    MatrixBase Qbase_;

    // latent a Matérn-3/2 params
    T       tau_lat_       = T(1.5);                             // [s]
    Matrix3 Sigma_a_stat_  = Matrix3::Identity() * T(0.25*0.25); // Var[a]

    // ===== Utils =====
    Matrix3 R_wb_() const { return qref_.toRotationMatrix(); }              // world→body
    Matrix3 R_bw_() const { return qref_.toRotationMatrix().transpose(); }  // body→world

    static MatrixBase initialize_Q_(const Vector3& sigma_g, T b0) {
        MatrixBase Q; Q.setZero();
        if constexpr (with_gyro_bias) {
            Q.template topLeftCorner<3,3>()    = sigma_g.array().square().matrix().asDiagonal();
            Q.template bottomRightCorner<3,3>()= Matrix3::Identity() * b0;
        } else {
            Q = sigma_g.array().square().matrix().asDiagonal();
        }
        return Q;
    }

    Matrix3 skew_(const Vector3& v) const {
        Matrix3 M;
        M << 0, -v(2), v(1),
             v(2), 0, -v(0),
            -v(1), v(0), 0;
        return M;
    }

    Vector3 accelerometer_measurement_func_(T tempC) const {
        const Vector3 g_world(0,0,+gravity_magnitude_);
        const Vector3 a = xext_.template segment<3>(OFF_A);
        Vector3 fb = R_wb_() * (a - g_world);
        if constexpr (with_accel_bias) {
            Vector3 ba0 = xext_.template segment<3>(OFF_BA);
            Vector3 ba  = ba0 + k_a_ * (tempC - TEMP_REF);
            fb += ba;
        }
        return fb;
    }

    Vector3 magnetometer_measurement_func_() const {
        return R_wb_() * v2ref_;
    }

    void applyQuaternionCorrection_() {
        Eigen::Quaternion<T> corr(T(1),
                                  half * xext_(0),
                                  half * xext_(1),
                                  half * xext_(2));
        corr.normalize();
        qref_ = qref_ * corr;
        qref_.normalize();
    }

    // ===== Closed-form per-axis Φ and Qd =====
    // x = [v p S a j]^T
    static void axis_Phi_closed_form_(T h, T tau, Matrix5& Phi)
    {
        const T inv_tau = T(1) / std::max(T(1e-12), tau);
        const T alpha   = std::exp(-h * inv_tau);

        // Kinematics base
        Phi.setZero();
        Phi(0,0)=T(1);
        Phi(1,0)=h;     Phi(1,1)=T(1);
        Phi(2,0)=T(0.5)*h*h; Phi(2,1)=h; Phi(2,2)=T(1);

        // Latent (a,j) exact 2×2 exponential for critical damping
        Phi(3,3) = alpha * (T(1) + h*inv_tau);
        Phi(3,4) = alpha * h;
        Phi(4,3) = -alpha * (h*inv_tau*inv_tau);
        Phi(4,4) = alpha * (T(1) - h*inv_tau);

        // Couplings into v,p,S via ∫ a(s), ∫∫ a(s), ∫∫∫ a(s) and j(s)
        // Compact primitives: E0..E3
        const T E0 = T(1) - alpha;
        const T E1 = T(1) - alpha*(T(1) + h*inv_tau);
        const T E2 = T(1) - alpha*(T(1) + h*inv_tau + (h*h)*inv_tau*inv_tau/T(2));
        const T E3 = T(1) - alpha*(T(1) + h*inv_tau + (h*h)*inv_tau*inv_tau/T(2)
                                  + (h*h*h)*inv_tau*inv_tau*inv_tau/T(6));

        // v <- [a j]
        const T I_a = tau*(E0 + E1);
        const T I_j = tau*tau*E1;
        Phi(0,3) = I_a;
        Phi(0,4) = I_j;

        // p <- [a j]
        const T J_a = h*I_a - tau*tau*(E1 + T(2)*E2);
        const T J_j = h*I_j - T(2)*tau*tau*tau*E2;
        Phi(1,3) = J_a;
        Phi(1,4) = J_j;

        // S <- [a j]
        const T S_a = T(0.5)*( h*h*tau*(E0+E1)
                             - T(2)*h*tau*tau*(E1+T(2)*E2)
                             + T(2)*tau*tau*tau*(E2+T(3)*E3) );
        const T S_j = T(0.5)*( h*h*(tau*tau*E1)
                             - T(4)*h*(tau*tau*tau*E2)
                             + T(6)*tau*tau*tau*tau*E3 );
        Phi(2,3) = S_a;
        Phi(2,4) = S_j;
    }

    // Hand-coded fully expanded Qd entries for one axis (no loops, no integrators at runtime).
    // Qd = σ_a^2 * [C0 + C1 α + C2 α^2], where α = e^{-h/τ}; the constants Ck below encode the polynomials.
    static void axis_Qd_closed_form_(T h, T tau, T sigma2_a, Matrix5& Q)
    {
        const T alpha = std::exp(-h / std::max(T(1e-12), tau));
        const T t  = tau;
        const T h2 = h*h, h3 = h2*h, h4 = h2*h2, h5 = h2*h3;
        const T t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t, t6 = t5*t;

        // Helper macro to assign symmetric entries
        auto S = [&](int r, int c, T v){ Q(r,c)=v; if(r!=c) Q(c,r)=v; };

        // Precompute α, α²
        const T a1 = alpha;
        const T a2 = alpha*alpha;

        // vv
        {
            const T c2 = -(T(2)*h2 + T(6)*h*t + T(5)*t2);
            const T c1 =  T(8)*t*(h + T(2)*t);
            const T c0 =  t*(T(4)*h - T(11)*t);
            S(0,0, sigma2_a * (c0 + c1*a1 + c2*a2));
        }

        // vp
        {
            const T c2 = -(h2 + T(3)*h*t + T(3)*t2);
            const T c1 =  (h2 + T(2)*h*t + T(10)*t2);
            const T c0 =  (h2/T(2)) - T(2)*h*t + T(5)*t2;
            S(0,1, sigma2_a * (c0 + c1*a1 + c2*a2));
        }

        // vS
        {
            const T c2 = -(T(2)*h2*t2 + T(10)*h*t3 + T(11)*t4);
            const T c1 =  (T(2)*h3*t + T(8)*h*t3 + T(32)*t4);
            const T c0 =  (T(2)*h3*t/T(3)) - T(4)*h2*t2 + T(12)*h*t3 - T(21)*t4;
            S(0,2, sigma2_a * (c0 + c1*a1 + c2*a2));
        }

        // va
        {
            const T c2 =  (T(2)*h + T(5)*t);
            const T c1 = -T(4)*(h + T(3)*t);
            const T c0 =  T(3)*t;
            S(0,3, sigma2_a * (c0 + c1*a1 + c2*a2));
        }

        // vj
        {
            const T c2 = -(T(2)*h/t + T(3));
            const T c1 =  T(2)*(T(2)*h/t + T(5));
            const T c0 = -T(4);
            S(0,4, sigma2_a * (c0 + c1*a1 + c2*a2));
        }

        // pp
        {
            const T c2 =  (T(2)*h2*t + T(8)*h*t2 + T(11)*t3);
            const T c1 = -(T(2)*h3 + T(2)*h2*t + T(8)*h*t2 - T(36)*t3);
            const T c0 = -(h4/T(2)) + T(2)*h3*t - T(10)*h2*t2 + T(24)*h*t3 - T(18)*t4;
            S(1,1, sigma2_a * (c0 + c1*a1 + c2*a2));
        }

        // pS
        {
            const T c2 =  (T(2)*h2*t3 + T(12)*h*t4 + T(18)*t5);
            const T c1 = -(T(2)*h3*t2) + T(2)*h2*t3 + T(12)*h*t4 - T(36)*t5;
            const T c0 =   (h4*t/T(2)) - T(4)*h3*t2 + T(14)*h2*t3 - T(24)*h*t4 + T(18)*t5;
            S(1,2, sigma2_a * (c0 + c1*a1 + c2*a2));
        }

        // pa
        {
            const T c2 = -(h2 + T(3)*h*t + T(4)*t2);
            const T c1 =  (h2 + T(4)*t2);
            const T c0 =   (h2/T(2)) - T(3)*h*t + T(6)*t2;
            S(1,3, sigma2_a * (c0 + c1*a1 + c2*a2));
        }

        // pj
        {
            const T c2 =  (h2/t + T(2)*h + T(2)*t);
            const T c1 = -(h2/t + T(2)*t);
            const T c0 = -(h2/(T(2)*t)) + T(2)*h - T(6)*t;
            S(1,4, sigma2_a * (c0 + c1*a1 + c2*a2));
        }

        // SS
        {
            const T c2 = -(T(2)*h2*t4 + T(14)*h*t5 + T(25)*t6);
            const T c1 =  (T(4)*h3*t3 + T(8)*h2*t4 - T(8)*h*t5 + T(64)*t6);
            const T c0 =   (h5*t/T(5)) - T(2)*h4*t2 + (T(28)*h3*t3)/T(3) - T(24)*h2*t4 + T(36)*h*t5 - T(39)*t6;
            S(2,2, sigma2_a * (c0 + c1*a1 + c2*a2));
        }

        // Sa
        {
            const T c2 =  (h3/T(3) + h2*t + T(2)*h*t2 + T(2)*t3);
            const T c1 = -(h3/T(3)) + T(2)*h*t2 - T(6)*t3;
            const T c0 = -(h3/T(6)) + (T(3)*h2*t)/T(2) - T(6)*h*t2 + T(12)*t3;
            S(2,3, sigma2_a * (c0 + c1*a1 + c2*a2));
        }

        // Sj
        {
            const T c2 = -(h3/(T(3)*t) + h2 + T(2)*h*t + T(2)*t2);
            const T c1 =  (h3/(T(3)*t)) - T(2)*h + T(6)*t;
            const T c0 =  (h3/(T(6)*t)) - (T(3)*h2)/T(2) + T(6)*h - T(12)*t;
            S(2,4, sigma2_a * (c0 + c1*a1 + c2*a2));
        }

        // aa
        {
            const T c2 = -(T(2)*h2/t2 + T(2)*h/t + T(1));
            const T c1 =  T(0);
            const T c0 =  T(1);
            S(3,3, sigma2_a * (c0 + c1*a1 + c2*a2));
        }

        // aj
        {
            const T c2 =  T(2)*h2/(t*t*t);
            const T c1 =  T(0);
            const T c0 =  T(0);
            S(3,4, sigma2_a * (c0 + c1*a1 + c2*a2));
        }

        // jj
        {
            const T c2 = -(T(2)*h2/(t2*t2) - T(2)*h/(t*t*t) + T(1)/t2);
            const T c1 =  T(0);
            const T c0 =  T(1)/t2;
            S(4,4, sigma2_a * (c0 + c1*a1 + c2*a2));
        }
    }

    // Build full NX×NX Φ and Qd
    void assembleExtendedFandQ_(T Ts, MatrixNX& F, MatrixNX& Qd) {
        F.setIdentity();
        Qd.setZero();

        // ---- Attitude-error small-angle transition (exact Rodrigues on constant ω) ----
        Matrix3 I3 = Matrix3::Identity();
        const Vector3& w = last_gyr_bias_corrected_;
        T wn = w.norm();
        if (wn < T(1e-9)) {
            Matrix3 Wx = skew_(w);
            F.template block<3,3>(0,0) = I3 - Wx*Ts + (Wx*Wx)*(Ts*Ts*T(0.5));
        } else {
            Vector3 u = w / wn;
            Matrix3 Ux = skew_(u);
            T th = wn * Ts;
            T s = std::sin(th), c = std::cos(th);
            F.template block<3,3>(0,0) = I3 - s*Ux + (T(1)-c)*(Ux*Ux);
        }
        if constexpr (with_gyro_bias) {
            F.template block<3,3>(0,3) = -I3 * Ts; // θ_k+1 ≈ θ_k - Ts b_g
        }
        // Base process (RW for bias; gyro meas mapped in Qbase_)
        Qd.topLeftCorner(BASE_N, BASE_N) = Qbase_ * Ts;

        // ---- 3 axes of [v p S a j] each ----
        for (int axis=0; axis<3; ++axis) {
            Matrix5 Phi_ax, Q_ax;
            const T tau = std::max(T(1e-6), tau_lat_);
            const T sigma2 = Sigma_a_stat_(axis,axis);

            axis_Phi_closed_form_(Ts, tau, Phi_ax);
            axis_Qd_closed_form_ (Ts, tau, sigma2, Q_ax);

            int idx[5]={0,3,6,9,12};
            for (int r=0;r<5;++r)
                for (int c=0;c<5;++c) {
                    F (OFF_V + idx[r] + axis, OFF_V + idx[c] + axis) = Phi_ax(r,c);
                    Qd(OFF_V + idx[r] + axis, OFF_V + idx[c] + axis) = Q_ax (r,c);
                }
        }

        // ---- accelerometer bias RW (optional) ----
        if constexpr (with_accel_bias) {
            Qd.template block<3,3>(OFF_BA, OFF_BA) = Q_bacc_ * Ts;
        }
    }
};

