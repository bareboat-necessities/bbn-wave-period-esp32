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

// ==== stable helpers ====
template<typename T>
inline void emoments_E0123(const T th, T& E0, T& E1, T& E2, T& E3)
{
    const T eps = T(1e-6);
    if (std::abs(th) < eps) {
        const T th2 = th*th, th3 = th2*th, th4 = th2*th2, th5 = th3*th2;
        E0 = th - th2/T(2) + th3/T(6) - th4/T(24) + th5/T(120);
        E1 = th2/T(2) - th3/T(3) + th4/T(8) - th5/T(30);
        E2 = th3/T(6) - th4/T(8) + th5/T(20);
        E3 = th4/T(24) - th5/T(30);
        return;
    }
    const T alpha = std::exp(-th);
    const T S1 = 1 + th;
    const T S2 = S1 + th*th/T(2);
    const T S3 = S2 + th*th*th/T(6);
    E0 = 1 - alpha;
    E1 = 1 - alpha*S1;
    E2 = 1 - alpha*S2;
    E3 = 1 - alpha*S3;
}

// ∫₀ʰ s^n e^{-λ s} ds, for n=0..4 (closed form, numerically stable)
template<typename T>
inline T integ_pow_exp_0toh(int n, T lambda, T h)
{
    if (lambda <= T(0)) {
        switch(n){
            case 0: return h;
            case 1: return T(0.5)*h*h;
            case 2: return h*h*h/T(3);
            case 3: return h*h*h*h/T(4);
            case 4: return h*h*h*h*h/T(5);
        }
    }
    const T x = lambda*h;
    T Sm = T(1), term = T(1);
    for (int k=1;k<=n;++k){ term *= x/T(k); Sm += term; }
    T fact = T(1); for (int k=2;k<=n;++k) fact *= T(k);
    T lam_pow = std::pow(lambda, T(n+1));
    return (fact/lam_pow) * (T(1) - std::exp(-x)*Sm);
}

// ----------------------------------------------------------------------
// Per-axis Matérn-3/2 (critically damped) closed-form Φ and Qd (5x5)
// x = [v, p, S, a, j]^T
// j̇ = -(2/τ) j - (1/τ²) a + w,  E[w wᵀ] = q_c δ,  q_c = 4 σ_a² / τ³
// ----------------------------------------------------------------------
template<typename T>
inline void phi_Qd_axis_M32(const T h, const T tau, const T sigma2_a,
                            Eigen::Matrix<T,5,5>& Phi,
                            Eigen::Matrix<T,5,5>& Qd)
{
    using Mat5 = Eigen::Matrix<T,5,5>;

    // ---------- Φ(h) ----------
    const T tau_c   = std::max(tau, T(1e-12));
    const T inv_tau = T(1) / tau_c;
    const T th      = h * inv_tau;
    const T alpha   = std::exp(-th);

    T E0,E1,E2,E3;
    emoments_E0123(th, E0, E1, E2, E3);

    Phi.setZero();
    // kinematics
    Phi(0,0)=T(1);
    Phi(1,0)=h;        Phi(1,1)=T(1);
    Phi(2,0)=T(0.5)*h*h; Phi(2,1)=h; Phi(2,2)=T(1);

    // latent (a,j)
    Phi(3,3) = alpha*(T(1) + th);
    Phi(3,4) = alpha*h;
    Phi(4,3) = -alpha*(h*inv_tau*inv_tau);
    Phi(4,4) = alpha*(T(1) - th);

    // couplings from (a0,j0) into (v,p,S)
    const T I_a = tau_c*(E0 + E1);
    const T I_j = tau_c*tau_c*E1;
    Phi(0,3) = I_a;
    Phi(0,4) = I_j;

    const T J_a = h*I_a - tau_c*tau_c*(E1 + T(2)*E2);
    const T J_j = h*I_j - T(2)*tau_c*tau_c*tau_c*E2;
    Phi(1,3) = J_a;
    Phi(1,4) = J_j;

    const T S_a = T(0.5)*( h*h*tau_c*(E0+E1)
                         - T(2)*h*tau_c*tau_c*(E1+T(2)*E2)
                         + T(2)*tau_c*tau_c*tau_c*(E2+T(3)*E3) );
    const T S_j = T(0.5)*( h*h*(tau_c*tau_c*E1)
                         - T(4)*h*(tau_c*tau_c*tau_c*E2)
                         + T(6)*tau_c*tau_c*tau_c*tau_c*E3 );
    Phi(2,3) = S_a;
    Phi(2,4) = S_j;

    // ---------- Qd(h) (exact from g(s)g(s)^T integral) ----------
    // Build g(s)=Φ(s)G for G=[0,0,0,0,1]^T (noise drives j).
    // Using the closed-form solutions with s as variable:
    //  a(s) = s e^{-s/τ}
    //  j(s) = e^{-s/τ}(1 - s/τ)
    //  v(s) = τ² - τ² e^{-s/τ} - τ s e^{-s/τ}
    //  p(s) = τ² s - 2τ³ + e^{-s/τ}(2τ³ + τ² s)
    //  S(s) = 0.5 τ² s² - 2τ³ s + 3τ⁴ + e^{-s/τ}(-3τ⁴ - τ³ s)
    //
    // Qd = q_c ∫₀ʰ g(s) g(s)^T ds,  with  q_c = 4 σ_a² / τ³
    const T qc = T(4)*sigma2_a/(tau_c*tau_c*tau_c);

    auto I0  = [&](int n){ return integ_pow_exp_0toh(n, T(0),          h); }; // ∫ s^n ds
    auto I1  = [&](int n){ return integ_pow_exp_0toh(n, inv_tau,       h); }; // ∫ s^n e^{-s/τ} ds
    auto I2  = [&](int n){ return integ_pow_exp_0toh(n, T(2)*inv_tau,  h); }; // ∫ s^n e^{-2s/τ} ds

    const T t1 = tau_c;
    const T t2 = t1*t1;
    const T t3 = t2*t1;
    const T t4 = t3*t1;

    // g components
    // v: A + B e^- + C s e^- (A=t2, B=-t2, C=-t1)
    const T Av=t2, Bv=-t2, Cv=-t1;
    // p: D s + E + (F + G s) e^- (D=t2, E=-2t3, F=2t3, G=t2)
    const T Dp=t2, Ep=-T(2)*t3, Fp=T(2)*t3, Gp=t2;
    // S: H s^2 + K s + L + (M + N s) e^- (H=0.5 t2, K=-2 t3, L=3 t4, M=-3 t4, N=-t3)
    const T Hs=T(0.5)*t2, Ks=-T(2)*t3, Ls=T(3)*t4, Ms=-T(3)*t4, Ns=-t3;
    // a: s e^-
    // j: e^- + (-1/τ) s e^-
    const T J0=T(1), J1=-inv_tau;

    Qd.setZero();
    auto S = [&](int i,int j,T v){ Qd(i,j)=v; if(i!=j) Qd(j,i)=v; };

    // <v,v>
    {
        T val = Av*Av*I0(0)
              + Bv*Bv*I2(0)
              + Cv*Cv*I2(2)
              + T(2)*Av*Bv*I1(0)
              + T(2)*Av*Cv*I1(1)
              + T(2)*Bv*Cv*I2(1);
        S(0,0, qc*val);
    }

    // <v,p>
    {
        T val = (Av*Dp)*I0(1) + (Av*Ep)*I0(0)
              + Av*Fp*I1(0) + Av*Gp*I1(1)
              + Bv*Dp*I1(1) + Bv*Ep*I1(0)
              + Bv*Fp*I2(0) + Bv*Gp*I2(1)
              + Cv*Dp*I1(2) + Cv*Ep*I1(1)
              + Cv*Fp*I2(1) + Cv*Gp*I2(2);
        S(0,1, qc*val);
    }

    // <v,S>
    {
        T val = Av*Hs*I0(2) + Av*Ks*I0(1) + Av*Ls*I0(0)
              + Av*Ms*I1(0) + Av*Ns*I1(1)
              + Bv*Hs*I1(2) + Bv*Ks*I1(1) + Bv*Ls*I1(0)
              + Bv*Ms*I2(0) + Bv*Ns*I2(1)
              + Cv*Hs*I1(3) + Cv*Ks*I1(2) + Cv*Ls*I1(1)
              + Cv*Ms*I2(1) + Cv*Ns*I2(2);
        S(0,2, qc*val);
    }

    // <v,a>
    {
        T val = Av*I1(1) + Bv*I2(1) + Cv*I2(2);
        S(0,3, qc*val);
    }

    // <v,j>
    {
        T val = Av*J0*I1(0) + Av*J1*I1(1)
              + Bv*J0*I2(0) + Bv*J1*I2(1)
              + Cv*J0*I2(1) + Cv*J1*I2(2);
        S(0,4, qc*val);
    }

    // <p,p>
    {
        T val = Dp*Dp*I0(2) + T(2)*Dp*Ep*I0(1) + Ep*Ep*I0(0)
              + T(2)*( Dp*Fp*I1(1) + Dp*Gp*I1(2) + Ep*Fp*I1(0) + Ep*Gp*I1(1) )
              + Fp*Fp*I2(0) + T(2)*Fp*Gp*I2(1) + Gp*Gp*I2(2);
        S(1,1, qc*val);
    }

    // <p,S>
    {
        T val = Dp*Hs*I0(3) + Dp*Ks*I0(2) + Dp*Ls*I0(1)
              + Ep*Hs*I0(2) + Ep*Ks*I0(1) + Ep*Ls*I0(0)
              + Dp*Ms*I1(1) + Dp*Ns*I1(2) + Ep*Ms*I1(0) + Ep*Ns*I1(1)
              + Fp*Hs*I1(2) + Fp*Ks*I1(1) + Fp*Ls*I1(0)
              + Gp*Hs*I1(3) + Gp*Ks*I1(2) + Gp*Ls*I1(1)
              + Fp*Ms*I2(0) + Fp*Ns*I2(1) + Gp*Ms*I2(1) + Gp*Ns*I2(2);
        S(1,2, qc*val);
    }

    // <p,a>
    {
        T val = Dp*I1(2) + Ep*I1(1) + Fp*I2(1) + Gp*I2(2);
        S(1,3, qc*val);
    }

    // <p,j>
    {
        T val = Dp*J0*I1(1) + Dp*J1*I1(2) + Ep*J0*I1(0) + Ep*J1*I1(1)
              + Fp*J0*I2(0) + Fp*J1*I2(1) + Gp*J0*I2(1) + Gp*J1*I2(2);
        S(1,4, qc*val);
    }

    // <S,S>
    {
        T val = Hs*Hs*I0(4) + T(2)*Hs*Ks*I0(3) + (T(2)*Hs*Ls + Ks*Ks)*I0(2)
              + T(2)*Ks*Ls*I0(1) + Ls*Ls*I0(0)
              + T(2)*( Hs*Ms*I1(2) + Hs*Ns*I1(3) + Ks*Ms*I1(1) + Ks*Ns*I1(2) + Ls*Ms*I1(0) + Ls*Ns*I1(1) )
              + Ms*Ms*I2(0) + T(2)*Ms*Ns*I2(1) + Ns*Ns*I2(2);
        S(2,2, qc*val);
    }

    // <S,a>
    {
        T val = Hs*I1(3) + Ks*I1(2) + Ls*I1(1) + Ms*I2(1) + Ns*I2(2);
        S(2,3, qc*val);
    }

    // <S,j>
    {
        T val = Hs*J0*I1(2) + Hs*J1*I1(3)
              + Ks*J0*I1(1) + Ks*J1*I1(2)
              + Ls*J0*I1(0) + Ls*J1*I1(1)
              + Ms*J0*I2(0) + Ms*J1*I2(1) + Ns*J0*I2(1) + Ns*J1*I2(2);
        S(2,4, qc*val);
    }

    // <a,a>
    S(3,3, qc*I2(2));

    // <a,j>
    {
        T val = J0*I2(1) + J1*I2(2);
        S(3,4, qc*val);
    }

    // <j,j>
    {
        T val = J0*J0*I2(0) + T(2)*J0*J1*I2(1) + J1*J1*I2(2);
        S(4,4, qc*val);
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
    // Instead of two separate stubs, just delegate to m32_analytic::
    // (remove axis_Phi_closed_form_ and axis_Qd_closed_form_ completely)

    // Build full NX×NX Φ and Qd
// Build full NX×NX Φ and Qd
void assembleExtendedFandQ_(T Ts, MatrixNX& F, MatrixNX& Qd) {
    F.setIdentity();
    Qd.setZero();

    // --- Attitude-error transition (small-angle error state) ---
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
        F.template block<3,3>(0,3) = -I3 * Ts;  // θ_{k+1} ≈ θ_k - Ts * b_g
    }
    Qd.topLeftCorner(BASE_N, BASE_N) = Qbase_ * Ts;

    // --- 3 axes of [v p S a j] each ---
    for (int axis=0; axis<3; ++axis) {
        Matrix5 Phi_ax, Q_ax;
        const T tau    = std::max(T(1e-6), tau_lat_);
        const T sigma2 = Sigma_a_stat_(axis,axis);

        m32_analytic::phi_Qd_axis_M32(Ts, tau, sigma2, Phi_ax, Q_ax);

        int idx[5]={0,3,6,9,12};
        for (int r=0;r<5;++r)
            for (int c=0;c<5;++c) {
                F (OFF_V + idx[r] + axis, OFF_V + idx[c] + axis) = Phi_ax(r,c);
                Qd(OFF_V + idx[r] + axis, OFF_V + idx[c] + axis) = Q_ax (r,c);
            }
    }

    // --- accelerometer bias RW (optional) ---
    if constexpr (with_accel_bias) {
        Qd.template block<3,3>(OFF_BA, OFF_BA) = Q_bacc_ * Ts;
    }
}
};

