#pragma once
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif

namespace Elliptic {

    // === Complete elliptic integral of the first kind (AGM, safeguarded) ===
    template<typename Real>
    Real ellipK(Real m) {
        if (!(m >= Real(0) && m <= Real(1))) {
            throw std::domain_error("ellipK: m out of range [0,1]");
        }
        if (m == Real(0)) return Real(M_PI) / Real(2);
        if (m == Real(1)) return std::numeric_limits<Real>::infinity();

        Real a = Real(1);
        Real b = std::sqrt(Real(1) - m);

        const Real atol = Real(1e-15);
        const int  maxit = 50;

        for (int it = 0; it < maxit; ++it) {
            const Real an = (a + b) / Real(2);
            const Real bn = std::sqrt(a * b);
            // convergence on |a-b|
            if (std::abs(an - bn) <= std::max(atol, atol * an)) {
                a = an; b = bn;
                break;
            }
            a = an; b = bn;
        }
        return Real(M_PI) / (Real(2) * a);
    }

    // === Complete elliptic integral of the second kind (AGM, safeguarded) ===
    // Uses the standard AGM accumulator for E based on c_n = (a_n - b_n)/2.
    template<typename Real>
    Real ellipE(Real m) {
        if (!(m >= Real(0) && m <= Real(1))) {
            throw std::domain_error("ellipE: m out of range [0,1]");
        }
        if (m == Real(0)) return Real(M_PI) / Real(2);
        if (m == Real(1)) return Real(1);

        Real a = Real(1);
        Real b = std::sqrt(Real(1) - m);

        Real sum = Real(0);
        Real pow2 = Real(1);     // 2^n
        const Real atol = Real(1e-15);
        const int  maxit = 50;

        for (int it = 0; it < maxit; ++it) {
            const Real c = (a - b) / Real(2);
            const Real an = (a + b) / Real(2);
            const Real bn = std::sqrt(a * b);

            sum  += pow2 * c * c;
            pow2 *= Real(2);

            if (std::abs(an - bn) <= std::max(atol, atol * an)) {
                a = an; b = bn;
                break;
            }
            a = an; b = bn;
        }
        return Real(M_PI) / (Real(2) * a) * (Real(1) - sum / Real(2));
    }

    // === Jacobi elliptic sn, cn, dn (simple approximation) ===
    // For surface-only kinematics this is sufficient; replace with a full algorithm if needed.
    template<typename Real>
    void jacobi_sn_cn_dn(Real u, Real m, Real &sn, Real &cn, Real &dn) {
        if (m == Real(0)) { sn = std::sin(u); cn = std::cos(u); dn = Real(1); return; }
        if (m == Real(1)) { sn = std::tanh(u); cn = Real(1)/std::cosh(u); dn = cn; return; }
        sn = std::sin(u);
        cn = std::cos(u);
        dn = std::sqrt(Real(1) - m * sn * sn);
    }

} // namespace Elliptic


template<typename Real = double>
class EIGEN_ALIGN_MAX CnoidalWave {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct State {
        Eigen::Vector3d displacement;   // (x,y,z)
        Eigen::Vector3d velocity;       // (vx,vy,vz)
        Eigen::Vector3d acceleration;   // (ax,ay,az)
    };

    CnoidalWave(Real depth,
                Real height,
                Real period,
                Real dir_angle = Real(0),
                Real gravity = Real(9.80665))
      : h(depth), H(height), T(period), theta(dir_angle), g(gravity)
    {
        if (h <= Real(0) || H <= Real(0) || T <= Real(0)) {
            throw std::domain_error("CnoidalWave: invalid parameters");
        }
        solveEllipticParameters();
        cosTheta = std::cos(theta);
        sinTheta = std::sin(theta);
    }

    // Free surface elevation η(x,y,t)
    Real surfaceElevation(Real x, Real y, Real t) const {
        const Real s = x * cosTheta + y * sinTheta;
        const Real arg = k * (s - c * t);
        Real sn, cn, dn;
        Elliptic::jacobi_sn_cn_dn(arg, m, sn, cn, dn);
        return eta0 + Hc * cn * cn;
    }

    // Vertical velocity (surface only)
    Real wVelocity(Real x, Real y, Real t) const {
        const Real s = x * cosTheta + y * sinTheta;
        const Real arg = k * (s - c * t);
        Real sn, cn, dn;
        Elliptic::jacobi_sn_cn_dn(arg, m, sn, cn, dn);
        // d/dt [Hc * cn^2] = 2 Hc * cn * (d/dt cn) = 2 Hc * cn * (-sn * dn) * (d/dt am)
        // For our simplified sn/cn/dn, use k*c as phase speed factor
        return Real(2) * Hc * cn * sn * k * c;
    }

    // Vertical acceleration (surface only)
    Real azAcceleration(Real x, Real y, Real t) const {
        const Real s = x * cosTheta + y * sinTheta;
        const Real arg = k * (s - c * t);
        Real sn, cn, dn;
        Elliptic::jacobi_sn_cn_dn(arg, m, sn, cn, dn);
        // Simple harmonic-like second derivative surrogate
        return -Real(2) * Hc * k * k * c * c * (cn * cn - sn * sn);
    }

    // Lagrangian-like surface state (vertical only for this model)
    State getLagrangianState(Real x, Real y, Real t) const {
        State st;
        st.displacement = Eigen::Vector3d(0.0, 0.0, static_cast<double>(surfaceElevation(x,y,t)));
        st.velocity     = Eigen::Vector3d(0.0, 0.0, static_cast<double>(wVelocity(x,y,t)));
        st.acceleration = Eigen::Vector3d(0.0, 0.0, static_cast<double>(azAcceleration(x,y,t)));
        return st;
    }

    // Accessors
    Real depth()      const { return h; }
    Real height()     const { return H; }
    Real period()     const { return T; }
    Real direction()  const { return theta; }
    Real wavenumber() const { return k; }
    Real speed()      const { return c; }
    Real frequency()  const { return omega; }
    Real ellipticM()  const { return m; }

private:
    // Inputs
    Real h, H, T, theta, g;

    // Derived
    Real m, K, E, k, c, eta0, Hc;
    Real omega;
    Real cosTheta, sinTheta;

    void solveEllipticParameters() {
        // Safeguarded solve for modulus m by matching period T
        const Real eps  = Real(1e-8);
        const Real tolF = Real(1e-9);
        const Real tolM = Real(1e-10);
        const int  maxit = 50;

        auto T_of_m = [&](Real mm) {
            Real Kloc = Elliptic::ellipK(mm);
            // Classical shallow-water cnoidal period scaling used in your formula:
            return Real(2) * Kloc * std::sqrt(h / (Real(3) * g * (Real(1) - mm)));
        };

        m = Real(0.8);
        Real m_prev = m;
        Real f_prev = std::numeric_limits<Real>::quiet_NaN();

        for (int it = 0; it < maxit; ++it) {
            K = Elliptic::ellipK(m);
            E = Elliptic::ellipE(m);

            const Real Tguess = T_of_m(m);
            const Real f = Tguess - T;

            if (std::abs(f) < tolF) break;

            // Try Newton step with central difference derivative
            const Real dm = Real(1e-4);
            const Real m_plus  = std::min(m + dm, Real(1) - eps);
            const Real m_minus = std::max(m - dm, eps);
            const Real Tp = T_of_m(m_plus);
            const Real Tm = T_of_m(m_minus);
            Real df = (Tp - Tm) / (m_plus - m_minus);

            Real delta;
            if (std::abs(df) > Real(1e-14) && std::isfinite(df)) {
                delta = -f / df; // Newton
            } else if (it > 0 && std::isfinite(f_prev)) {
                // Secant fallback
                Real denom = (f - f_prev);
                if (std::abs(denom) < Real(1e-14)) denom = (denom >= 0 ? Real(1e-14) : Real(-1e-14));
                delta = -f * (m - m_prev) / denom;
            } else {
                // Small nudge if we have nothing better
                delta = (f > 0 ? Real(-1e-2) : Real(1e-2));
            }

            // Damp step to avoid overshoot
            const Real maxStep = Real(0.25);
            if (delta >  maxStep) delta =  maxStep;
            if (delta < -maxStep) delta = -maxStep;

            m_prev = m;
            f_prev = f;

            m += delta;
            m = std::clamp(m, eps, Real(1) - eps);

            if (std::abs(delta) < tolM * std::max(Real(1), std::abs(m))) break;
        }

        // Finalize parameters
        K = Elliptic::ellipK(m);
        E = Elliptic::ellipE(m);
        k = Real(M_PI) / (K * h);
        omega = Real(2 * M_PI) / T;
        c = omega / k;
        eta0 = Real(0);
        Hc   = H;
    }
};
