#pragma once
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

/*
   Cnoidal wave (symmetric cn-form) for zero-mean Lagrangian vertical motion
   Copyright 2025, Mikhail Grushinskiy
*/

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif

// Elliptic integrals and Jacobi functions (AGM-based)
namespace Elliptic {

    // Complete elliptic integral of the first kind (K)
    // K(m) = ∫₀^(π/2) dφ / √(1 - m sin²φ)
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
            if (std::abs(an - bn) <= std::max(atol, atol * an)) { a = an; break; }
            a = an; b = bn;
        }
        return Real(M_PI) / (Real(2) * a);
    }

    // Complete elliptic integral of the second kind (E)
    // E(m) = ∫₀^(π/2) √(1 - m sin²φ) dφ
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
        Real pow2 = Real(1);   // 2^n
        const Real atol = Real(1e-15);
        const int  maxit = 50;

        for (int it = 0; it < maxit; ++it) {
            const Real c  = (a - b) / Real(2);
            const Real an = (a + b) / Real(2);
            const Real bn = std::sqrt(a * b);

            sum  += pow2 * c * c;
            pow2 *= Real(2);

            if (std::abs(an - bn) <= std::max(atol, atol * an)) { a = an; break; }
            a = an; b = bn;
        }
        return Real(M_PI) / (Real(2) * a) * (Real(1) - sum / Real(2));
    }

    // Jacobi elliptic functions sn, cn, dn (AGM/Landen method)
    // Robust for 0 ≤ m ≤ 1, u ∈ ℝ
    template<typename Real>
    void jacobi_sn_cn_dn(Real u, Real m, Real &sn, Real &cn, Real &dn) {
        if (!(m >= Real(0) && m <= Real(1))) {
            throw std::domain_error("jacobi_sn_cn_dn: m out of range [0,1]");
        }
        // Edge cases
        if (m == Real(0)) { sn = std::sin(u); cn = std::cos(u); dn = Real(1); return; }
        if (m == Real(1)) { sn = std::tanh(u); cn = Real(1)/std::cosh(u); dn = cn; return; }

        constexpr int NMAX = 32;
        const Real tol = Real(1e-12);

        Real a[NMAX], c[NMAX];
        int  n = 0;

        Real a_n = Real(1);
        Real b_n = std::sqrt(Real(1) - m);
        Real twon = Real(1);

        for (; n < NMAX; ++n) {
            a[n] = a_n;
            c[n] = (a_n - b_n) / Real(2);
            const Real an1 = (a_n + b_n) / Real(2);
            const Real bn1 = std::sqrt(a_n * b_n);

            if (std::abs(c[n]) <= tol * an1) { a_n = an1; ++n; break; }

            a_n = an1;
            b_n = bn1;
            twon *= Real(2);
        }

        // Bulirsch scaling of phase
        Real phi = twon * a_n * u;

        // Backward recurrence for amplitude
        for (int j = n - 1; j >= 0; --j) {
            const Real s = std::sin(phi);
            const Real t = (c[j] * s) / a[j];
            const Real arg = std::clamp(t, Real(-1), Real(1));
            phi = (std::asin(arg) + phi) / Real(2);
        }

        sn = std::sin(phi);
        cn = std::cos(phi);
        dn = std::sqrt(std::max(Real(0), Real(1) - m * sn * sn));
    }

} // namespace Elliptic


// Cnoidal Wave Model (symmetric form with cn)
// 
// We use   η(x,y,t) = A * cn(θ | m),   A = H/2
// so that:
//  • mean(η) = 0 over one period,
//  • crest = +H/2, trough = −H/2 (crest–trough = H),
//  • simple, stable kinematics for w and a_z.
//
// Phase:   θ = k (s − c t),  s = x cos(θ_dir) + y sin(θ_dir)
// wavenumber: k = π / (K(m) h)
// angular frequency: ω = 2π / T
// phase speed: c = ω / k
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
        A = H / Real(2); // symmetric amplitude
    }

    // Free surface elevation:
    //   η(x,y,t) = A * cn(θ | m),  with A = H/2
    //   mean(η) = 0 over one full period (4K).
    Real surfaceElevation(Real x, Real y, Real t) const {
        const Real s   = x * cosTheta + y * sinTheta;
        const Real th  = k * (s - c * t);
        Real sn, cn, dn;
        Elliptic::jacobi_sn_cn_dn(th, m, sn, cn, dn);
        return A * cn;
    }

    // Vertical velocity:
    //   w = dη/dt = A * d/dt[cn(θ)] = A * (-sn dn) * (dθ/dt)
    //   dθ/dt = -kc = -ω  →  w = A * ω * sn * dn
    Real wVelocity(Real x, Real y, Real t) const {
        const Real s   = x * cosTheta + y * sinTheta;
        const Real th  = k * (s - c * t);
        Real sn, cn, dn;
        Elliptic::jacobi_sn_cn_dn(th, m, sn, cn, dn);
        return A * omega * sn * dn;
    }

    // Vertical acceleration:
    //   a = d/dt w = A ω * d/dt[sn dn] = A ω * (d(sn dn)/dθ) * (dθ/dt)
    //   d(sn dn)/dθ = cn dn² - m sn² cn = cn(1 - m sn²) - m sn² cn
    //                = cn (1 - 2 m sn²)
    //   dθ/dt = -ω → a = -A ω² cn (1 - 2 m sn²)
    Real azAcceleration(Real x, Real y, Real t) const {
        const Real s   = x * cosTheta + y * sinTheta;
        const Real th  = k * (s - c * t);
        Real sn, cn, dn;
        Elliptic::jacobi_sn_cn_dn(th, m, sn, cn, dn);
        const Real sn2 = sn * sn;
        return -A * omega * omega * cn * (Real(1) - Real(2) * m * sn2);
    }

    // (vertical only for this model)
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
    Real m, K, E, k, c, omega;
    Real A;                  // amplitude = H/2 (symmetric)
    Real cosTheta, sinTheta;

    // Elliptic parameter solver:
    // Iteratively solves for m to match desired period T
    // using Newton + secant fallback (no hangs).
    void solveEllipticParameters() {
        const Real eps   = Real(1e-8);
        const Real tolF  = Real(1e-9);
        const Real tolM  = Real(1e-10);
        const int  maxit = 50;

        // Physics-aware initial guess for m
        // T0 = π * sqrt(h / (3 g))  (value of T when m -> 0 and K ~ π/2)
        const Real T0  = Real(M_PI) * std::sqrt(h / (Real(3) * g));
        const Real tau = T / T0; // normalized period

        auto clamp01 = [&](Real x) {
            return std::clamp(x, eps, Real(1) - eps);
        };

        // If tau <= 1, the linear limit (m≈0) is appropriate.
        if (tau <= Real(1) + Real(1e-6)) {
            m = eps; // essentially sinusoidal
        } else {
            // First pass ignoring K(m) variation: tau ≈ 1/sqrt(1-m) ⇒ m ≈ 1 - 1/tau^2
            m = clamp01(Real(1) - Real(1) / (tau * tau));

            // 1–2 fixed-point refinements using actual K(m):
            // tau = (2K/π) / sqrt(1-m)  ⇒ 1 - m = ( (2K/π) / tau )^2  ⇒ m = 1 - (...)
            for (int it = 0; it < 2; ++it) {
                const Real Ktmp = Elliptic::ellipK(m);
                const Real Ktil = (Real(2) * Ktmp) / Real(M_PI);
                const Real one_minus_m = (Ktil / tau) * (Ktil / tau);
                const Real m_new = clamp01(Real(1) - one_minus_m);
                if (std::abs(m_new - m) < Real(1e-6)) { m = m_new; break; }
                m = m_new;
            }
        }

        // Newton with secant fallback 
        auto T_of_m = [&](Real mm) {
            Real Kloc = Elliptic::ellipK(mm);
            return Real(2) * Kloc * std::sqrt(h / (Real(3) * g * (Real(1) - mm)));
        };

        Real m_prev = m;
        Real f_prev = std::numeric_limits<Real>::quiet_NaN();

        for (int it = 0; it < maxit; ++it) {
            K = Elliptic::ellipK(m);
            E = Elliptic::ellipE(m);

            const Real Tguess = T_of_m(m);
            const Real f = Tguess - T;
            if (std::abs(f) < tolF) break;

            const Real dm     = Real(1e-4);
            const Real m_plus  = std::min(m + dm, Real(1) - eps);
            const Real m_minus = std::max(m - dm, eps);
            const Real Tp = T_of_m(m_plus);
            const Real Tm = T_of_m(m_minus);
            Real df = (Tp - Tm) / (m_plus - m_minus);

            Real delta;
            if (std::abs(df) > Real(1e-14) && std::isfinite(df)) {
                // Newton step
                delta = -f / df;
            } else if (it > 0 && std::isfinite(f_prev)) {
                // Secant fallback
                Real denom = (f - f_prev);
                if (std::abs(denom) < Real(1e-14)) denom = (denom >= 0 ? Real(1e-14) : Real(-1e-14));
                delta = -f * (m - m_prev) / denom;
            } else {
                // Gentle nudge if derivative is unusable
                delta = (f > 0 ? Real(-1e-2) : Real(1e-2));
            }

            // Damp to avoid overshoot
            delta = std::clamp(delta, Real(-0.25), Real(0.25));

            m_prev = m;
            f_prev = f;
            m += delta;
            m = std::clamp(m, eps, Real(1) - eps);

            if (std::abs(delta) < tolM * std::max(Real(1), std::abs(m))) break;
        }

        // Finalize parameters and kinematics
        K = Elliptic::ellipK(m);
        E = Elliptic::ellipE(m);
        k = Real(M_PI) / (K * h);   // pseudo-dispersion for KdV-type cnoidal
        omega = Real(2 * M_PI) / T; // angular frequency
        c = omega / k;              // phase speed
    }
};
