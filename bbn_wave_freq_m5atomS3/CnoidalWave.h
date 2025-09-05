#pragma once
#include <cmath>
#include <stdexcept>

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif

namespace Elliptic {

// === Complete elliptic integral of the first kind (AGM) ===
template<typename Real>
Real ellipK(Real m) {
    if (m < 0 || m > 1) throw std::domain_error("ellipK: m out of range [0,1]");
    if (m == 0) return M_PI/2;
    if (m == 1) return INFINITY;

    Real a = 1.0, b = std::sqrt(1.0 - m), c = std::sqrt(m);
    while (std::abs(c) > 1e-15) {
        Real an = (a + b) / 2;
        Real bn = std::sqrt(a * b);
        c = (a - b) / 2;
        a = an; b = bn;
    }
    return M_PI / (2.0 * a);
}

// === Complete elliptic integral of the second kind (AGM) ===
template<typename Real>
Real ellipE(Real m) {
    if (m < 0 || m > 1) throw std::domain_error("ellipE: m out of range [0,1]");
    if (m == 0) return M_PI/2;
    if (m == 1) return 1.0;

    Real a = 1.0, b = std::sqrt(1.0 - m), c = std::sqrt(m);
    Real sum = 0.0, pow2 = 1.0;
    while (std::abs(c) > 1e-15) {
        Real an = (a + b) / 2;
        Real bn = std::sqrt(a * b);
        c = (a - b) / 2;
        pow2 *= 2;
        sum += pow2 * c * c;
        a = an; b = bn;
    }
    return M_PI/(2.0*a) * (1.0 - sum/2.0);
}

// === Jacobi elliptic sn, cn, dn (simplified) ===
template<typename Real>
void jacobi_sn_cn_dn(Real u, Real m, Real &sn, Real &cn, Real &dn) {
    if (m == 0) { sn = std::sin(u); cn = std::cos(u); dn = 1.0; return; }
    if (m == 1) { sn = std::tanh(u); cn = 1.0/std::cosh(u); dn = cn; return; }
    sn = std::sin(u);
    cn = std::cos(u);
    dn = std::sqrt(1 - m*sn*sn);
}

} // namespace Elliptic


template<typename Real = double>
class CnoidalWave {
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
                Real dir_angle = 0.0,
                Real gravity = Real(9.80665))
      : h(depth), H(height), T(period), theta(dir_angle), g(gravity)
    {
        if (h <= 0 || H <= 0 || T <= 0) throw std::domain_error("Invalid wave parameters");
        solveEllipticParameters();
        cosTheta = std::cos(theta);
        sinTheta = std::sin(theta);
    }

    // Free surface elevation η(x,y,t)
    Real surfaceElevation(Real x, Real y, Real t) const {
        Real s = x*cosTheta + y*sinTheta;
        Real arg = k*(s - c*t);
        Real sn, cn, dn;
        Elliptic::jacobi_sn_cn_dn(arg, m, sn, cn, dn);
        return eta0 + Hc * cn*cn;
    }

    // Vertical velocity (surface only)
    Real wVelocity(Real x, Real y, Real t) const {
        Real s = x*cosTheta + y*sinTheta;
        Real arg = k*(s - c*t);
        Real sn, cn, dn;
        Elliptic::jacobi_sn_cn_dn(arg, m, sn, cn, dn);
        return +2.0 * Hc * cn * sn * k * c;
    }

    // Vertical acceleration (surface only)
    Real azAcceleration(Real x, Real y, Real t) const {
        Real s = x*cosTheta + y*sinTheta;
        Real arg = k*(s - c*t);
        Real sn, cn, dn;
        Elliptic::jacobi_sn_cn_dn(arg, m, sn, cn, dn);
        return -2.0 * Hc * k*k * c*c * (cn*cn - sn*sn);
    }

    // Unified interface for consistency with JONSWAP/PM models
    State getLagrangianState(Real x, Real y, Real t) const {
        State st;
        st.displacement = Eigen::Vector3d(0.0, 0.0, surfaceElevation(x,y,t));
        st.velocity     = Eigen::Vector3d(0.0, 0.0, wVelocity(x,y,t));
        st.acceleration = Eigen::Vector3d(0.0, 0.0, azAcceleration(x,y,t));
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
        // crude solve for modulus m by period match
        m = 0.8;
        for (int iter = 0; iter < 20; ++iter) {
            K = Elliptic::ellipK(m);
            E = Elliptic::ellipE(m);
            Real Tguess = 2*K * std::sqrt(h/(3*g*(1-m)));
            Real f = Tguess - T;
            Real dm = 1e-6;
            Real K2 = Elliptic::ellipK(m+dm);
            Real T2 = 2*K2 * std::sqrt(h/(3*g*(1-(m+dm))));
            Real df = (T2 - Tguess)/dm;
            Real delta = -f/df;
            m += delta;
            if (std::abs(delta) < 1e-12) break;
            if (m <= 0) m = 1e-8;
            if (m >= 1) m = 1 - 1e-8;
        }

        K = Elliptic::ellipK(m);
        E = Elliptic::ellipE(m);

        k = M_PI / (K*h);
        omega = 2*M_PI/T;
        c = omega/k;
        eta0 = 0.0;
        Hc   = H;
    }
};
