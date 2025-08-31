#pragma once

#include <cmath>
#include <stdexcept>

namespace Elliptic {

template<typename Real>
Real ellipK(Real m) {
    if (m < 0 || m > 1) throw std::domain_error("ellipK: m out of range [0,1]");
    if (m == 0) return M_PI/2;
    if (m == 1) return INFINITY;

    Real a = 1.0, b = std::sqrt(1.0 - m), c = std::sqrt(m);
    int n = 0;
    while (std::abs(c) > 1e-15 && n < 50) {
        Real an = (a + b) / 2;
        Real bn = std::sqrt(a * b);
        c = (a - b) / 2;
        a = an; b = bn; n++;
    }
    return M_PI / (2.0 * a);
}

template<typename Real>
Real ellipE(Real m) {
    if (m < 0 || m > 1) throw std::domain_error("ellipE: m out of range [0,1]");
    if (m == 0) return M_PI/2;
    if (m == 1) return 1.0;

    Real a = 1.0, b = std::sqrt(1.0 - m), c = std::sqrt(m);
    Real sum = 0.0, pow2 = 1.0;
    int n = 0;
    while (std::abs(c) > 1e-15 && n < 50) {
        Real an = (a + b) / 2;
        Real bn = std::sqrt(a * b);
        c = (a - b) / 2;
        pow2 *= 2;
        sum += pow2 * c * c;
        a = an; b = bn; n++;
    }
    return M_PI/(2.0*a) * (1.0 - sum/2.0);
}

template<typename Real>
void jacobi_sn_cn_dn(Real u, Real m, Real &sn, Real &cn, Real &dn) {
    if (m < 0 || m > 1) throw std::domain_error("jacobi: m out of range [0,1]");
    if (m == 0) { sn = std::sin(u); cn = std::cos(u); dn = 1.0; return; }
    if (m == 1) { sn = std::tanh(u); cn = 1.0/std::cosh(u); dn = cn; return; }

    const int N = 16;
    Real a[N], c[N];
    a[0] = 1.0; c[0] = std::sqrt(m);
    Real twon = 1.0;
    int i = 0;
    for (; i < N-1; ++i) {
        if (c[i] < 1e-15) break;
        a[i+1] = (a[i] + c[i]) / 2;
        c[i+1] = std::sqrt(a[i] * c[i]);
        twon *= 2;
    }
    Real phi = twon * a[i] * u;
    sn = std::sin(phi);
    cn = std::cos(phi);
    for (int j = i; j >= 0; --j) {
        Real t = (c[j] * sn) / a[j];
        Real denom = 1 + t*t;
        sn = (sn*cn) / denom;
        cn = (cn - sn*t) / denom;
    }
    dn = std::sqrt(1 - m*sn*sn);
}

} // namespace Elliptic

template<typename Real = double>
class CnoidalWave {
public:
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

    Real surfaceElevation(Real x, Real y, Real t) const {
        Real s = x*cosTheta + y*sinTheta;
        Real theta_ = k*(s - c*t);
        Real sn, cn, dn;
        Elliptic::jacobi_sn_cn_dn(theta_, m, sn, cn, dn);
        return eta0 + Hc * cn*cn;
    }

    Real uVelocity(Real x, Real y, Real z, Real t) const {
        Real s = x*cosTheta + y*sinTheta;
        Real theta_ = k*(s - c*t);
        Real sn, cn, dn;
        Elliptic::jacobi_sn_cn_dn(theta_, m, sn, cn, dn);
        return  omega * Hc * sn*cn * cosh(k*(z+h)) / sinh(k*h);
    }

    Real wVelocity(Real x, Real y, Real z, Real t) const {
        Real s = x*cosTheta + y*sinTheta;
        Real theta_ = k*(s - c*t);
        Real sn, cn, dn;
        Elliptic::jacobi_sn_cn_dn(theta_, m, sn, cn, dn);
        return  omega * Hc * sn*cn * sinh(k*(z+h)) / sinh(k*h);
    }

    Real axAcceleration(Real x, Real y, Real z, Real t) const {
        Real s = x*cosTheta + y*sinTheta;
        Real theta_ = k*(s - c*t);
        Real sn, cn, dn;
        Elliptic::jacobi_sn_cn_dn(theta_, m, sn, cn, dn);
        return -omega*omega * Hc * (cn*cn - sn*sn) * cosh(k*(z+h)) / sinh(k*h);
    }

    Real azAcceleration(Real x, Real y, Real z, Real t) const {
        Real s = x*cosTheta + y*sinTheta;
        Real theta_ = k*(s - c*t);
        Real sn, cn, dn;
        Elliptic::jacobi_sn_cn_dn(theta_, m, sn, cn, dn);
        return -omega*omega * Hc * (cn*cn - sn*sn) * sinh(k*(z+h)) / sinh(k*h);
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
        // crude Newton solve for m
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
