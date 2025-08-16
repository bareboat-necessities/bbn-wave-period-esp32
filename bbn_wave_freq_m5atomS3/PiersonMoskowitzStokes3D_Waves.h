#pragma once

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif
#include <random>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>

/*
  Pierson–Moskowitz spectrum + Stokes-N waves (deep water).
  - Spectrum parameterized by (Hs, Tp), normalized to match Hs.
  - Directional spreading: cos^s about mean direction (rejection sampling).
  - Kinematics at the free surface with Stokes harmonics up to ORDER.
*/

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ==============================
// PiersonMoskowitzSpectrum (Hs/Tp normalization, like before)
// ==============================
template<int N_FREQ = 256>
class PiersonMoskowitzSpectrum {
public:
    PiersonMoskowitzSpectrum(double Hs, double Tp,
                             double f_min = 0.02, double f_max = 0.8,
                             double g = 9.81)
        : Hs_(Hs), Tp_(Tp), f_min_(f_min), f_max_(f_max), g_(g)
    {
        if (N_FREQ < 2) throw std::runtime_error("N_FREQ must be >= 2");
        if (!(Hs_ > 0.0)) throw std::runtime_error("Hs must be > 0");
        if (!(Tp_ > 0.0)) throw std::runtime_error("Tp must be > 0");
        if (!(f_min_ > 0.0) || !(f_max_ > f_min_)) throw std::runtime_error("Invalid frequency range");
        if (!((1.0 / Tp_) >= f_min_ && (1.0 / Tp_) <= f_max_))
            throw std::runtime_error("1/Tp must be within [f_min, f_max]");

        frequencies_.setZero(); S_.setZero(); A_.setZero(); df_.setZero();

        computeLogFrequencySpacing();
        computeFrequencyIncrements();
        computePMSpectrumFromHs();
    }

    const Eigen::Matrix<double, N_FREQ, 1>& frequencies() const { return frequencies_; }
    const Eigen::Matrix<double, N_FREQ, 1>& spectrum()    const { return S_; }
    const Eigen::Matrix<double, N_FREQ, 1>& amplitudes()  const { return A_; }
    const Eigen::Matrix<double, N_FREQ, 1>& df()          const { return df_; }

    double integratedVariance() const { return (S_.cwiseProduct(df_)).sum(); }

private:
    double Hs_, Tp_, f_min_, f_max_, g_;
    Eigen::Matrix<double, N_FREQ, 1> frequencies_, S_, A_, df_;

    void computeLogFrequencySpacing() {
        double log_f_min = std::log(f_min_);
        double log_f_max = std::log(f_max_);
        for (int i = 0; i < N_FREQ; ++i) {
            double u = (N_FREQ == 1) ? 0.0 : double(i) / double(N_FREQ - 1);
            frequencies_(i) = std::exp(log_f_min + (log_f_max - log_f_min) * u);
        }
    }

    void computeFrequencyIncrements() {
        if (N_FREQ < 2) { df_.setZero(); return; }
        df_(0) = std::max(1e-12, frequencies_(1) - frequencies_(0));
        for (int i = 1; i < N_FREQ - 1; ++i)
            df_(i) = std::max(1e-12, 0.5 * (frequencies_(i + 1) - frequencies_(i - 1)));
        df_(N_FREQ - 1) = std::max(1e-12, frequencies_(N_FREQ - 1) - frequencies_(N_FREQ - 2));
    }

    void computePMSpectrumFromHs() {
        const double fp = 1.0 / Tp_;
        Eigen::Matrix<double, N_FREQ, 1> S0;

        // Canonical PM (JONSWAP with gamma=1)
        for (int i = 0; i < N_FREQ; ++i) {
            const double f = frequencies_(i);
            const double base = (g_ * g_) / std::pow(2.0 * M_PI, 4.0);
            S0(i) = base * std::pow(f, -5.0) * std::exp(-1.25 * std::pow(fp / f, 4.0));
        }

        const double variance_unit   = (S0.cwiseProduct(df_)).sum();
        if (!(variance_unit > 0.0)) throw std::runtime_error("PM: zero/negative variance (grid?)");

        const double variance_target = (Hs_ * Hs_) / 16.0;
        const double alpha           = variance_target / variance_unit;

        S_ = S0 * alpha;
        A_ = (2.0 * S_.cwiseProduct(df_)).cwiseSqrt();

        // Correct small mismatch
        const double Hs_est = 4.0 * std::sqrt(0.5 * A_.squaredNorm());
        if (Hs_est <= 0.0) throw std::runtime_error("PM: Hs_est <= 0 after amplitude computation");
        const double rel_err = std::abs(Hs_est - Hs_) / Hs_;
        if (rel_err > 1e-3) {
            A_ *= (Hs_ / Hs_est);
            for (int i = 0; i < N_FREQ; ++i) {
                const double dfi = df_(i) > 0.0 ? df_(i) : 1e-12;
                S_(i) = (A_(i) * A_(i)) / (2.0 * dfi);
            }
        }
    }
};

// ==============================
// PMStokesN3dWaves (ORDER = 1..5)
// ==============================
template<int N_FREQ = 256, int ORDER = 3>
class PMStokesN3dWaves {
    static_assert(ORDER >= 1 && ORDER <= 5, "ORDER supported range is 1..5");
public:
    struct WaveState {
        Eigen::Vector3d displacement; // particle displacement (surface)
        Eigen::Vector3d velocity;     // time derivative of displacement
        Eigen::Vector3d acceleration; // second time derivative
    };

    PMStokesN3dWaves(double Hs, double Tp,
                     double mean_direction_deg = 0.0,
                     double f_min = 0.02,
                     double f_max = 0.8,
                     double g = 9.81,
                     double spreading_exponent = 15.0,
                     unsigned int seed = 239u)
        : spectrum_(Hs, Tp, f_min, f_max, g),
          Hs_(Hs), Tp_(Tp),
          mean_dir_rad_(mean_direction_deg * M_PI / 180.0),
          g_(g), spreading_exponent_(spreading_exponent), seed_(seed)
    {
        // Spectrum outputs
        frequencies_ = spectrum_.frequencies();
        A1_          = spectrum_.amplitudes(); // first-order amplitude per bin
        df_          = spectrum_.df();

        // Arrays
        omega_.setZero(); k_.setZero();
        phi_.setZero();
        dir_x_.setZero(); dir_y_.setZero();
        kx_.setZero(); ky_.setZero();

        // Dispersion
        omega_ = 2.0 * M_PI * frequencies_;
        k_     = omega_.array().square() / g_; // deep water: w^2 = g k

        // Randomization and spreading
        initializeRandomPhases();
        initializeDirectionalSpreadRejection();
        computeWaveDirectionComponents();

        // Steepness safety (based on first-order amp)
        checkSteepness();
    }

    // Lagrangian surface state at (x0, y0, t)
    WaveState getLagrangianState(double x0, double y0, double t) const {
        Eigen::Vector3d d = Eigen::Vector3d::Zero();
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        Eigen::Vector3d a = Eigen::Vector3d::Zero();

        for (int i = 0; i < N_FREQ; ++i) {
            const double th  = kx_(i) * x0 + ky_(i) * y0 - omega_(i) * t + phi_(i);
            const double ka  = k_(i) * A1_(i);
            const double w   = omega_(i);

            // Series sums for x/y and z using Stokes coefficients up to ORDER
            double sum_cos = 0.0;      // for horizontal displacement (with sign and projection applied later)
            double sum_sin = 0.0;      // for vertical displacement
            double sum_sin_dt = 0.0;   // for horizontal velocity (d/dt cos = + n w sin)
            double sum_cos_dt = 0.0;   // for vertical   velocity (d/dt sin = + n w cos)
            double sum_cos_ddt = 0.0;  // for horizontal accel (d^2/dt^2 cos = + n^2 w^2 cos)
            double sum_sin_ddt = 0.0;  // for vertical   accel (d^2/dt^2 sin = - n^2 w^2 sin)

            for (int n = 1; n <= ORDER; ++n) {
                const double cn = stokesCoeff(n, ka); // includes a1 * (ka)^(n-1) factor
                const double n_th = n * th;
                const double cos_n = std::cos(n_th);
                const double sin_n = std::sin(n_th);

                // Displacements
                sum_cos += cn * cos_n;  // horizontal uses cos(nθ)
                sum_sin += cn * sin_n;  // vertical   uses sin(nθ)

                // Velocities
                sum_sin_dt += cn * (n * w) * sin_n; // d/dt cos(nθ) = + n w sin(nθ)
                sum_cos_dt += cn * (n * w) * cos_n; // d/dt sin(nθ) = + n w cos(nθ)

                // Accelerations
                sum_cos_ddt += cn * (n * n * w * w) * cos_n;   // d2/dt2 cos(nθ) = +n^2 w^2 cos(nθ)
                sum_sin_ddt -= cn * (n * n * w * w) * sin_n;   // d2/dt2 sin(nθ) = -n^2 w^2 sin(nθ)
            }

            // Project horizontal components along direction
            // Note: sign chosen to match your previous convention (Gerstner: -a cosθ along dir).
            d[0] += -sum_cos * dir_x_(i);
            d[1] += -sum_cos * dir_y_(i);
            d[2] +=  sum_sin;

            v[0] +=  sum_sin_dt * dir_x_(i);
            v[1] +=  sum_sin_dt * dir_y_(i);
            v[2] +=  sum_cos_dt;

            a[0] +=  sum_cos_ddt * dir_x_(i);
            a[1] +=  sum_cos_ddt * dir_y_(i);
            a[2] +=  sum_sin_ddt;
        }

        return {d, v, a};
    }

    // Eulerian at fixed (x, y, t) — same surface formulas here
    WaveState getEulerianState(double x, double y, double t) const {
        return getLagrangianState(x, y, t);
    }

    // frequency, first-order amplitude (a1), direction angle (rad)
    Eigen::Matrix<double, N_FREQ, 3> exportSpectrum() const {
        Eigen::Matrix<double, N_FREQ, 3> result;
        for (int i = 0; i < N_FREQ; ++i) {
            double dir_angle = std::atan2(dir_y_(i), dir_x_(i));
            result(i, 0) = frequencies_(i);
            result(i, 1) = A1_(i);
            result(i, 2) = dir_angle;
        }
        return result;
    }

private:
    // Spectrum
    PiersonMoskowitzSpectrum<N_FREQ> spectrum_;

    // Parameters
    double Hs_, Tp_, mean_dir_rad_, g_, spreading_exponent_;
    unsigned int seed_;

    // Arrays
    Eigen::Matrix<double, N_FREQ, 1> frequencies_, omega_, k_, A1_, phi_, df_;
    Eigen::Matrix<double, N_FREQ, 1> dir_x_, dir_y_, kx_, ky_;

    void initializeRandomPhases() {
        std::mt19937 gen(seed_);
        std::uniform_real_distribution<double> dist(0.0, 2.0 * M_PI);
        for (int i = 0; i < N_FREQ; ++i) phi_(i) = dist(gen);
    }

    // cos^s directional spreading via rejection sampling (same as before)
    void initializeDirectionalSpreadRejection() {
        std::mt19937 gen(seed_ + 1);
        std::uniform_real_distribution<double> u_dist(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<double> y_dist(0.0, 1.0);

        for (int i = 0; i < N_FREQ; ++i) {
            double theta = 0.0;
            while (true) {
                double candidate = u_dist(gen);
                double base = std::cos(candidate - mean_dir_rad_);
                double pdf_val = std::pow(std::clamp(base, 0.0, 1.0), spreading_exponent_);
                if (y_dist(gen) <= pdf_val) { theta = candidate; break; }
            }
            dir_x_(i) = std::cos(theta);
            dir_y_(i) = std::sin(theta);
        }
    }

    void computeWaveDirectionComponents() {
        for (int i = 0; i < N_FREQ; ++i) {
            kx_(i) = k_(i) * dir_x_(i);
            ky_(i) = k_(i) * dir_y_(i);
        }
    }

    // Stokes harmonic coefficient c_n * a1 * (ka)^(n-1)
    // Deep-water coefficients up to 5th order
    static double stokesCoeff(int n, double ka) {
        // base first-order amplitude a1 factor will be multiplied outside, so return c_n * a1 * (ka)^(n-1)
        // We fold a1 in here for simplicity: cn(ka) = c_n * a1 * (ka)^(n-1)
        // c1=1, c2=1/2, c3=3/8, c4=1/3, c5=125/384 (common deep-water set)
        // NOTE: This function expects ka computed with a1 (first-order amplitude).
        static const double c[6] = {0.0, 1.0, 0.5, 3.0/8.0, 1.0/3.0, 125.0/384.0};
        if (n < 1 || n > 5) return 0.0;
        // We will multiply by a1 outside, so return c_n * (ka)^(n-1)
        return c[n] * std::pow(std::max(ka, 0.0), n - 1);
    }

    void checkSteepness() const {
        const double max_ka = (A1_.array() * k_.array()).maxCoeff();
        if (max_ka > 0.20)
            throw std::runtime_error("Stokes: max steepness k*a exceeds ~0.2 (validity warning)");
    }
};
