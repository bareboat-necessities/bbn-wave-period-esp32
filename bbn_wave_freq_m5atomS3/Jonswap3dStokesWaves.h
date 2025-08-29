#pragma once

/*
  Copyright 2025, Mikhail Grushinskiy

  JONSWAP-spectrum 3D Stokes-corrected waves simulation (surface, deep-water).
  - 1st order: Airy (linear) components
  - 2nd order: simplified deep-water sum-frequency bound harmonics + simple
    estimate of surface Stokes drift (monochromatic approximation)
*/

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
#include <limits>
#include <vector>

#ifndef PI
static constexpr double PI = 3.14159265358979323846264338327950288;
#else
static constexpr double PI = M_PI;
#endif

#ifdef JONSWAP_TEST
#include <iostream>
#include <fstream>
#endif

// Portable fast_sincos helper
inline void fast_sincos(double x, double &s, double &c) {
#if defined(__GNUC__) || defined(__clang__)
    #if defined(__GLIBC__) || defined(_GNU_SOURCE)
        ::sincos(x, &s, &c);
    #else
        s = std::sin(x);
        c = std::cos(x);
    #endif
#elif defined(_MSC_VER)
    #ifdef _MSC_VER
        double cs;
        _sincos(x, &s, &cs);
        c = cs;
    #else
        s = std::sin(x);
        c = std::cos(x);
    #endif
#else
    s = std::sin(x);
    c = std::cos(x);
#endif
}

// JonswapSpectrum
template<int N_FREQ = 256>
class JonswapSpectrum {
public:
    JonswapSpectrum(double Hs, double Tp,
                    double f_min = 0.02, double f_max = 0.8,
                    double gamma = 2.0, double g = 9.81)
        : Hs_(Hs), Tp_(Tp), f_min_(f_min), f_max_(f_max), gamma_(gamma), g_(g)
    {
        if (N_FREQ < 2) throw std::runtime_error("N_FREQ must be >= 2");
        if (!(Hs_ > 0.0)) throw std::runtime_error("Hs must be > 0");
        if (!(Tp_ > 0.0)) throw std::runtime_error("Tp must be > 0");
        if (!(f_min_ > 0.0) || !(f_max_ > f_min_)) throw std::runtime_error("Invalid frequency range");
        if (!(1.0/Tp >= f_min_ && 1.0/Tp <= f_max_)) throw std::runtime_error("1/Tp must be within [f_min, f_max]");

        frequencies_.setZero();
        S_.setZero();
        A_.setZero();
        df_.setZero();

        computeLogFrequencySpacing();
        computeFrequencyIncrements();
        computeJonswapSpectrumFromHs();
    }

    const Eigen::Matrix<double, N_FREQ, 1>& frequencies() const { return frequencies_; }
    const Eigen::Matrix<double, N_FREQ, 1>& spectrum() const { return S_; }
    const Eigen::Matrix<double, N_FREQ, 1>& amplitudes() const { return A_; }
    const Eigen::Matrix<double, N_FREQ, 1>& df() const { return df_; }

    double integratedVariance() const {
        return (S_.cwiseProduct(df_)).sum();
    }

private:
    double Hs_, Tp_, f_min_, f_max_, gamma_, g_;
    Eigen::Matrix<double, N_FREQ, 1> frequencies_, S_, A_, df_;

    void computeLogFrequencySpacing() {
        double log_f_min = std::log(f_min_);
        double log_f_max = std::log(f_max_);
        for (int i = 0; i < N_FREQ; ++i)
            frequencies_(i) = std::exp(log_f_min + (log_f_max - log_f_min) * i / (N_FREQ - 1));
    }

    void computeFrequencyIncrements() {
        if (N_FREQ < 2) { df_.setZero(); return; }
        df_(0) = frequencies_(1) - frequencies_(0);
        for (int i = 1; i < N_FREQ - 1; ++i)
            df_(i) = 0.5 * (frequencies_(i + 1) - frequencies_(i - 1));
        df_(N_FREQ - 1) = frequencies_(N_FREQ - 1) - frequencies_(N_FREQ - 2);
    }

    void computeJonswapSpectrumFromHs() {
        const double fp = 1.0 / Tp_;
        Eigen::Matrix<double, N_FREQ, 1> S0;
        for (int i = 0; i < N_FREQ; ++i) {
            double f = frequencies_(i);
            // avoid std::pow for squares and fourths to be robust with -ffast-math
            double sigma = (f <= fp) ? 0.07 : 0.09;
            double dfreq = f - fp;
            double denom = 2.0 * sigma * sigma * fp * fp;
            double r = std::exp(-(dfreq * dfreq) / denom);

            // compute 1/f^5 as explicit multiplication to avoid pow instabilities
            double f2 = f * f;
            double f4 = f2 * f2;
            double inv_f5 = 1.0 / (f * f4); // 1 / f^5

            // compute exp(-1.25 * (fp/f)^4) using manual multiplications
            double fp2 = fp * fp;
            double ratio2 = fp2 / f2; // (fp/f)^2
            double ratio4 = ratio2 * ratio2; // (fp/f)^4

            double base = (g_ * g_) / ( (2.0 * PI) * (2.0 * PI) * (2.0 * PI) * (2.0 * PI) ) * inv_f5
                          * std::exp(-1.25 * ratio4);

            // use exp(log(gamma) * r) instead of pow for slightly better numerical behavior
            double gamma_r = std::exp(r * std::log(gamma_));
            S0(i) = base * gamma_r;
        }

        double variance_unit = (S0.cwiseProduct(df_)).sum();
        if (!(variance_unit > 0.0)) throw std::runtime_error("JonswapSpectrum: computed zero/negative variance");

        double variance_target = (Hs_ * Hs_) / 16.0;
        double alpha = variance_target / variance_unit;

        S_ = S0 * alpha;
        A_ = (2.0 * S_.cwiseProduct(df_)).cwiseSqrt();

        // Adjust amplitudes to match exact Hs
        double Hs_est = 4.0 * std::sqrt(0.5 * A_.squaredNorm());
        if (Hs_est <= 0.0) throw std::runtime_error("JonswapSpectrum: Hs_est <= 0");
        double rel_err = std::abs(Hs_est - Hs_) / Hs_;
        if (rel_err > 1e-3) {
            A_ *= (Hs_ / Hs_est);
            for (int i = 0; i < N_FREQ; ++i)
                S_(i) = (A_(i) * A_(i)) / (2.0 * df_(i));
        }
    }
};

// Jonswap3dStokesWaves
template<int N_FREQ = 256>
class Jonswap3dStokesWaves {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct WaveState {
        Eigen::Vector3d displacement;
        Eigen::Vector3d velocity;
        Eigen::Vector3d acceleration;
    };

    Jonswap3dStokesWaves(double Hs, double Tp,
                         double mean_direction_deg = 0.0,
                         double f_min = 0.02, double f_max = 0.8,
                         double gamma = 2.0, double g = 9.81,
                         double spreading_exponent = 15.0,
                         unsigned int seed = 239u,
                         double cutoff_tol = 1e-8)
        : Hs_(Hs), Tp_(Tp), mean_dir_rad_(mean_direction_deg*PI/180.0),
          gamma_(gamma), g_(g), spreading_exponent_(spreading_exponent),
          seed_(seed), cutoff_tol_(cutoff_tol),
          pairwise_size_((size_t)N_FREQ*(N_FREQ+1)/2),
          spectrum_(Hs, Tp, f_min, f_max, gamma, g),
          theta2_cache_(pairwise_size_),
          exp_kz_freq_cache_(N_FREQ),
          exp_kz_pair_cache_(pairwise_size_),
          skip_pair_mask_(pairwise_size_, 0),
          theta0_(N_FREQ),
          sin0_(N_FREQ),
          cos0_(N_FREQ),
          exp_kz_cached_z_(std::numeric_limits<double>::quiet_NaN()),
          exp_kz_cached_z_flag_(false),
          theta2_cached_x_(std::numeric_limits<double>::quiet_NaN()),
          theta2_cached_y_(std::numeric_limits<double>::quiet_NaN()),
          stokes_drift_mean_xy_cache_(0.0, 0.0),
          stokes_drift_mean_xy_cache_z_flag_(false)
    {
        frequencies_ = spectrum_.frequencies();
        S_ = spectrum_.spectrum();
        A_ = spectrum_.amplitudes();
        df_ = spectrum_.df();

        omega_ = 2.0 * PI * frequencies_;
        k_ = omega_.array().square() / g_;

        dir_x_.setZero();
        dir_y_.setZero();
        kx_.setZero();
        ky_.setZero();
        phi_.setZero();
        stokes_drift_scalar_.setZero();

        trig_cache_.sin_second.resize(pairwise_size_);
        trig_cache_.cos_second.resize(pairwise_size_);
        trig_cache_.last_t = std::numeric_limits<double>::quiet_NaN();

        initializeRandomPhases();
        initializeDirectionalSpread();
        computeWaveDirectionComponents();
        computePerComponentStokesDriftEstimate();
        precomputePairwise();
        checkSteepness();
    }

    WaveState getLagrangianState(double x, double y, double t, double z=0.0) const {
        Eigen::Vector3d disp = Eigen::Vector3d::Zero();
        Eigen::Vector3d vel  = Eigen::Vector3d::Zero();
        Eigen::Vector3d acc  = Eigen::Vector3d::Zero();

        // Depth-dependent exp(k z)
        if(!exp_kz_cached_z_flag_ || exp_kz_cached_z_ != z){
            for(int i=0;i<N_FREQ;++i) {
                // compute exp(k*z) with long double intermediate to be robust under -ffast-math
                long double kv = static_cast<long double>(k_(i));
                long double zv = static_cast<long double>(z);
                long double ev = std::exp(kv * zv); // calls overload for long double
                exp_kz_freq_cache_[i] = static_cast<double>(ev);
            }

            for(size_t idx=0; idx<pairwise_size_; ++idx) {
                long double kval = static_cast<long double>(k_sum_flat_[idx]);
                long double zv = static_cast<long double>(z);
                long double ev = std::exp(kval * zv);
                exp_kz_pair_cache_[idx] = static_cast<double>(ev);
            }

            for(size_t idx=0; idx<pairwise_size_; ++idx)
                skip_pair_mask_[idx] = (cutoff_tol_>0.0 && std::abs(Bij_flat_[idx])*exp_kz_pair_cache_[idx]<cutoff_tol_)?1:0;

            exp_kz_cached_z_ = z;
            exp_kz_cached_z_flag_ = true;
            stokes_drift_mean_xy_cache_z_flag_ = false;
        }

        // First-order theta & trig
        for(int i=0;i<N_FREQ;++i)
            theta0_(i) = kx_(i)*x + ky_(i)*y + phi_(i);

        for(int i=0;i<N_FREQ;++i)
            fast_sincos(theta0_(i) - omega_(i)*t, sin0_(i), cos0_(i));

        for(int i=0;i<N_FREQ;++i){
            sin0_(i) *= exp_kz_freq_cache_[i];
            cos0_(i) *= exp_kz_freq_cache_[i];
        }

        // --- SERIAL FIRST-ORDER ---
        for(int i=0;i<N_FREQ;++i){
            double Ai=A_(i), wi=omega_(i);
            double dirx=dir_x_(i), diry=dir_y_(i);
            double s=sin0_(i), c=cos0_(i);
            disp.x() -= Ai*c*dirx;
            disp.y() -= Ai*c*diry;
            disp.z() += Ai*s;
            vel.x()  += Ai*wi*s*dirx;
            vel.y()  += Ai*wi*s*diry;
            vel.z()  += Ai*wi*c;
            acc.x()  += Ai*wi*wi*c*dirx;
            acc.y()  += Ai*wi*wi*c*diry;
            acc.z()  -= Ai*wi*wi*s;
        }

        // Second-order theta caching
        if(std::isnan(theta2_cached_x_) || std::isnan(theta2_cached_y_) ||
           theta2_cached_x_ != x || theta2_cached_y_ != y)
        {
            for(size_t idx=0; idx<pairwise_size_; ++idx)
                theta2_cache_[idx] = kx_sum_flat_[idx]*x + ky_sum_flat_[idx]*y + phi_sum_flat_[idx];
            theta2_cached_x_ = x;
            theta2_cached_y_ = y;
        }

        // Trig cache second-order
        if(trig_cache_.last_t != t){
            for(size_t idx=0; idx<pairwise_size_; ++idx){
                if(skip_pair_mask_[idx]){
                    trig_cache_.sin_second(idx)=0.0;
                    trig_cache_.cos_second(idx)=0.0;
                } else {
                    fast_sincos(theta2_cache_[idx]-omega_sum_flat_[idx]*t,
                                trig_cache_.sin_second(idx),
                                trig_cache_.cos_second(idx));
                }
            }
            trig_cache_.last_t = t;
        }

        // --- SERIAL SECOND-ORDER ---
        for(size_t idx=0; idx<pairwise_size_; ++idx){
            if(skip_pair_mask_[idx]) continue;
            double coeff=factor_flat_[idx]*exp_kz_pair_cache_[idx];
            double Bij=Bij_flat_[idx];
            double omega_sum=omega_sum_flat_[idx];
            double ksum=k_sum_flat_[idx], kxsum=kx_sum_flat_[idx], kysum=ky_sum_flat_[idx];
            double cos2=trig_cache_.cos_second(idx), sin2=trig_cache_.sin_second(idx);
            double hx=(ksum>1e-18)? kxsum/ksum : 0.0;
            double hy=(ksum>1e-18)? kysum/ksum : 0.0;
            double omega2=omega_sum*omega_sum;
            // Note: Bij_flat_ stores A_i * A_j and factor_flat_ is 0.5 for diagonal, 1.0 for off-diagonals
            disp.z() += coeff * Bij * cos2;
            disp.x() -= coeff * Bij * cos2 * hx;
            disp.y() -= coeff * Bij * cos2 * hy;
            vel.z() -= coeff * omega_sum * Bij * sin2;
            vel.x() -= coeff * omega_sum * Bij * sin2 * hx;
            vel.y() -= coeff * omega_sum * Bij * sin2 * hy;
            acc.z() += coeff * Bij * omega2 * cos2;
            acc.x() += coeff * Bij * omega2 * cos2 * hx;
            acc.y() += coeff * Bij * omega2 * cos2 * hy;
        }

        // Depth-dependent Stokes drift
        if(!stokes_drift_mean_xy_cache_z_flag_){
            stokes_drift_mean_xy_cache_.setZero();
            for(int i=0;i<N_FREQ;++i){
                double exp2 = exp_kz_freq_cache_[i]*exp_kz_freq_cache_[i];
                double Usi_z = stokes_drift_scalar_(i)*exp2;
                stokes_drift_mean_xy_cache_.x() += Usi_z*dir_x_(i);
                stokes_drift_mean_xy_cache_.y() += Usi_z*dir_y_(i);
            }
            stokes_drift_mean_xy_cache_z_flag_ = true;
        }

        vel.x() += stokes_drift_mean_xy_cache_.x();
        vel.y() += stokes_drift_mean_xy_cache_.y();

        return {disp, vel, acc};
    }

private:
    JonswapSpectrum<N_FREQ> spectrum_;
    double Hs_, Tp_, mean_dir_rad_, gamma_, g_, spreading_exponent_;
    unsigned int seed_;
    double cutoff_tol_;
    size_t pairwise_size_;

    Eigen::Matrix<double, N_FREQ, 1> frequencies_, S_, A_, df_;
    Eigen::Matrix<double, N_FREQ, 1> omega_, k_, phi_;
    Eigen::Matrix<double, N_FREQ, 1> dir_x_, dir_y_, kx_, ky_;
    Eigen::Matrix<double, N_FREQ, 1> stokes_drift_scalar_;

    std::vector<double, Eigen::aligned_allocator<double>> Bij_flat_, kx_sum_flat_, ky_sum_flat_, 
                                                         k_sum_flat_, omega_sum_flat_, phi_sum_flat_, factor_flat_;
    
    mutable std::vector<double> theta2_cache_;
    mutable std::vector<double> exp_kz_freq_cache_;
    mutable std::vector<double> exp_kz_pair_cache_;
    mutable std::vector<char> skip_pair_mask_;
    mutable Eigen::ArrayXd theta0_, sin0_, cos0_;
    mutable double exp_kz_cached_z_;
    mutable bool exp_kz_cached_z_flag_;
    mutable double theta2_cached_x_, theta2_cached_y_;
    mutable Eigen::Vector2d stokes_drift_mean_xy_cache_;
    mutable bool stokes_drift_mean_xy_cache_z_flag_;

    struct TrigCache {
        Eigen::ArrayXd sin_second, cos_second;
        double last_t;
    };
    mutable TrigCache trig_cache_;

    void initializeRandomPhases() {
        std::mt19937 rng(seed_);
        std::uniform_real_distribution<double> dist(0.0, 2.0*PI);

        // Create symmetric paired phases to ensure zero-mean (avoid DC bias under -ffast-math)
        // For even N_FREQ: pair i with N_FREQ-1-i using phi and -phi.
        // For odd N_FREQ: middle element gets its own phase.
        for (int i = 0; i < N_FREQ/2; ++i) {
            double phi = dist(rng);
            phi_(i) = phi;
            phi_(N_FREQ - 1 - i) = -phi;
        }
        if (N_FREQ % 2 == 1) {
            // middle index
            int mid = N_FREQ / 2;
            phi_(mid) = dist(rng);
        }
    }

    void initializeDirectionalSpread() {
        std::mt19937 rng_dir(seed_ + 1234567u);
        std::uniform_real_distribution<double> u01(-1.0, 1.0);
        Eigen::ArrayXd amplitude_ratio = A_.array() / A_.maxCoeff();
        Eigen::ArrayXd spread_scale = amplitude_ratio.pow(spreading_exponent_);
        const double max_spread = PI * 0.5;

        // Generate paired opposite directions so mean direction cancels exactly
        for (int i = 0; i < N_FREQ/2; ++i) {
            double dev = u01(rng_dir);
            double delta = dev * (spread_scale(i) * max_spread);
            double angle = mean_dir_rad_ + delta;
            dir_x_(i) = std::cos(angle);
            dir_y_(i) = std::sin(angle);
            // opposite partner
            int j = N_FREQ - 1 - i;
            dir_x_(j) = -dir_x_(i);
            dir_y_(j) = -dir_y_(i);
        }

        if (N_FREQ % 2 == 1) {
            int mid = N_FREQ / 2;
            double dev = u01(rng_dir);
            double delta = dev * (spread_scale(mid) * max_spread);
            double angle = mean_dir_rad_ + delta;
            dir_x_(mid) = std::cos(angle);
            dir_y_(mid) = std::sin(angle);
        }

        // Normalize directions (should already be unit but keep just in case)
        for (int i = 0; i < N_FREQ; ++i) {
            double norm = std::hypot(dir_x_(i), dir_y_(i));
            if (norm > 0.0) {
                dir_x_(i) /= norm;
                dir_y_(i) /= norm;
            } else {
                dir_x_(i) = std::cos(mean_dir_rad_);
                dir_y_(i) = std::sin(mean_dir_rad_);
            }
        }
    }

    void computeWaveDirectionComponents() {
        kx_ = k_.array() * dir_x_.array();
        ky_ = k_.array() * dir_y_.array();
    }

    void computePerComponentStokesDriftEstimate() {
        stokes_drift_scalar_ = (A_.array().square() * omega_.array()) / 2.0;
    }

    void precomputePairwise() {
        Bij_flat_.resize(pairwise_size_);
        kx_sum_flat_.resize(pairwise_size_);
        ky_sum_flat_.resize(pairwise_size_);
        k_sum_flat_.resize(pairwise_size_);
        omega_sum_flat_.resize(pairwise_size_);
        phi_sum_flat_.resize(pairwise_size_);
        factor_flat_.resize(pairwise_size_);
        
        size_t idx=0;
        for(int i=0;i<N_FREQ;++i){
            for(int j=i;j<N_FREQ;++j,++idx){
                // store Bij as A_i * A_j (no division here) and set factor to 0.5 on diagonal,
                // 1.0 for off-diagonals. This avoids tiny asymmetries due to /2.0 under -ffast-math.
                Bij_flat_[idx] = A_(i) * A_(j);
                kx_sum_flat_[idx] = kx_(i)+kx_(j);
                ky_sum_flat_[idx] = ky_(i)+ky_(j);
                k_sum_flat_[idx] = k_(i)+k_(j);
                omega_sum_flat_[idx] = omega_(i)+omega_(j);
                phi_sum_flat_[idx] = phi_(i)+phi_(j);
                factor_flat_[idx] = (i==j) ? 0.5 : 1.0;
            }
        }
    }

    void checkSteepness() {
        double max_steepness = (k_.array()*A_.array()).maxCoeff();
        if(max_steepness > 0.4)
            throw std::runtime_error("Wave too steep (>0.4), unstable");
    }
};

#ifdef JONSWAP_TEST
void generateWaveJonswapCSV(const std::string& filename,
                            double Hs, double Tp, double mean_dir_deg,
                            double duration = 40.0, double dt = 0.005) {

    constexpr int N_FREQ = 256;
    auto waveModel = std::make_unique<Jonswap3dStokesWaves<N_FREQ>>(Hs, Tp, mean_dir_deg, 0.02, 0.8, 2.0, 9.81, 10.0);

    const int N_time = static_cast<int>(duration / dt) + 1;
    Eigen::ArrayXd time = Eigen::ArrayXd::LinSpaced(N_time, 0.0, duration);

    // Output arrays: 3 x N_time
    Eigen::ArrayXXd disp(3, N_time), vel(3, N_time), acc(3, N_time);

    for(int i = 0; i < N_time; ++i) {
        auto state = waveModel->getLagrangianState(0.0, 0.0, time(i));
        for(int j = 0; j < 3; ++j) {
            disp(j,i) = state.displacement(j);
            vel(j,i)  = state.velocity(j);
            acc(j,i)  = state.acceleration(j);
        }
    }

    std::ofstream file(filename);
    file << "time,disp_x,disp_y,disp_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z\n";
    for (int i = 0; i < N_time; ++i) {
        file << time(i) << ","
             << disp(0,i) << "," << disp(1,i) << "," << disp(2,i) << ","
             << vel(0,i)  << "," << vel(1,i)  << "," << vel(2,i)  << ","
             << acc(0,i)  << "," << acc(1,i)  << "," << acc(2,i)  << "\n";
    }
}

void Jonswap_testWavePatterns() {
    generateWaveJonswapCSV("short_waves_stokes.csv", 0.5, 3.0, 30.0);
    generateWaveJonswapCSV("medium_waves_stokes.csv", 2.0, 7.0, 30.0);
    generateWaveJonswapCSV("long_waves_stokes.csv", 4.0, 12.0, 30.0);
}
#endif
