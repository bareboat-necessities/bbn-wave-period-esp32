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

#ifdef _OPENMP
#include <omp.h>
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

// -------------------- JonswapSpectrum --------------------
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
            double sigma = (f <= fp) ? 0.07 : 0.09;
            double r = std::exp(-std::pow(f - fp, 2) / (2.0 * sigma * sigma * fp * fp));
            double base = (g_ * g_) / std::pow(2.0 * PI, 4.0) * std::pow(f, -5.0)
                          * std::exp(-1.25 * std::pow(fp / f, 4.0));
            S0(i) = base * std::pow(gamma_, r);
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

// -------------------- Jonswap3dStokesWaves --------------------
template<int N_FREQ = 256>
class Jonswap3dStokesWaves {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct WaveState {
        Eigen::Vector3d displacement;
        Eigen::Vector3d velocity;
        Eigen::Vector3d acceleration;
    };

    // Constructor
    Jonswap3dStokesWaves(double Hs, double Tp,
                         double mean_direction_deg = 0.0,
                         double f_min = 0.02, double f_max = 0.8,
                         double gamma = 2.0, double g = 9.81,
                         double spreading_exponent = 15.0,
                         unsigned int seed = 239u,
                         double cutoff_tol = 1e-6)
        : Hs_(Hs), Tp_(Tp), mean_dir_rad_(mean_direction_deg*PI/180.0),
          gamma_(gamma), g_(g), spreading_exponent_(spreading_exponent),
          seed_(seed), pairwise_size_((size_t)N_FREQ*(N_FREQ+1)/2),
          spectrum_(Hs, Tp, f_min, f_max, gamma, g),
          theta2_cache_(pairwise_size_, std::numeric_limits<double>::quiet_NaN()),
          exp_kz_freq_cache_(N_FREQ, std::numeric_limits<double>::quiet_NaN()),
          exp_kz_pair_cache_(pairwise_size_, std::numeric_limits<double>::quiet_NaN()),
          skip_pair_mask_(pairwise_size_, 0),
          exp_kz_cached_z_(std::numeric_limits<double>::quiet_NaN()),
          exp_kz_cached_z_flag_(false),
          theta2_cached_x_(std::numeric_limits<double>::quiet_NaN()),
          theta2_cached_y_(std::numeric_limits<double>::quiet_NaN()),
          cutoff_tol_(cutoff_tol),
          stokes_drift_mean_xy_cache_(0.0,0.0),
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

        initializeRandomPhases();
        initializeDirectionalSpread();
        computeWaveDirectionComponents();
        computePerComponentStokesDriftEstimate();
        precomputePairwise();
        checkSteepness();

        trig_cache_.sin_first.resize(N_FREQ);
        trig_cache_.cos_first.resize(N_FREQ);
        trig_cache_.sin_second.resize(static_cast<int>(pairwise_size_));
        trig_cache_.cos_second.resize(static_cast<int>(pairwise_size_));
    }

    // -------------------- Multithreaded Lagrangian evaluation --------------------
    WaveState getLagrangianState(double x, double y, double t, double z=0.0) const {
        Eigen::Vector3d disp = Eigen::Vector3d::Zero();
        Eigen::Vector3d vel  = Eigen::Vector3d::Zero();
        Eigen::Vector3d acc  = Eigen::Vector3d::Zero();

        // Depth-dependent exp(k z)
        if(!exp_kz_cached_z_flag_ || exp_kz_cached_z_ != z){
            exp_kz_freq_cache_ = (k_ * z).array().exp();
            exp_kz_pair_cache_ = (k_sum_flat_ * z).array().exp();

#pragma omp parallel for
            for(size_t idx=0; idx<pairwise_size_; ++idx){
                skip_pair_mask_[idx] = (cutoff_tol_>0.0 && std::abs(Bij_flat_[idx])*exp_kz_pair_cache_[idx]<cutoff_tol_)?1:0;
            }

            exp_kz_cached_z_ = z;
            exp_kz_cached_z_flag_ = true;
            stokes_drift_mean_xy_cache_z_flag_ = false;
        }

        // --- First-order theta ---
        Eigen::ArrayXd theta0 = kx_.array()*x + ky_.array()*y + phi_.array();
        Eigen::ArrayXd sin0(N_FREQ), cos0(N_FREQ);

#pragma omp parallel for
        for(int i=0;i<N_FREQ;++i)
            fast_sincos(theta0(i) - omega_(i)*t, sin0(i), cos0(i));

        sin0 *= exp_kz_freq_cache_;
        cos0 *= exp_kz_freq_cache_;

        // --- First-order contributions ---
        Eigen::Vector3d disp_first=Eigen::Vector3d::Zero(), vel_first=Eigen::Vector3d::Zero(), acc_first=Eigen::Vector3d::Zero();
#pragma omp parallel
        {
            Eigen::Vector3d disp_priv=Eigen::Vector3d::Zero(), vel_priv=Eigen::Vector3d::Zero(), acc_priv=Eigen::Vector3d::Zero();
#pragma omp for nowait
            for(int i=0;i<N_FREQ;++i){
                double Ai=A_(i), wi=omega_(i), dirx=dir_x_(i), diry=dir_y_(i);
                double s=sin0(i), c=cos0(i);
                disp_priv.x() += -Ai*c*dirx;
                disp_priv.y() += -Ai*c*diry;
                disp_priv.z() +=  Ai*s;
                vel_priv.x() += Ai*wi*s*dirx;
                vel_priv.y() += Ai*wi*s*diry;
                vel_priv.z() += Ai*wi*c;
                acc_priv.x() += Ai*wi*wi*c*dirx;
                acc_priv.y() += Ai*wi*wi*c*diry;
                acc_priv.z() += -Ai*wi*wi*s;
            }
#pragma omp critical
            { disp_first+=disp_priv; vel_first+=vel_priv; acc_first+=acc_priv; }
        }
        disp += disp_first; vel += vel_first; acc += acc_first;

        // --- Second-order theta caching ---
        if(std::isnan(theta2_cached_x_) || std::isnan(theta2_cached_y_) ||
           theta2_cached_x_ != x || theta2_cached_y_ != y)
        {
            for(size_t idx=0; idx<pairwise_size_; ++idx)
                theta2_cache_[idx] = kx_sum_flat_[idx]*x + ky_sum_flat_[idx]*y + phi_sum_flat_[idx];
            theta2_cached_x_ = x;
            theta2_cached_y_ = y;
        }

        // --- Trig cache second-order ---
        if(trig_cache_.last_t != t){
#pragma omp parallel for
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

        // --- Second-order contributions ---
        Eigen::Vector3d disp_second=Eigen::Vector3d::Zero(), vel_second=Eigen::Vector3d::Zero(), acc_second=Eigen::Vector3d::Zero();
#pragma omp parallel
        {
            Eigen::Vector3d disp_priv=Eigen::Vector3d::Zero(), vel_priv=Eigen::Vector3d::Zero(), acc_priv=Eigen::Vector3d::Zero();
#pragma omp for
            for(size_t idx=0; idx<pairwise_size_; ++idx){
                if(skip_pair_mask_[idx]) continue;
                double coeff=factor_flat_[idx]*exp_kz_pair_cache_[idx];
                double Bij=Bij_flat_[idx], omega_sum=omega_sum_flat_[idx];
                double ksum=k_sum_flat_[idx], kxsum=kx_sum_flat_[idx], kysum=ky_sum_flat_[idx];
                double cos2=trig_cache_.cos_second(idx), sin2=trig_cache_.sin_second(idx);
                double hx=(ksum>1e-18)? kxsum/ksum : 0.0;
                double hy=(ksum>1e-18)? kysum/ksum : 0.0;
                double omega2=omega_sum*omega_sum;
                disp_priv.z() += coeff*Bij*cos2;
                disp_priv.x() += -coeff*Bij*cos2*hx;
                disp_priv.y() += -coeff*Bij*cos2*hy;
                vel_priv.z() += -coeff*omega_sum*Bij*sin2;
                vel_priv.x() += -coeff*omega_sum*Bij*sin2*hx;
                vel_priv.y() += -coeff*omega_sum*Bij*sin2*hy;
                acc_priv.z() += coeff*Bij*omega2*cos2;
                acc_priv.x() += coeff*Bij*omega2*cos2*hx;
                acc_priv.y() += coeff*Bij*omega2*cos2*hy;
            }
#pragma omp critical
            { disp_second+=disp_priv; vel_second+=vel_priv; acc_second+=acc_priv; }
        }
        disp += disp_second; vel += vel_second; acc += acc_second;

        // --- Depth-dependent Stokes drift ---
        if(!stokes_drift_mean_xy_cache_z_flag_){
            stokes_drift_mean_xy_cache_.setZero();
#pragma omp parallel for reduction(+:stokes_drift_mean_xy_cache_.x(), stokes_drift_mean_xy_cache_.y())
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
    size_t pairwise_size_;
    double cutoff_tol_;

    Eigen::Matrix<double, N_FREQ, 1> frequencies_, S_, A_, df_;
    Eigen::Matrix<double, N_FREQ, 1> omega_, k_, phi_;
    Eigen::Matrix<double, N_FREQ, 1> dir_x_, dir_y_, kx_, ky_;
    Eigen::Matrix<double, N_FREQ, 1> stokes_drift_scalar_;
    mutable Eigen::Vector2d stokes_drift_mean_xy_cache_;
    mutable bool stokes_drift_mean_xy_cache_z_flag_;

    std::vector<double> Bij_flat_, kx_sum_flat_, ky_sum_flat_, k_sum_flat_, omega_sum_flat_, phi_sum_flat_, factor_flat_;
    mutable std::vector<double> theta2_cache_;
    mutable double theta2_cached_x_, theta2_cached_y_;
    mutable std::vector<double> exp_kz_freq_cache_;
    mutable std::vector<double> exp_kz_pair_cache_;
    mutable std::vector<char> skip_pair_mask_;
    mutable double exp_kz_cached_z_;
    mutable bool exp_kz_cached_z_flag_;

    struct TrigCache {
        Eigen::ArrayXd sin_first, cos_first;
        Eigen::ArrayXd sin_second, cos_second;
        double last_t = std::numeric_limits<double>::quiet_NaN();
    };
    mutable TrigCache trig_cache_;

    void initializeRandomPhases() {
        std::mt19937 rng(seed_);
        std::uniform_real_distribution<double> dist(0.0, 2.0*PI);
        for(int i=0;i<N_FREQ;++i) phi_(i)=dist(rng);
    }

    void initializeDirectionalSpread() {
        Eigen::ArrayXd spread_angle(N_FREQ);
        for(int i=0;i<N_FREQ;++i)
            spread_angle(i)=std::pow(A_(i)/A_.maxCoeff(), spreading_exponent_);
        dir_x_ = spread_angle.cos();
        dir_y_ = spread_angle.sin();
    }

    void computeWaveDirectionComponents() {
        kx_ = k_.array() * dir_x_.array();
        ky_ = k_.array() * dir_y_.array();
    }

    void computePerComponentStokesDriftEstimate() {
        stokes_drift_scalar_.setZero();
        for(int i=0;i<N_FREQ;++i) stokes_drift_scalar_(i) = A_(i)*A_(i)*omega_(i)/2.0;
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
                Bij_flat_[idx] = A_(i)*A_(j)/2.0;
                kx_sum_flat_[idx] = kx_(i)+kx_(j);
                ky_sum_flat_[idx] = ky_(i)+ky_(j);
                k_sum_flat_[idx] = k_(i)+k_(j);
                omega_sum_flat_[idx] = omega_(i)+omega_(j);
                phi_sum_flat_[idx] = phi_(i)+phi_(j);
                factor_flat_[idx] = (i==j)?1.0:2.0;
            }
        }
    }

    void checkSteepness() {
        if((k_.array()*A_.array()).maxCoeff() > 0.4)
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

    // Output matrices
    Eigen::ArrayXXd disp(3, N_time), vel(3, N_time), acc(3, N_time);

    // Vectorized over time
    for (int i = 0; i < N_time; ++i) {
        auto state = waveModel->getLagrangianState(0.0, 0.0, time(i));
        disp.col(i) = state.displacement;
        vel.col(i) = state.velocity;
        acc.col(i) = state.acceleration;
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

