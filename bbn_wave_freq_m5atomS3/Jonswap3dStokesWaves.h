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
          // Initialize all mutable members
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

        // Initialize trig cache
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
            for(int i=0;i<N_FREQ;++i)
                exp_kz_freq_cache_[i] = std::exp(k_(i) * z);

            #pragma omp parallel for
            for(size_t idx=0; idx<pairwise_size_; ++idx)
                exp_kz_pair_cache_[idx] = std::exp(k_sum_flat_[idx] * z);

            #pragma omp parallel for
            for(size_t idx=0; idx<pairwise_size_; ++idx){
                skip_pair_mask_[idx] = (cutoff_tol_>0.0 && std::abs(Bij_flat_[idx])*exp_kz_pair_cache_[idx]<cutoff_tol_)?1:0;
            }

            exp_kz_cached_z_ = z;
            exp_kz_cached_z_flag_ = true;
            stokes_drift_mean_xy_cache_z_flag_ = false;
        }

        // First-order theta & trig
        for(int i=0;i<N_FREQ;++i)
            theta0_(i) = kx_(i)*x + ky_(i)*y + phi_(i);

        #pragma omp parallel for
        for(int i=0;i<N_FREQ;++i)
            fast_sincos(theta0_(i) - omega_(i)*t, sin0_(i), cos0_(i));

        for(int i=0;i<N_FREQ;++i){
            sin0_(i) *= exp_kz_freq_cache_[i];
            cos0_(i) *= exp_kz_freq_cache_[i];
        }

        // --- REPLACED: First-order contributions (thread-local long double + Kahan combine) ---
        struct AccLong { long double x=0.0L, y=0.0L, z=0.0L; };

#if defined(_OPENMP)
        int nthreads = omp_get_max_threads();
#else
        int nthreads = 1;
#endif
        std::vector<AccLong> disp_p(nthreads), vel_p(nthreads), acc_p(nthreads);

        #pragma omp parallel
        {
            int tid = 0;
    #ifdef _OPENMP
            tid = omp_get_thread_num();
    #endif
            AccLong ld_disp{0,0,0}, ld_vel{0,0,0}, ld_acc{0,0,0};
            #pragma omp for nowait
            for(int i=0;i<N_FREQ;++i){
                double Ai=A_(i), wi=omega_(i), dirx=dir_x_(i), diry=dir_y_(i);
                double s=sin0_(i), c=cos0_(i);
                // accumulate in long double locally
                ld_disp.x -= (long double)Ai * (long double)c * (long double)dirx;
                ld_disp.y -= (long double)Ai * (long double)c * (long double)diry;
                ld_disp.z += (long double)Ai * (long double)s;
                ld_vel.x += (long double)Ai * (long double)wi * (long double)s * (long double)dirx;
                ld_vel.y += (long double)Ai * (long double)wi * (long double)s * (long double)diry;
                ld_vel.z += (long double)Ai * (long double)wi * (long double)c;
                ld_acc.x += (long double)Ai * (long double)wi * (long double)wi * (long double)c * (long double)dirx;
                ld_acc.y += (long double)Ai * (long double)wi * (long double)wi * (long double)c * (long double)diry;
                ld_acc.z -= (long double)Ai * (long double)wi * (long double)wi * (long double)s;
            }
            // store per-thread partials
            disp_p[tid].x = ld_disp.x; disp_p[tid].y = ld_disp.y; disp_p[tid].z = ld_disp.z;
            vel_p[tid].x  = ld_vel.x;  vel_p[tid].y  = ld_vel.y;  vel_p[tid].z  = ld_vel.z;
            acc_p[tid].x  = ld_acc.x;  acc_p[tid].y  = ld_acc.y;  acc_p[tid].z  = ld_acc.z;
        }

        // Combine per-thread partials using Kahan-style compensated summation (serial)
        auto kahan_combine = [](const std::vector<AccLong>& parts) {
            AccLong sum{0,0,0}, c{0,0,0};
            for (const auto &p : parts) {
                long double yx = p.x - c.x;
                long double tx = sum.x + yx;
                c.x = (tx - sum.x) - yx;
                sum.x = tx;

                long double yy = p.y - c.y;
                long double ty = sum.y + yy;
                c.y = (ty - sum.y) - yy;
                sum.y = ty;

                long double yz = p.z - c.z;
                long double tz = sum.z + yz;
                c.z = (tz - sum.z) - yz;
                sum.z = tz;
            }
            return sum;
        };

        AccLong sum_disp = kahan_combine(disp_p);
        AccLong sum_vel  = kahan_combine(vel_p);
        AccLong sum_acc  = kahan_combine(acc_p);

        Eigen::Vector3d disp_first((double)sum_disp.x, (double)sum_disp.y, (double)sum_disp.z),
                        vel_first((double)sum_vel.x,   (double)sum_vel.y,   (double)sum_vel.z),
                        acc_first((double)sum_acc.x,   (double)sum_acc.y,   (double)sum_acc.z);
        disp += disp_first; vel += vel_first; acc += acc_first;
        // --- END FIRST-ORDER PATCH ---

        // Second-order theta caching
        if(std::isnan(theta2_cached_x_) || std::isnan(theta2_cached_y_) ||
           theta2_cached_x_ != x || theta2_cached_y_ != y)
        {
            #pragma omp parallel for
            for(size_t idx=0; idx<pairwise_size_; ++idx)
                theta2_cache_[idx] = kx_sum_flat_[idx]*x + ky_sum_flat_[idx]*y + phi_sum_flat_[idx];
            theta2_cached_x_ = x;
            theta2_cached_y_ = y;
        }

        // Trig cache second-order
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

        // --- REPLACED: Second-order contributions (thread-local long double + Kahan combine) ---
        std::vector<AccLong> disp2_p(nthreads), vel2_p(nthreads), acc2_p(nthreads);

        #pragma omp parallel
        {
            int tid = 0;
    #ifdef _OPENMP
            tid = omp_get_thread_num();
    #endif
            AccLong ld_disp{0,0,0}, ld_vel{0,0,0}, ld_acc{0,0,0};

            #pragma omp for nowait
            for(size_t idx=0; idx<pairwise_size_; ++idx){
                if(skip_pair_mask_[idx]) continue;
                double coeff=(double)factor_flat_[idx]*exp_kz_pair_cache_[idx];
                double Bij=Bij_flat_[idx], omega_sum=omega_sum_flat_[idx];
                double ksum=k_sum_flat_[idx], kxsum=kx_sum_flat_[idx], kysum=ky_sum_flat_[idx];
                double cos2=trig_cache_.cos_second(idx), sin2=trig_cache_.sin_second(idx);
                double hx=(ksum>1e-18)? kxsum/ksum : 0.0;
                double hy=(ksum>1e-18)? kysum/ksum : 0.0;
                double omega2=omega_sum*omega_sum;
                // accumulate in long double
                ld_disp.z += (long double)coeff * (long double)Bij * (long double)cos2;
                ld_disp.x += -(long double)coeff * (long double)Bij * (long double)cos2 * (long double)hx;
                ld_disp.y += -(long double)coeff * (long double)Bij * (long double)cos2 * (long double)hy;
                ld_vel.z += -(long double)coeff * (long double)omega_sum * (long double)Bij * (long double)sin2;
                ld_vel.x += -(long double)coeff * (long double)omega_sum * (long double)Bij * (long double)sin2 * (long double)hx;
                ld_vel.y += -(long double)coeff * (long double)omega_sum * (long double)Bij * (long double)sin2 * (long double)hy;
                ld_acc.z += (long double)coeff * (long double)Bij * (long double)omega2 * (long double)cos2;
                ld_acc.x += (long double)coeff * (long double)Bij * (long double)omega2 * (long double)cos2 * (long double)hx;
                ld_acc.y += (long double)coeff * (long double)Bij * (long double)omega2 * (long double)cos2 * (long double)hy;
            }
            disp2_p[tid].x = ld_disp.x; disp2_p[tid].y = ld_disp.y; disp2_p[tid].z = ld_disp.z;
            vel2_p[tid].x  = ld_vel.x;  vel2_p[tid].y  = ld_vel.y;  vel2_p[tid].z  = ld_vel.z;
            acc2_p[tid].x  = ld_acc.x;  acc2_p[tid].y  = ld_acc.y;  acc2_p[tid].z  = ld_acc.z;
        }

        AccLong sum_disp2 = kahan_combine(disp2_p);
        AccLong sum_vel2  = kahan_combine(vel2_p);
        AccLong sum_acc2  = kahan_combine(acc2_p);

        Eigen::Vector3d disp_second((double)sum_disp2.x, (double)sum_disp2.y, (double)sum_disp2.z),
                        vel_second((double)sum_vel2.x,   (double)sum_vel2.y,   (double)sum_vel2.z),
                        acc_second((double)sum_acc2.x,   (double)sum_acc2.y,   (double)sum_acc2.z);
        disp += disp_second; vel += vel_second; acc += acc_second;
        // --- END SECOND-ORDER PATCH ---

        // Depth-dependent Stokes drift
        if(!stokes_drift_mean_xy_cache_z_flag_){
            // --- REPLACED: Stokes drift accumulation (thread-local long double + Kahan combine) ---
            std::vector<std::pair<long double,long double>> stokes_p(nthreads, {0.0L,0.0L});

            #pragma omp parallel
            {
                int tid = 0;
    #ifdef _OPENMP
                tid = omp_get_thread_num();
    #endif
                long double lx = 0.0L, ly = 0.0L;
                #pragma omp for nowait
                for(int i=0;i<N_FREQ;++i){
                    double exp2 = exp_kz_freq_cache_[i]*exp_kz_freq_cache_[i];
                    double Usi_z = stokes_drift_scalar_(i)*exp2;
                    lx += (long double)Usi_z * (long double)dir_x_(i);
                    ly += (long double)Usi_z * (long double)dir_y_(i);
                }
                stokes_p[tid].first = lx;
                stokes_p[tid].second = ly;
            }

            // Kahan combine stokes components
            long double stx = 0.0L, cstx = 0.0L;
            long double sty = 0.0L, csty = 0.0L;
            for (int tt = 0; tt < nthreads; ++tt) {
                long double yx = stokes_p[tt].first - cstx;
                long double tx = stx + yx;
                cstx = (tx - stx) - yx;
                stx = tx;

                long double yy = stokes_p[tt].second - csty;
                long double ty = sty + yy;
                csty = (ty - sty) - yy;
                sty = ty;
            }
            stokes_drift_mean_xy_cache_ = Eigen::Vector2d((double)stx, (double)sty);
            stokes_drift_mean_xy_cache_z_flag_ = true;
            // --- END STOKES PATCH ---
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

    // Pairwise arrays with aligned allocator
    std::vector<double, Eigen::aligned_allocator<double>> Bij_flat_, kx_sum_flat_, ky_sum_flat_, 
                                                         k_sum_flat_, omega_sum_flat_, phi_sum_flat_, factor_flat_;
    
    // Mutable caches
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
        for(int i=0;i<N_FREQ;++i) phi_(i)=dist(rng);
    }

    void initializeDirectionalSpread() {
        // Use Eigen array operations for better performance
        Eigen::ArrayXd amplitude_ratio = A_.array() / A_.maxCoeff();
        Eigen::ArrayXd spread_angle = amplitude_ratio.pow(spreading_exponent_);
        
        // Convert back to matrix format
        dir_x_ = spread_angle.cos();
        dir_y_ = spread_angle.sin();
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

    // Parallel over time steps
#pragma omp parallel for
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
