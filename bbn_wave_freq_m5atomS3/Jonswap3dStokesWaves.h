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

#ifdef JONSWAP_TEST
#include <iostream>
#include <fstream>
#endif

/*
  Copyright 2025, Mikhail Grushinskiy

  JONSWAP-spectrum 3D Stokes-corrected waves simulation (surface, deep-water).
  - 1st order: Airy (linear) components
  - 2nd order: simplified deep-water sum-frequency bound harmonics + simple
    estimate of surface Stokes drift (monochromatic approximation)
*/

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

    // Accessors (fixed-size Eigen vectors)
    const Eigen::Matrix<double, N_FREQ, 1>& frequencies() const { return frequencies_; }
    const Eigen::Matrix<double, N_FREQ, 1>& spectrum() const { return S_; }
    const Eigen::Matrix<double, N_FREQ, 1>& amplitudes() const { return A_; }
    const Eigen::Matrix<double, N_FREQ, 1>& df() const { return df_; }

    // Diagnostics
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
        if (N_FREQ < 2) {
            df_.setZero();
            return;
        }
        df_(0) = frequencies_(1) - frequencies_(0); // forward diff
        for (int i = 1; i < N_FREQ - 1; ++i) {       // central diff
            df_(i) = 0.5 * (frequencies_(i + 1) - frequencies_(i - 1));
        }
        df_(N_FREQ - 1) = frequencies_(N_FREQ - 1) - frequencies_(N_FREQ - 2); // backward diff
    }

    void computeJonswapSpectrumFromHs() {
        const double fp = 1.0 / Tp_;
        Eigen::Matrix<double, N_FREQ, 1> S0;
        for (int i = 0; i < N_FREQ; ++i) {
            double f = frequencies_(i);
            double sigma = (f <= fp) ? 0.07 : 0.09;
            double r = std::exp(-std::pow(f - fp, 2) / (2.0 * sigma * sigma * fp * fp));
            double base = (g_ * g_) / std::pow(2.0 * M_PI, 4.0) * std::pow(f, -5.0)
                          * std::exp(-1.25 * std::pow(fp / f, 4.0));
            S0(i) = base * std::pow(gamma_, r);
        }

        double variance_unit = (S0.cwiseProduct(df_)).sum();
        if (!(variance_unit > 0.0)) throw std::runtime_error("JonswapSpectrum: computed zero/negative variance (check frequency grid)");

        double variance_target = (Hs_ * Hs_) / 16.0;
        double alpha = variance_target / variance_unit;

        S_ = S0 * alpha;
        A_ = (2.0 * S_.cwiseProduct(df_)).cwiseSqrt();

        // sanity: tiny relative mismatch should be corrected
        double Hs_est = 4.0 * std::sqrt(0.5 * A_.squaredNorm());
        if (Hs_est <= 0.0) throw std::runtime_error("JonswapSpectrum: Hs_est <= 0 after amplitude computation");
        double rel_err = std::abs(Hs_est - Hs_) / Hs_;
        if (rel_err > 1e-3) {
            // correct numerical mismatch
            A_ *= (Hs_ / Hs_est);
            // recompute S from A for internal consistency
            for (int i = 0; i < N_FREQ; ++i) {
                double dfi = df_(i) > 0.0 ? df_(i) : 1e-12;
                S_(i) = (A_(i) * A_(i)) / (2.0 * dfi);
            }
        }
    }
};

// Jonswap3dStokesWaves
template<int N_FREQ = 256>
class Jonswap3dStokesWaves {
public:
    struct WaveState {
        Eigen::Vector3d displacement;
        Eigen::Vector3d velocity;
        Eigen::Vector3d acceleration;
    };

    static constexpr size_t N_PAIRWISE = N_FREQ*(N_FREQ+1)/2;

    Jonswap3dStokesWaves(double Hs, double Tp,
                         double mean_direction_deg = 0.0,
                         double f_min = 0.02, double f_max = 0.8,
                         double gamma = 2.0, double g = 9.81,
                         double spreading_exponent = 15.0,
                         unsigned int seed = 239u)
        : Hs_(Hs), Tp_(Tp), mean_dir_rad_(mean_direction_deg*M_PI/180.0),
          gamma_(gamma), g_(g), spreading_exponent_(spreading_exponent),
          seed_(seed), spectrum_(Hs,Tp,f_min,f_max,gamma,g),
          exp_kz_cached_z_(std::numeric_limits<double>::quiet_NaN())
    {
        // Copy spectrum
        frequencies_ = spectrum_.frequencies();
        S_ = spectrum_.spectrum();
        A_ = spectrum_.amplitudes();
        df_ = spectrum_.df();
        omega_ = 2.0*M_PI*frequencies_;
        k_ = omega_.array().square() / g_;

        phi_.setZero(); dir_x_.setZero(); dir_y_.setZero();
        kx_.setZero(); ky_.setZero();
        stokes_drift_scalar_.setZero(); stokes_drift_mean_xy_.setZero();
        exp_kz_cache_.setConstant(std::numeric_limits<double>::quiet_NaN());

        Bij_flat_.resize(N_PAIRWISE); kx_sum_flat_.resize(N_PAIRWISE);
        ky_sum_flat_.resize(N_PAIRWISE); k_sum_flat_.resize(N_PAIRWISE);
        omega_sum_flat_.resize(N_PAIRWISE); phi_sum_flat_.resize(N_PAIRWISE);
        factor_flat_.resize(N_PAIRWISE);

        th2_.resize(N_PAIRWISE); cos_th2_.resize(N_PAIRWISE); sin_th2_.resize(N_PAIRWISE);
        ksum_safe_.resize(N_PAIRWISE); factor_omega_.resize(N_PAIRWISE); factor_omega2_.resize(N_PAIRWISE);

        th_.setZero(); cos_th_.setZero(); sin_th_.setZero(); exp_z_.setZero();

        initializeRandomPhases();
        initializeDirectionalSpread();
        computeWaveDirectionComponents();
        computePerComponentStokesDriftEstimate();
        precomputePairwiseVectorized();
        checkSteepness();
    }

    WaveState getLagrangianState(double x0,double y0,double t,double z=0.0) const {
        return {evaluateDisplacement(x0,y0,t,z),
                evaluateVelocity(x0,y0,t,z),
                evaluateAcceleration(x0,y0,t,z)};
    }

private:
    double Hs_, Tp_, mean_dir_rad_, gamma_, g_, spreading_exponent_;
    unsigned int seed_;
    JonswapSpectrum<N_FREQ> spectrum_;

    Eigen::Matrix<double,N_FREQ,1> frequencies_, S_, A_, df_;
    Eigen::Matrix<double,N_FREQ,1> omega_, k_, phi_;
    Eigen::Matrix<double,N_FREQ,1> dir_x_, dir_y_, kx_, ky_;
    Eigen::Matrix<double,N_FREQ,1> stokes_drift_scalar_;
    Eigen::Vector2d stokes_drift_mean_xy_;

    mutable Eigen::Array<double,N_FREQ,1> exp_kz_cache_;
    mutable double exp_kz_cached_z_;

    Eigen::ArrayXd Bij_flat_, kx_sum_flat_, ky_sum_flat_, k_sum_flat_;
    Eigen::ArrayXd omega_sum_flat_, phi_sum_flat_, factor_flat_;
    mutable Eigen::ArrayXd th2_, cos_th2_, sin_th2_, ksum_safe_, factor_omega_, factor_omega2_;
    mutable Eigen::Array<double,N_FREQ,1> th_, cos_th_, sin_th_, exp_z_;

    void initializeRandomPhases() {
        std::mt19937 gen(seed_);
        std::uniform_real_distribution<double> dist(0.0, 2*M_PI);
        for(int i=0;i<N_FREQ;++i) phi_(i) = dist(gen);
    }

    void initializeDirectionalSpread() {
        std::mt19937 gen(seed_+1);
        std::uniform_real_distribution<double> u_dist(0.0,2*M_PI);
        std::uniform_real_distribution<double> y_dist(0.0,1.0);
        for(int i=0;i<N_FREQ;++i){
            double theta=0.0;
            while(true){
                double candidate = u_dist(gen);
                double clamped = std::max(0.0,std::cos(candidate-mean_dir_rad_));
                if(y_dist(gen)<=std::pow(clamped,spreading_exponent_)){theta=candidate; break;}
            }
            dir_x_(i)=std::cos(theta); dir_y_(i)=std::sin(theta);
        }
    }

    void computeWaveDirectionComponents() { kx_ = k_.array()*dir_x_.array(); ky_ = k_.array()*dir_y_.array(); }

    void computePerComponentStokesDriftEstimate() {
        stokes_drift_scalar_.setZero(); stokes_drift_mean_xy_.setZero();
        for(int i=0;i<N_FREQ;++i){
            double Usi = 0.5*A_(i)*A_(i)*k_(i)*omega_(i);
            stokes_drift_scalar_(i) = Usi;
            stokes_drift_mean_xy_.x() += Usi*dir_x_(i);
            stokes_drift_mean_xy_.y() += Usi*dir_y_(i);
        }
    }

    void precomputePairwiseVectorized() {
        // Create meshgrid arrays
        Eigen::ArrayXXd kx_i = kx_.replicate(1,N_FREQ);
        Eigen::ArrayXXd kx_j = kx_.transpose().replicate(N_FREQ,1);
        Eigen::ArrayXXd ky_i = ky_.replicate(1,N_FREQ);
        Eigen::ArrayXXd ky_j = ky_.transpose().replicate(N_FREQ,1);
        Eigen::ArrayXXd omega_i = omega_.replicate(1,N_FREQ);
        Eigen::ArrayXXd omega_j = omega_.transpose().replicate(N_FREQ,1);
        Eigen::ArrayXXd phi_i = phi_.replicate(1,N_FREQ);
        Eigen::ArrayXXd phi_j = phi_.transpose().replicate(N_FREQ,1);
        Eigen::ArrayXXd A_i = A_.replicate(1,N_FREQ);
        Eigen::ArrayXXd A_j = A_.transpose().replicate(N_FREQ,1);

        Eigen::ArrayXXd kx_sum = kx_i + kx_j;
        Eigen::ArrayXXd ky_sum = ky_i + ky_j;
        Eigen::ArrayXXd ksum = (kx_sum.square() + ky_sum.square()).sqrt();
        Eigen::ArrayXXd kdot = kx_i*kx_j + ky_i*ky_j;
        Eigen::ArrayXXd Bij = kdot/(2.0*g_) * (A_i*A_j);
        Eigen::ArrayXXd omega_sum = omega_i + omega_j;
        Eigen::ArrayXXd phi_sum = phi_i + phi_j;
        Eigen::ArrayXXd factor = (kx_i==kx_j && ky_i==ky_j).select(1.0,2.0);

        // Flatten upper triangle
        size_t idx=0;
        for(int i=0;i<N_FREQ;++i){
            for(int j=i;j<N_FREQ;++j){
                Bij_flat_(idx) = Bij(i,j);
                kx_sum_flat_(idx) = kx_sum(i,j);
                ky_sum_flat_(idx) = ky_sum(i,j);
                k_sum_flat_(idx) = ksum(i,j);
                omega_sum_flat_(idx) = omega_sum(i,j);
                phi_sum_flat_(idx) = phi_sum(i,j);
                factor_flat_(idx) = factor(i,j);
                ++idx;
            }
        }
    }

    void checkSteepness() const { if((A_.array()*k_.array()).maxCoeff()>0.2) throw std::runtime_error("Wave steepness >0.2"); }

    void ensureExpKzCached(double z) const {
        if(!std::isfinite(exp_kz_cached_z_) || std::abs(exp_kz_cached_z_-z)>1e-12){
            exp_kz_cache_ = (-k_.array()*z).exp();
            exp_kz_cached_z_ = z;
        }
    }

    Eigen::Vector3d evaluateDisplacement(double x,double y,double t,double z) const {
        ensureExpKzCached(z);
        th_ = kx_*x + ky_*y - omega_*t + phi_;
        cos_th_ = th_.cos(); sin_th_ = th_.sin(); exp_z_ = exp_kz_cache_;
        double dx = (-A_.array()*cos_th_*dir_x_.array()*exp_z_).sum();
        double dy = (-A_.array()*cos_th_*dir_y_.array()*exp_z_).sum();
        double dz = (A_.array()*sin_th_*exp_z_).sum();

        th2_ = kx_sum_flat_*x + ky_sum_flat_*y - omega_sum_flat_*t + phi_sum_flat_;
        cos_th2_ = th2_.cos();
        ksum_safe_ = k_sum_flat_.cwiseMax(1e-12);

        dx += (factor_flat_ * (-Bij_flat_) * cos_th2_ * kx_sum_flat_ / ksum_safe_).sum();
        dy += (factor_flat_ * (-Bij_flat_) * cos_th2_ * ky_sum_flat_ / ksum_safe_).sum();
        return {dx, dy, dz};
    }

    Eigen::Vector3d evaluateVelocity(double x,double y,double t,double z) const {
        ensureExpKzCached(z);
        th_ = kx_*x + ky_*y - omega_*t + phi_;
        sin_th_ = th_.sin(); cos_th_ = th_.cos(); exp_z_ = exp_kz_cache_;
        Eigen::Array<double,N_FREQ,1> fac = A_.array()*omega_.array()*exp_z_;
        double vx = (fac*sin_th_*dir_x_.array()).sum() + stokes_drift_mean_xy_.x();
        double vy = (fac*sin_th_*dir_y_.array()).sum() + stokes_drift_mean_xy_.y();
        double vz = (fac*cos_th_).sum();

        th2_ = kx_sum_flat_*x + ky_sum_flat_*y - omega_sum_flat_*t + phi_sum_flat_;
        sin_th2_ = th2_.sin();
        ksum_safe_ = k_sum_flat_.cwiseMax(1e-12);
        factor_omega_ = factor_flat_ * (-Bij_flat_) * omega_sum_flat_.array();
        vx += (factor_omega_*sin_th2_*kx_sum_flat_/ksum_safe_).sum();
        vy += (factor_omega_*sin_th2_*ky_sum_flat_/ksum_safe_).sum();
        vz += (factor_omega_*sin_th2_).sum();
        return {vx,vy,vz};
    }

    Eigen::Vector3d evaluateAcceleration(double x,double y,double t,double z) const {
        ensureExpKzCached(z);
        th_ = kx_*x + ky_*y - omega_*t + phi_;
        sin_th_ = th_.sin(); cos_th_ = th_.cos(); exp_z_ = exp_kz_cache_;
        Eigen::Array<double,N_FREQ,1> fac = A_.array()*omega_.array().square()*exp_z_;
        double ax = (fac*cos_th_*dir_x_.array()).sum();
        double ay = (fac*cos_th_*dir_y_.array()).sum();
        double az = (-fac*sin_th_).sum();

        th2_ = kx_sum_flat_*x + ky_sum_flat_*y - omega_sum_flat_*t + phi_sum_flat_;
        cos_th2_ = th2_.cos();
        ksum_safe_ = k_sum_flat_.cwiseMax(1e-12);
        factor_omega2_ = factor_flat_ * Bij_flat_ * omega_sum_flat_.array().square();
        ax += (factor_omega2_*cos_th2_*kx_sum_flat_/ksum_safe_).sum();
        ay += (factor_omega2_*cos_th2_*ky_sum_flat_/ksum_safe_).sum();
        az += (factor_omega2_*cos_th2_).sum();
        return {ax,ay,az};
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
