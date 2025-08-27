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
#pragma once

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>

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
                         double f_min = 0.02,
                         double f_max = 0.8,
                         double gamma = 2.0,
                         double g = 9.81,
                         double spreading_exponent = 15.0,
                         unsigned int seed = 239u)
        : Hs_(Hs), Tp_(Tp), mean_dir_rad_(mean_direction_deg * M_PI / 180.0),
          gamma_(gamma), g_(g), spreading_exponent_(spreading_exponent),
          seed_(seed), exp_kz_cached_z_(std::numeric_limits<double>::quiet_NaN())
    {
        spectrum_ = JonswapSpectrum<N_FREQ>(Hs, Tp, f_min, f_max, gamma, g);
        frequencies_ = spectrum_.frequencies();
        S_ = spectrum_.spectrum();
        A_ = spectrum_.amplitudes();
        df_ = spectrum_.df();

        omega_ = 2.0 * M_PI * frequencies_;
        k_ = omega_.array().square() / g_;

        initializeRandomPhases();
        initializeDirectionalSpread();
        computeWaveDirectionComponents();
        computePerComponentStokesDriftEstimate();
        precomputePairwise();
        checkSteepness();
    }

    WaveState getLagrangianState(double x0, double y0, double t, double z = 0.0) const {
        Eigen::Vector3d disp = evaluateDisplacement(x0, y0, t, z);
        Eigen::Vector3d vel  = evaluateVelocity(x0, y0, t, z);
        Eigen::Vector3d acc  = evaluateAcceleration(x0, y0, t, z);
        return {disp, vel, acc};
    }

    Eigen::Matrix<double, N_FREQ, 4> exportSpectrum(double z = 0.0) const {
        ensureExpKzCached(z);
        Eigen::Matrix<double, N_FREQ, 4> result;
        for (int i = 0; i < N_FREQ; ++i) {
            double dir_angle = std::atan2(dir_y_(i), dir_x_(i));
            result(i, 0) = frequencies_(i);
            result(i, 1) = A_(i);
            result(i, 2) = dir_angle;
            result(i, 3) = exp_kz_cache_[i];
        }
        return result;
    }

private:
    // Member variables
    double Hs_, Tp_, mean_dir_rad_, gamma_, g_, spreading_exponent_;
    unsigned int seed_;
    JonswapSpectrum<N_FREQ> spectrum_;

    Eigen::Matrix<double, N_FREQ, 1> frequencies_, S_, A_, df_;
    Eigen::Matrix<double, N_FREQ, 1> omega_, k_, phi_;
    Eigen::Matrix<double, N_FREQ, 1> dir_x_, dir_y_, kx_, ky_;
    Eigen::Matrix<double, N_FREQ, 1> stokes_drift_scalar_;
    Eigen::Vector2d stokes_drift_mean_xy_;

    // Fixed-size pairwise arrays
    Eigen::Array<double, N_PAIRWISE, 1> Bij_flat_, kx_sum_flat_, ky_sum_flat_, k_sum_flat_, omega_sum_flat_, phi_sum_flat_, factor_flat_;

    mutable Eigen::Array<double, N_FREQ, 1> exp_kz_cache_;
    mutable double exp_kz_cached_z_;

    inline size_t upper_index(int i,int j) const {
        size_t ii = static_cast<size_t>(i);
        size_t N = static_cast<size_t>(N_FREQ);
        return ii*N - ii*(ii-1)/2 + static_cast<size_t>(j-i);
    }

    void initializeRandomPhases() {
        std::mt19937 gen(seed_);
        std::uniform_real_distribution<double> dist(0.0, 2*M_PI);
        for (int i=0;i<N_FREQ;++i) phi_(i)=dist(gen);
    }

    void initializeDirectionalSpread() {
        std::mt19937 gen(seed_+1);
        std::uniform_real_distribution<double> u_dist(0.0, 2*M_PI);
        std::uniform_real_distribution<double> y_dist(0.0,1.0);
        for (int i=0;i<N_FREQ;++i) {
            double theta=0.0;
            while(true){
                double candidate=u_dist(gen);
                double clamped=std::max(0.0,std::cos(candidate-mean_dir_rad_));
                double pdf_val=std::pow(clamped,spreading_exponent_);
                if(y_dist(gen)<=pdf_val){theta=candidate;break;}
            }
            dir_x_(i)=std::cos(theta);
            dir_y_(i)=std::sin(theta);
        }
    }

    void computeWaveDirectionComponents() {
        kx_ = k_.array()*dir_x_.array();
        ky_ = k_.array()*dir_y_.array();
    }

    void computePerComponentStokesDriftEstimate() {
        stokes_drift_scalar_.setZero();
        stokes_drift_mean_xy_.setZero();
        for(int i=0;i<N_FREQ;++i){
            double a=A_(i);
            double ki=k_(i);
            double wi=omega_(i);
            double Usi=0.5*a*a*ki*wi;
            stokes_drift_scalar_(i)=Usi;
            stokes_drift_mean_xy_.x()+=Usi*dir_x_(i);
            stokes_drift_mean_xy_.y()+=Usi*dir_y_(i);
        }
    }

    void precomputePairwise() {
        for(int i=0;i<N_FREQ;++i){
            for(int j=i;j<N_FREQ;++j){
                size_t idx=upper_index(i,j);
                double kx_sum=kx_(i)+kx_(j);
                double ky_sum=ky_(i)+ky_(j);
                double kdot=kx_(i)*kx_(j)+ky_(i)*ky_(j);
                double Bij=(kdot)/(2.0*g_)*(A_(i)*A_(j));
                double omega_sum=omega_(i)+omega_(j);
                double phi_sum=phi_(i)+phi_(j);
                double ksum=std::sqrt(kx_sum*kx_sum+ky_sum*ky_sum);

                Bij_flat_(idx)=Bij;
                kx_sum_flat_(idx)=kx_sum;
                ky_sum_flat_(idx)=ky_sum;
                k_sum_flat_(idx)=ksum;
                omega_sum_flat_(idx)=omega_sum;
                phi_sum_flat_(idx)=phi_sum;
                factor_flat_(idx) = (i==j)?1.0:2.0;
            }
        }
        exp_kz_cache_.setConstant(std::numeric_limits<double>::quiet_NaN());
        exp_kz_cached_z_ = std::numeric_limits<double>::quiet_NaN();
    }

    void ensureExpKzCached(double z) const {
        if(!std::isfinite(exp_kz_cached_z_)||std::abs(exp_kz_cached_z_-z)>1e-12){
            for(int i=0;i<N_FREQ;++i) exp_kz_cache_(i)=std::exp(-k_(i)*z);
            exp_kz_cached_z_=z;
        }
    }

    template<typename F>
    void loopOverPairs(F&& f,double x,double y,double t,double z) const {
        ensureExpKzCached(z);
        for(int i=0;i<N_FREQ;++i){
            for(int j=i;j<N_FREQ;++j){
                size_t idx=upper_index(i,j);
                if(std::abs(Bij_flat_(idx))<1e-18) continue;
                double th = kx_sum_flat_(idx)*x + ky_sum_flat_(idx)*y - omega_sum_flat_(idx)*t + phi_sum_flat_(idx);
                f(idx, Bij_flat_(idx), std::cos(th), std::sin(th), k_sum_flat_(idx),
                  kx_sum_flat_(idx), ky_sum_flat_(idx), omega_sum_flat_(idx), factor_flat_(idx));
            }
        }
    }

    Eigen::Vector3d evaluateDisplacement(double x,double y,double t,double z) const {
        ensureExpKzCached(z);
        Eigen::Array<double, N_FREQ, 1> th = (kx_*x + ky_*y - omega_*t + phi_).array();
        Eigen::Array<double, N_FREQ, 1> sin_th = th.sin();
        Eigen::Array<double, N_FREQ, 1> cos_th = th.cos();

        Eigen::Array<double, N_FREQ, 1> exp_z = exp_kz_cache_;

        double dx = (-A_.array()*cos_th*dir_x_.array()*exp_z).sum();
        double dy = (-A_.array()*cos_th*dir_y_.array()*exp_z).sum();
        double dz = (A_.array()*sin_th*exp_z).sum();

        loopOverPairs([&](size_t idx,double Bij,double cos_th2,double,double s,double kx_sum,double ky_sum,double,double factor){
            dx += factor*(-Bij)*cos_th2*kx_sum/(s>1e-12?s:1.0);
            dy += factor*(-Bij)*cos_th2*ky_sum/(s>1e-12?s:1.0);
        },x,y,t,z);

        return {dx,dy,dz};
    }

    Eigen::Vector3d evaluateVelocity(double x,double y,double t,double z) const {
        ensureExpKzCached(z);
        Eigen::Array<double, N_FREQ, 1> th = (kx_*x + ky_*y - omega_*t + phi_).array();
        Eigen::Array<double, N_FREQ, 1> sin_th = th.sin();
        Eigen::Array<double, N_FREQ, 1> cos_th = th.cos();
        Eigen::Array<double, N_FREQ, 1> fac = A_.array()*omega_.array()*exp_kz_cache_;

        double vx = (fac*sin_th*dir_x_.array()).sum() + stokes_drift_mean_xy_.x();
        double vy = (fac*sin_th*dir_y_.array()).sum() + stokes_drift_mean_xy_.y();
        double vz = (fac*cos_th).sum();

        loopOverPairs([&](size_t idx,double Bij,double,double sin_th2,double,double,double,double omega_sum,double factor){
            vx += factor*(-Bij)*omega_sum*sin_th2*kx_sum_flat_(idx)/(k_sum_flat_(idx)>1e-12?k_sum_flat_(idx):1.0);
            vy += factor*(-Bij)*omega_sum*sin_th2*ky_sum_flat_(idx)/(k_sum_flat_(idx)>1e-12?k_sum_flat_(idx):1.0);
            vz += factor*(-Bij)*omega_sum*sin_th2;
        },x,y,t,z);

        return {vx,vy,vz};
    }

    Eigen::Vector3d evaluateAcceleration(double x,double y,double t,double z) const {
        ensureExpKzCached(z);
        Eigen::Array<double, N_FREQ, 1> th = (kx_*x + ky_*y - omega_*t + phi_).array();
        Eigen::Array<double, N_FREQ, 1> sin_th = th.sin();
        Eigen::Array<double, N_FREQ, 1> cos_th = th.cos();
        Eigen::Array<double, N_FREQ, 1> fac = A_.array()*omega_.array().square()*exp_kz_cache_;

        double ax = (fac*cos_th*dir_x_.array()).sum();
        double ay = (fac*cos_th*dir_y_.array()).sum();
        double az = (-fac*sin_th).sum();

        loopOverPairs([&](size_t idx,double Bij,double cos_th2,double,double,double,double,double omega_sum,double factor){
            ax += factor*Bij*omega_sum*omega_sum*cos_th2*kx_sum_flat_(idx)/(k_sum_flat_(idx)>1e-12?k_sum_flat_(idx):1.0);
            ay += factor*Bij*omega_sum*omega_sum*cos_th2*ky_sum_flat_(idx)/(k_sum_flat_(idx)>1e-12?k_sum_flat_(idx):1.0);
            az += factor*Bij*omega_sum*omega_sum*cos_th2;
        },x,y,t,z);

        return {ax,ay,az};
    }

    void checkSteepness() const {
        double max_steepness=(A_.array()*k_.array()).maxCoeff();
        if(max_steepness>0.2)
            throw std::runtime_error("Wave steepness exceeds 0.2");
    }
};

#ifdef JONSWAP_TEST
void generateWaveJonswapCSV(const std::string& filename,
                            double Hs, double Tp, double mean_dir_deg,
                            double duration = 40.0, double dt = 0.005) {

    Jonswap3dStokesWaves<256> waveModel(
        Hs, Tp, mean_dir_deg, 0.02, 0.8, 2.0, 9.81, 10.0
    );

    std::ofstream file(filename);
    file << "time,disp_x,disp_y,disp_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z\n";

    const double x0 = 0.0, y0 = 0.0;
    for (double t = 0; t <= duration; t += dt) {
        auto state = waveModel.getLagrangianState(x0, y0, t);
        file << t << ","
             << state.displacement.x() << ","
             << state.displacement.y() << ","
             << state.displacement.z() << ","
             << state.velocity.x() << ","
             << state.velocity.y() << ","
             << state.velocity.z() << ","
             << state.acceleration.x() << ","
             << state.acceleration.y() << ","
             << state.acceleration.z() << "\n";
    }
}

void Jonswap_testWavePatterns() {
    generateWaveJonswapCSV("short_waves_stokes.csv", 0.5, 3.0, 30.0);
    generateWaveJonswapCSV("medium_waves_stokes.csv", 2.0, 7.0, 30.0);
    generateWaveJonswapCSV("long_waves_stokes.csv", 4.0, 12.0, 30.0);
}
#endif
