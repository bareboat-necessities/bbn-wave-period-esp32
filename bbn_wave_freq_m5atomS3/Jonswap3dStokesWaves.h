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
                         unsigned int seed = 239u)
        : Hs_(Hs), Tp_(Tp), mean_dir_rad_(mean_direction_deg * M_PI / 180.0),
          gamma_(gamma), g_(g), spreading_exponent_(spreading_exponent),
          seed_(seed), pairwise_size_((size_t)N_FREQ*(N_FREQ+1)/2),
          spectrum_(Hs, Tp, f_min, f_max, gamma, g),
          exp_kz_cached_z_(std::numeric_limits<double>::quiet_NaN())
    {
        frequencies_ = spectrum_.frequencies();
        S_ = spectrum_.spectrum();
        A_ = spectrum_.amplitudes();
        df_ = spectrum_.df();

        omega_ = 2.0 * M_PI * frequencies_;
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
    }

    // Get Lagrangian state at single point (x,y) and time t
    WaveState getLagrangianState(double x, double y, double t, double z=0.0) const
    {
        return evalLagrangianCached(
            x, y, t,
            // First-order lambda
            [&](Eigen::Vector3d &disp, Eigen::Vector3d &vel, Eigen::Vector3d &acc,
                int i, double sin_th, double cos_th)
            {
                disp.x() += -A_(i) * cos_th * dir_x_(i);
                disp.y() += -A_(i) * cos_th * dir_y_(i);
                disp.z() +=  A_(i) * sin_th;

                double fac = A_(i) * omega_(i);
                vel.x() += fac * sin_th * dir_x_(i);
                vel.y() += fac * sin_th * dir_y_(i);
                vel.z() += fac * cos_th;

                fac = A_(i) * omega_(i) * omega_(i);
                acc.x() += fac * cos_th * dir_x_(i);
                acc.y() += fac * cos_th * dir_y_(i);
                acc.z() += -fac * sin_th;
            },
            // Second-order lambda
            [&](Eigen::Vector3d &disp, Eigen::Vector3d &vel, Eigen::Vector3d &acc,
                size_t idx, double factor, double sin_th, double cos_th)
            {
                disp.z() += factor * Bij_flat_[idx] * cos_th;
                double ksum2 = k_sum_flat_[idx]*k_sum_flat_[idx];
                if(ksum2>1e-18){
                    double hx = kx_sum_flat_[idx]/k_sum_flat_[idx];
                    double hy = ky_sum_flat_[idx]/k_sum_flat_[idx];
                    disp.x() += factor * (-Bij_flat_[idx]) * cos_th * hx;
                    disp.y() += factor * (-Bij_flat_[idx]) * cos_th * hy;
                }

                double sum_omega = omega_sum_flat_[idx];
                double hx = (ksum2>1e-18)? kx_sum_flat_[idx]/k_sum_flat_[idx]:0.0;
                double hy = (ksum2>1e-18)? ky_sum_flat_[idx]/k_sum_flat_[idx]:0.0;
                vel.x() += factor * (-Bij_flat_[idx]) * sum_omega * sin_th * hx;
                vel.y() += factor * (-Bij_flat_[idx]) * sum_omega * sin_th * hy;
                vel.z() += factor * (-Bij_flat_[idx]) * sum_omega * sin_th;

                double sum_omega2 = sum_omega*sum_omega;
                acc.x() += factor * Bij_flat_[idx] * sum_omega2 * cos_th * hx;
                acc.y() += factor * Bij_flat_[idx] * sum_omega2 * cos_th * hy;
                acc.z() += factor * Bij_flat_[idx] * sum_omega2 * cos_th;
            }
        );
    }

private:
    // Member variables
    JonswapSpectrum<N_FREQ> spectrum_;
    double Hs_, Tp_, mean_dir_rad_, gamma_, g_, spreading_exponent_;
    unsigned int seed_;
    size_t pairwise_size_;

    Eigen::Matrix<double, N_FREQ, 1> frequencies_, S_, A_, df_;
    Eigen::Matrix<double, N_FREQ, 1> omega_, k_, phi_;
    Eigen::Matrix<double, N_FREQ, 1> dir_x_, dir_y_, kx_, ky_;
    Eigen::Matrix<double, N_FREQ, 1> stokes_drift_scalar_;
    Eigen::Vector2d stokes_drift_mean_xy_;

    std::vector<double> Bij_flat_, kx_sum_flat_, ky_sum_flat_, k_sum_flat_, omega_sum_flat_, phi_sum_flat_;
    mutable std::vector<double> exp_kz_cache_;
    mutable double exp_kz_cached_z_;

    // Trig caching per time
    struct TrigCache {
        Eigen::ArrayXd sin_first, cos_first;
        Eigen::ArrayXd sin_second, cos_second;
        double last_t = std::numeric_limits<double>::quiet_NaN();
    };
    mutable TrigCache trig_cache_;

    // --- Initialization helpers ---
    void initializeRandomPhases() {
        std::mt19937 gen(seed_);
        std::uniform_real_distribution<double> dist(0.0, 2.0*M_PI);
        for(int i=0;i<N_FREQ;++i) phi_(i) = dist(gen);
    }

    void initializeDirectionalSpread() {
        std::mt19937 gen(seed_+1);
        std::uniform_real_distribution<double> u_dist(0.0, 2.0*M_PI);
        std::uniform_real_distribution<double> y_dist(0.0, 1.0);
        for(int i=0;i<N_FREQ;++i){
            double theta=0.0;
            while(true){
                double candidate=u_dist(gen);
                double clamped=std::max(0.0,std::cos(candidate-mean_dir_rad_));
                double pdf_val=std::pow(clamped,spreading_exponent_);
                if(y_dist(gen)<=pdf_val){theta=candidate; break;}
            }
            dir_x_(i)=std::cos(theta);
            dir_y_(i)=std::sin(theta);
        }
    }

    void computeWaveDirectionComponents() {
        for(int i=0;i<N_FREQ;++i){
            kx_(i) = k_(i) * dir_x_(i);
            ky_(i) = k_(i) * dir_y_(i);
        }
    }

    void computePerComponentStokesDriftEstimate() {
        stokes_drift_scalar_.setZero();
        stokes_drift_mean_xy_.setZero();
        for(int i=0;i<N_FREQ;++i){
            double Usi=0.5*A_(i)*A_(i)*k_(i)*omega_(i);
            stokes_drift_scalar_(i)=Usi;
            stokes_drift_mean_xy_.x() += Usi*dir_x_(i);
            stokes_drift_mean_xy_.y() += Usi*dir_y_(i);
        }
    }

    void precomputePairwise() {
        pairwise_size_ = (size_t)N_FREQ*(N_FREQ+1)/2;
        Bij_flat_.assign(pairwise_size_,0.0);
        kx_sum_flat_.assign(pairwise_size_,0.0);
        ky_sum_flat_.assign(pairwise_size_,0.0);
        k_sum_flat_.assign(pairwise_size_,0.0);
        omega_sum_flat_.assign(pairwise_size_,0.0);
        phi_sum_flat_.assign(pairwise_size_,0.0);

        for(int i=0;i<N_FREQ;++i){
            for(int j=i;j<N_FREQ;++j){
                size_t idx = upper_index(i,j);
                double kx_sum = kx_(i)+kx_(j);
                double ky_sum = ky_(i)+ky_(j);
                double kdot = kx_(i)*kx_(j)+ky_(i)*ky_(j);
                double Bij = (kdot)/(2.0*g_)*(A_(i)*A_(j));
                double omega_sum = omega_(i)+omega_(j);
                double phi_sum = phi_(i)+phi_(j);
                double ksum = std::sqrt(kx_sum*kx_sum + ky_sum*ky_sum);

                Bij_flat_[idx]=Bij;
                kx_sum_flat_[idx]=kx_sum;
                ky_sum_flat_[idx]=ky_sum;
                k_sum_flat_[idx]=ksum;
                omega_sum_flat_[idx]=omega_sum;
                phi_sum_flat_[idx]=phi_sum;
            }
        }
        exp_kz_cache_.assign(pairwise_size_, std::numeric_limits<double>::quiet_NaN());
        exp_kz_cached_z_ = std::numeric_limits<double>::quiet_NaN();
    }

    void checkSteepness() const {
        double max_steepness = (A_.array()*k_.array()).maxCoeff();
        if(max_steepness>0.2) throw std::runtime_error("Wave steepness exceeds 0.2");
    }

    inline size_t upper_index(int i,int j) const {
        const size_t N = (size_t)N_FREQ;
        size_t ii = (size_t)i;
        return ii*N-(ii*(ii-1))/2 + (size_t)(j-i);
    }

    // --- Core cached evaluation ---
    template<typename FuncFirst, typename FuncSecond>
    WaveState evalLagrangianCached(double x, double y, double t,
                                   FuncFirst computeFirst,
                                   FuncSecond computeSecond) const
    {
        Eigen::Vector3d disp=Eigen::Vector3d::Zero();
        Eigen::Vector3d vel=Eigen::Vector3d::Zero();
        Eigen::Vector3d acc=Eigen::Vector3d::Zero();

        Eigen::ArrayXd theta0 = kx_.array()*x + ky_.array()*y + phi_.array();
        static std::vector<double> theta2_static;
        if(theta2_static.size() != pairwise_size_){
            theta2_static.resize(pairwise_size_);
            for(int i=0;i<N_FREQ;++i)
                for(int j=i;j<N_FREQ;++j)
                    theta2_static[upper_index(i,j)] = kx_sum_flat_[upper_index(i,j)]*x +
                                                     ky_sum_flat_[upper_index(i,j)]*y +
                                                     phi_sum_flat_[upper_index(i,j)];
        }

        if(trig_cache_.last_t != t){
            trig_cache_.sin_first = (theta0 - omega_.array()*t).sin();
            trig_cache_.cos_first = (theta0 - omega_.array()*t).cos();

            trig_cache_.sin_second.resize(pairwise_size_);
            trig_cache_.cos_second.resize(pairwise_size_);
            for(size_t idx=0; idx<pairwise_size_; ++idx){
                double th = theta2_static[idx] - omega_sum_flat_[idx]*t;
                trig_cache_.sin_second(idx) = std::sin(th);
                trig_cache_.cos_second(idx) = std::cos(th);
            }
            trig_cache_.last_t = t;
        }

        for(int i=0;i<N_FREQ;++i)
            computeFirst(disp, vel, acc, i, trig_cache_.sin_first(i), trig_cache_.cos_first(i));

        for(int i=0;i<N_FREQ;++i)
            for(int j=i;j<N_FREQ;++j){
                size_t idx=upper_index(i,j);
                double factor=(i==j)?1.0:2.0;
                computeSecond(disp, vel, acc, idx, factor,
                              trig_cache_.sin_second(idx), trig_cache_.cos_second(idx));
            }

        vel.x() += stokes_drift_mean_xy_.x();
        vel.y() += stokes_drift_mean_xy_.y();

        return {disp, vel, acc};
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
