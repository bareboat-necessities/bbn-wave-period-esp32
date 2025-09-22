#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <vector>

#define EIGEN_NON_ARDUINO
#include <Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float g_std = 9.80665f; // standard gravity acceleration m/s²

#include "Jonswap3dStokesWaves.h"
#include "PiersonMoskowitzStokes3D_Waves.h"
#include "DirectionalSpread.h"
#include "WaveFilesSupport.h"   // for WaveParameters

// ==================== Input test waves ====================
const std::vector<WaveParameters> waveParamsList = {
    {3.0f,   0.27f, static_cast<float>(M_PI/3.0), 30.0f},
    {5.7f,   1.5f,  static_cast<float>(M_PI/1.5), 30.0f},
    {8.5f,   4.0f,  static_cast<float>(M_PI/6.0), 30.0f},
    {11.4f,  8.5f,  static_cast<float>(M_PI/2.5), 30.0f},
    {14.3f,  14.8f, static_cast<float>(M_PI/5.0), 30.0f}
};

// ==================== Data structures ====================
struct TuningSpec {
    double tau, sigma_a, R_S;
    double Hs_spec;
    double m0, m1, m2, m4, mneg1;
    double Tm01, Tm02, Tm10;
    double f_disp_mean;   // (m1/m0)/(2π)  [Hz]
    double f_acc_mean;    // (m5/m4)/(2π)  [Hz]
    double f_peak_spec;   // argmax S(f)   [Hz]
    double f_peak_input;  // 1/Tp_input    [Hz]
};

struct TuningHeur {
    double tau, sigma_a, R_S;
    double m0_from_Hs;
};

struct TuningIMU {
    double tau_eff, sigma_a_eff, R_S_eff;
};

// ==================== Spectrum-based tuning ====================
//
// Spectral moments of displacement spectrum Sη(f):
//   m₀  = ∫ S(f) df
//   m₁  = ∫ ω S(f) df
//   m₂  = ∫ ω² S(f) df
//   m₄  = ∫ ω⁴ S(f) df
//   m₋₁ = ∫ (1/ω) S(f) df
//
// Acceleration-mean requires m₅ = ∫ ω⁵ S(f) df
//
// Significant wave height: Hs = 4√m₀
// τ (time constant): τ = 1/ωₚ with ωₚ = 2π f_peak (from spectrum)
// σₐ = √m₄
// R_S ≈ 4 m₀  (pseudo-measurement variance)
//
template<int N>
static TuningSpec compute_from_spectrum(const Eigen::Matrix<double, N, 1>& freqs,
                                        const Eigen::Matrix<double, N, 1>& S,
                                        const Eigen::Matrix<double, N, 1>& df,
                                        double Tp_input)
{
    TuningSpec out{};
    Eigen::Array<double, N, 1> omega = 2.0 * M_PI * freqs.array();

    // Raw spectral moments
    out.m0    = (S.cwiseProduct(df)).sum();
    out.m1    = ((omega * S.array()) * df.array()).sum();
    out.m2    = ((omega.square() * S.array()) * df.array()).sum();
    out.m4    = ((omega.pow(4)   * S.array()) * df.array()).sum();
    out.mneg1 = (((1.0 / omega)  * S.array()) * df.array()).sum();

    // m5 for acceleration mean frequency
    double m5 = ((omega.pow(5) * S.array()) * df.array()).sum();

    // === Spectral peak frequency (displacement spectrum) ===
    int idx_max;
    S.maxCoeff(&idx_max);
    out.f_peak_spec = freqs(idx_max);          // Hz
    double omega_p_spec = 2.0 * M_PI * out.f_peak_spec;

    // === Input nominal peak ===
    out.f_peak_input = 1.0 / Tp_input;         // Hz

    // τ, σₐ, R_S
    out.tau     = (omega_p_spec > 0.0) ? (1.0 / omega_p_spec) : NAN;
    out.sigma_a = std::sqrt(out.m4);
    out.R_S     = 4.0 * out.m0;

    // Heights & mean periods
    out.Hs_spec = 4.0 * std::sqrt(out.m0);
    out.Tm01    = (out.m1 > 0.0) ? (out.m0 / out.m1) : NAN;
    out.Tm02    = (out.m2 > 0.0) ? std::sqrt(out.m0 / out.m2) : NAN;
    out.Tm10    = (out.m0 > 0.0) ? (out.mneg1 / out.m0) : NAN;

    // Mean frequencies
    out.f_disp_mean = (out.m0 > 0.0) ? (out.m1 / out.m0) / (2.0 * M_PI) : NAN;
    out.f_acc_mean  = (out.m4 > 0.0) ? (m5 / out.m4) / (2.0 * M_PI) : NAN;

    return out;
}

// ==================== Heuristic tuning from Hs,Tp ====================
//
// Classic oceanographic formulas:
//   m₀ = Hs² / 16
//   τ  = Tₚ / (2π)
//   σₐ ≈ ωₚ² √m₀
//   R_S ≈ 4 m₀
//
static TuningHeur compute_heuristic_from_HsTp(double Hs, double Tp, double c_RS = 4.0)
{
    TuningHeur h{};
    const double m0 = (Hs*Hs)/16.0;
    const double omega_p = 2.0 * M_PI / Tp;
    h.tau      = 1.0 / omega_p;
    h.sigma_a  = (omega_p*omega_p) * std::sqrt(m0);
    h.R_S      = c_RS * m0;
    h.m0_from_Hs = m0;
    return h;
}

// ==================== Heuristic tuning from spectral peak (heur2) ====================
//
// Using peak frequency f_peak_spec from displacement spectrum:
//   m₀_from_Hs = Hs² / 16
//   ωₚ = 2π f_peak_spec
//   τ     = 1 / ωₚ
//   σₐ    = ωₚ² √m₀_from_Hs
//   R_S   = c_RS m₀_from_Hs
//
static TuningHeur compute_heuristic_from_spectral_peak(double Hs,
                                                       double f_peak_spec,
                                                       double c_RS = 4.0)
{
    TuningHeur h{};
    const double m0_from_Hs = (Hs * Hs) / 16.0;
    h.m0_from_Hs = m0_from_Hs;

    if (f_peak_spec > 0.0) {
        const double omega_p = 2.0 * M_PI * f_peak_spec;
        h.tau     = 1.0 / omega_p;
        h.sigma_a = (omega_p * omega_p) * std::sqrt(m0_from_Hs);
    } else {
        h.tau = std::numeric_limits<double>::quiet_NaN();
        h.sigma_a = std::numeric_limits<double>::quiet_NaN();
    }

    h.R_S = c_RS * m0_from_Hs;
    return h;
}

// ==================== IMU-adjusted tuning ====================
//
// Adds noise and bias effects from IMU model.
// See your earlier sim constants.
//
static TuningIMU compute_with_imu_same_as_sim(const TuningSpec& base)
{
    // Filter noise levels
    const Eigen::Vector3d sigma_acc_filter(0.04, 0.04, 0.04);          // m/s²
    const Eigen::Vector3d sigma_gyro_filter(0.00134, 0.00134, 0.00134);// rad/s

    // True injected white noise
    const double sigma_acc_true = 0.03;   // m/s²
    const double sigma_gyro_true = 0.001; // rad/s

    // Static biases
    const double bacc_mag = 0.02;   // m/s²
    const double bgyr_mag = 0.0004; // rad/s

    const double sigma_acc_filter_rms = sigma_acc_filter.norm() / std::sqrt(3.0);
    const double sigma_gyro_filter_rms = sigma_gyro_filter.norm() / std::sqrt(3.0);

    TuningIMU t{};

    // Effective accel std
    const double sigma_a_eff = std::sqrt(
        base.sigma_a * base.sigma_a +
        sigma_acc_filter_rms * sigma_acc_filter_rms +
        sigma_acc_true * sigma_acc_true +
        bacc_mag * bacc_mag
    );
    t.sigma_a_eff = sigma_a_eff;

    // Effective tau (tilt diffusion shortens τ)
    const double tilt_var = (sigma_gyro_filter_rms * sigma_gyro_filter_rms) +
                            (sigma_gyro_true * sigma_gyro_true) +
                            (bgyr_mag * bgyr_mag);
    t.tau_eff = base.tau / (1.0 + tilt_var * base.tau);

    // R_S inflated by noise/bias
    t.R_S_eff = base.R_S + 3.0 * (
        sigma_acc_filter_rms * sigma_acc_filter_rms +
        sigma_acc_true * sigma_acc_true +
        bacc_mag * bacc_mag
    );

    return t;
}

// ==================== Runner ====================
template<typename WaveModel>
void process_wave(const WaveParameters& wp,
                  size_t wave_index,
                  const std::string& type,
                  std::ofstream& file)
{
    auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
        wp.direction * M_PI / 180.0, 10.0, 42u);

    WaveModel model(wp.height, wp.period, dirDist, 0.02, 0.8, 9.81, 42u);

    const auto t_spec  = compute_from_spectrum<128>(
        model.frequencies(), model.spectrum(), model.df(), wp.period);
    const auto t_heur  = compute_heuristic_from_HsTp(wp.height, wp.period);
    const auto t_heur2 = compute_heuristic_from_spectral_peak(wp.height, t_spec.f_peak_spec);
    const auto t_imu   = compute_with_imu_same_as_sim(t_spec);

    // === CSV output ===
    file << wave_index << "," << type << ","
         << wp.height << "," << wp.period << ","
         << t_spec.tau << "," << t_spec.sigma_a << "," << t_spec.R_S << "," << t_spec.Hs_spec << ","
         << t_spec.m0 << "," << t_spec.m2 << "," << t_spec.m4
         << "," << t_spec.Tm01 << "," << t_spec.Tm02 << "," << t_spec.Tm10 << ","
         << t_spec.f_disp_mean << "," << t_spec.f_acc_mean
         << "," << t_spec.f_peak_spec << "," << t_spec.f_peak_input << ","
         << t_heur.tau << "," << t_heur.sigma_a << "," << t_heur.R_S << ","
         << t_heur2.tau << "," << t_heur2.sigma_a << "," << t_heur2.R_S << ","
         << t_imu.tau_eff << "," << t_imu.sigma_a_eff << "," << t_imu.R_S_eff
         << "\n";

    // === Human-readable report ===
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Wave " << wave_index << " (" << type << ")\n"
              << "  Input: Hs=" << wp.height << " m, Tp=" << wp.period << " s\n"
              << "  Spectrum-derived:\n"
              << "    tau=" << t_spec.tau << " s (1/ωp, exact from spectrum)\n"
              << "    sigma_a=" << t_spec.sigma_a << " m/s^2, R_S=" << t_spec.R_S << " m^2\n"
              << "    Hs_spec=" << t_spec.Hs_spec << " m\n"
              << "    m0=" << t_spec.m0 << ", m2=" << t_spec.m2 << ", m4=" << t_spec.m4 << "\n"
              << "    Tm01=" << t_spec.Tm01 << " s, Tm02=" << t_spec.Tm02
              << " s, Tm10=" << t_spec.Tm10 << " s\n"
              << "    f_disp_mean=" << t_spec.f_disp_mean << " Hz (displacement mean)\n"
              << "    f_acc_mean =" << t_spec.f_acc_mean  << " Hz (acceleration mean)\n"
              << "    f_peak_spec=" << t_spec.f_peak_spec << " Hz (spectrum peak)\n"
              << "    f_peak_in  =" << t_spec.f_peak_input << " Hz (input nominal)\n"
              << "  Heuristic (Hs,Tp classic): tau=" << t_heur.tau
              << " s, sigma_a=" << t_heur.sigma_a << " m/s^2, R_S=" << t_heur.R_S << " m^2\n"
              << "  Heuristic (Spectral peak, heur2): tau=" << t_heur2.tau
              << " s, sigma_a=" << t_heur2.sigma_a << " m/s^2, R_S=" << t_heur2.R_S << " m^2\n"
              << "  IMU-adjusted (same as sim constants): tau_eff=" << t_imu.tau_eff
              << " s, sigma_a_eff=" << t_imu.sigma_a_eff << " m/s^2, R_S_eff=" << t_imu.R_S_eff << " m^2\n\n";
}

// ==================== Main ====================
int main() {
    const std::string out_csv = "wave_tunings.csv";
    std::ofstream file(out_csv);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << out_csv << " for writing\n";
        return 1;
    }

    file << "wave_index,wave_type,Hs_input,Tp,"
         << "tau_spec,sigma_a_spec,R_S_spec,Hs_spec,"
         << "m0,m2,m4,Tm01,Tm02,Tm10,"
         << "f_disp_mean,f_acc_mean,f_peak_spec,f_peak_input,"
         << "tau_heur,sigma_a_heur,R_S_heur,"
         << "tau_heur2,sigma_a_heur2,R_S_heur2,"
         << "tau_imu,sigma_a_imu,R_S_imu\n";

    for (size_t idx = 0; idx < waveParamsList.size(); ++idx) {
        process_wave<Jonswap3dStokesWaves<128>>(waveParamsList[idx], idx, "JONSWAP", file);
        process_wave<PMStokesN3dWaves<128,3>>(waveParamsList[idx], idx, "PMSTOKES", file);
    }

    file.close();
    std::cout << "Wrote " << out_csv << "\n";
    return 0;
}

