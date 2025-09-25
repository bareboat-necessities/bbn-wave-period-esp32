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
    {3.0f,   0.27f, static_cast<float>(M_PI/3.0), 25.0f},
    {5.7f,   1.5f,  static_cast<float>(M_PI/1.5), 25.0f},
    {8.5f,   4.0f,  static_cast<float>(M_PI/6.0), 25.0f},
    {11.4f,  8.5f,  static_cast<float>(M_PI/2.5), 25.0f}
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

// ==================== Spectrum-based tuning (band-limited) ====================
//
// We compute moments of the *displacement* spectrum S_eta(f) after applying a
// gentle 2nd-order low-pass to avoid overweighting the ω^4 tail (problematic on PM).
//
// Filter:
//   H(f) = 1 / (1 + (f/f_c)^2)^2,   with  f_c = 3 * f_peak_spec
//
// Band-limited moments (using S_bl = S_eta * H):
//   m0    = ∫ S_bl df
//   m1    = ∫ ω S_bl df
//   m2    = ∫ ω^2 S_bl df
//   m4    = ∫ ω^4 S_bl df
//   m-1   = ∫ (1/ω) S_bl df
//   m5    = ∫ ω^5 S_bl df   (for acceleration-mean frequency)
//
// Tunings derived from band-limited moments:
//   sigma_a = sqrt(m4)                                      // process std for a_w
//   f_disp_mean = (m1/m0)/(2π)
//   f_acc_mean  = (m5/m4)/(2π)
//   tau = 0.8 / (2π * min(f_acc_mean, 2 * f_peak_spec))     // shorter than 1/ω_p on broad spectra
//   R_S = 4 * m0                                            //  pseudo-measurement Kalman R
//
// Notes:
// - On JONSWAP (narrow), the LP has little effect; τ and R_S change slightly.
// - On PM (broad), this prevents σ_a inflation from ω^4 and shortens τ appropriately.
//
template<int N>
static TuningSpec compute_from_spectrum(const Eigen::Matrix<double, N, 1>& freqs,
                                        const Eigen::Matrix<double, N, 1>& S,
                                        const Eigen::Matrix<double, N, 1>& df,
                                        double Tp_input)
{
    TuningSpec out{};
    const double epsf = 1e-6;

    // Angular frequency
    Eigen::Array<double, N, 1> omega = 2.0 * M_PI * freqs.array();

    // Displacement spectral peak (for cutoff and reporting)
    int idx_max;
    S.maxCoeff(&idx_max);
    out.f_peak_spec = freqs(idx_max);                 // Hz
    const double fp = std::max(out.f_peak_spec, epsf);

    // Gentle LP to tame ω^4 tail (helps PM, harmless for JONSWAP)
    const double fc = 3.0 * fp;
    Eigen::Array<double, N, 1> H = 1.0 / (1.0 + (freqs.array() / fc).square()).square();
    Eigen::Array<double, N, 1> Sbl = S.array() * H;

    // Band-limited moments
    out.m0    = (Sbl.matrix().cwiseProduct(df)).sum();
    out.m1    = ((omega * Sbl) * df.array()).sum();
    out.m2    = ((omega.square() * Sbl) * df.array()).sum();
    out.m4    = ((omega.pow(4)   * Sbl) * df.array()).sum();
    out.mneg1 = (((1.0 / omega)  * Sbl) * df.array()).sum();
    const double m5 = ((omega.pow(5) * Sbl) * df.array()).sum();

    // Input nominal peak (from provided Tp)
    out.f_peak_input = (Tp_input > 0.0) ? (1.0 / Tp_input) : std::numeric_limits<double>::quiet_NaN();

    // Derived means and tunings
    out.f_disp_mean = (out.m0 > 0.0) ? (out.m1 / out.m0) / (2.0 * M_PI) : std::numeric_limits<double>::quiet_NaN();
    out.f_acc_mean  = (out.m4 > 0.0) ? (m5 / out.m4)     / (2.0 * M_PI) : std::numeric_limits<double>::quiet_NaN();

    // sigma_a from band-limited m4
    out.sigma_a = (out.m4 > 0.0) ? std::sqrt(out.m4) : 0.0;

    // tau shortened toward acceleration content; clamp with 2*fp
    const double f_tau = std::min(std::max(out.f_acc_mean, epsf), 2.0 * fp);
    out.tau = 0.8 / (2.0 * M_PI * f_tau);

    // Pseudo-measurement variance relaxed for energetic/broad seas
    out.R_S = 4.0 * out.m0;

    // Heights & mean periods (band-limited, consistent with above)
    out.Hs_spec = (out.m0 > 0.0) ? 4.0 * std::sqrt(out.m0) : 0.0;
    out.Tm01    = (out.m1 > 0.0) ? (out.m0 / out.m1) : std::numeric_limits<double>::quiet_NaN();
    out.Tm02    = (out.m2 > 0.0) ? std::sqrt(out.m0 / out.m2) : std::numeric_limits<double>::quiet_NaN();
    out.Tm10    = (out.m0 > 0.0) ? (out.mneg1 / out.m0) : std::numeric_limits<double>::quiet_NaN();

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
static TuningHeur compute_heuristic_from_HsTp(double Hs, double Tp, double c_RS = 2.0)
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
// Intent:
// - Keep σ_a representing *stochastic* acceleration (white + prefilter), not constant bias.
// - Soften τ only when gyro noise approaches sea dynamics.
// - Absorb accelerometer bias via extra slack in R_S (drift pseudo-measurement), not in σ_a.
//
// Effective terms:
//   sigma_a_eff = hypot(sigma_a_bandlimited, sigma_acc_white, sigma_acc_prefilter)
//   tau_eff = tau / sqrt(1 + (sigma_gyro_eff / (2π f_tau))^2) ,  f_tau ≈ max(f_acc_mean, f_peak_spec)
//   R_S_eff = R_S + k_bias * bacc_mag^2   (k_bias ≈ 3)
//
// Notes:
// - Uses the same “sim constants” you had; adjust to your measured sensor if needed.
//
static TuningIMU compute_with_imu_same_as_sim(const TuningSpec& base)
{
    // Sim / filter noise levels (RMS per axis where applicable)
    const Eigen::Vector3d sigma_acc_filter(0.04, 0.04, 0.04);          // m/s^2
    const Eigen::Vector3d sigma_gyro_filter(0.00134, 0.00134, 0.00134);// rad/s

    const double sigma_acc_true = 0.03;   // m/s^2 (white)
    const double sigma_gyro_true = 0.001; // rad/s (white)

    // Bias magnitudes (do NOT fold into σ_a)
    const double bacc_mag = 0.02;   // m/s^2
    const double bgyr_mag = 0.0004; // rad/s (kept for reference; not used directly here)

    const double sigma_acc_filter_rms = sigma_acc_filter.norm() / std::sqrt(3.0);
    const double sigma_gyro_filter_rms = sigma_gyro_filter.norm() / std::sqrt(3.0);

    TuningIMU t{};

    // σ_a: stochastic only (band-limited base + white + prefilter), exclude bias
    t.sigma_a_eff = std::hypot(base.sigma_a, std::hypot(sigma_acc_true, sigma_acc_filter_rms));

    // τ: gentle trim if gyro noise competes with sea dynamics at f_tau
    const double f_tau = std::max((std::isfinite(base.f_acc_mean) ? base.f_acc_mean : base.f_peak_spec), 1e-6);
    const double sigma_gyro_eff = std::hypot(sigma_gyro_true, sigma_gyro_filter_rms);
    const double factor = std::sqrt(1.0 + std::pow(sigma_gyro_eff / (2.0 * M_PI * f_tau), 2.0));
    t.tau_eff = base.tau / factor;

    // R_S: absorb accel bias as slow drift via inflation
    const double k_bias = 3.0;
    t.R_S_eff = base.R_S + k_bias * (bacc_mag * bacc_mag);

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
              << "    tau=" << t_spec.tau << " s (from spectrum)\n"
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

