#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <vector>

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "Jonswap3dStokesWaves.h"
#include "PiersonMoskowitzStokes3D_Waves.h"
#include "DirectionalSpread.h"
#include "WaveFilesSupport.h"   // for WaveParameters, waveParamsList

// Example test cases
const std::vector<WaveParameters> waveParamsList = {
    {3.0f,   0.27f, static_cast<float>(M_PI/3.0), 30.0f},
    {5.7f,   1.5f,  static_cast<float>(M_PI/1.5), 30.0f},
    {8.5f,   4.0f,  static_cast<float>(M_PI/6.0), 30.0f},
    {11.4f,  8.5f,  static_cast<float>(M_PI/2.5), 30.0f},
    {14.3f,  14.8f, static_cast<float>(M_PI/5.0), 30.0f}
};

struct TuningSpec {
    double tau, sigma_a, R_S;
    double Hs_spec;
    double m0, m1, m2, m4, mneg1;
    double Tm01, Tm02, Tm10;
    double f_mean;
};

struct TuningHeur {
    double tau, sigma_a, R_S;
    double m0_from_Hs;
};

// === From-spectrum tuning ===
template<int N>
static TuningSpec compute_from_spectrum(const Eigen::Matrix<double, N, 1>& freqs,
                                        const Eigen::Matrix<double, N, 1>& S,
                                        const Eigen::Matrix<double, N, 1>& df,
                                        double Tp)
{
    TuningSpec out{};
    Eigen::Array<double, N, 1> omega = 2.0 * M_PI * freqs.array();

    // Spectral moments:
    // m₀ = ∫ S(f) df               (displacement variance)
    // m₁ = ∫ ω S(f) df
    // m₂ = ∫ ω² S(f) df
    // m₄ = ∫ ω⁴ S(f) df            (acceleration variance)
    // m₋₁ = ∫ (1/ω) S(f) df
    out.m0    = (S.cwiseProduct(df)).sum();
    out.m1    = ((omega * S.array()) * df.array()).sum();
    out.m2    = ((omega.square() * S.array()) * df.array()).sum();
    out.m4    = ((omega.pow(4)   * S.array()) * df.array()).sum();
    out.mneg1 = (((1.0 / omega)  * S.array()) * df.array()).sum();

    // Tuning parameters:
    // τ = 1/ωₚ ≈ Tₚ / (2π)
    // σₐ = √m₄
    // R_S ≈ 4 m₀  (pseudo-measurement variance)
    const double omega_p = 2.0 * M_PI / Tp;
    out.tau     = 1.0 / omega_p;
    out.sigma_a = std::sqrt(out.m4);
    out.R_S     = 4.0 * out.m0;

    // Significant wave height from spectrum: Hs_spec = 4√m₀
    out.Hs_spec = 4.0 * std::sqrt(out.m0);

    // Spectral mean periods
    // Tm01 = m₀ / m₁
    // Tm02 = √(m₀ / m₂)
    // Tm10 = m₋₁ / m₀
    out.Tm01    = (out.m1 > 0.0) ? (out.m0 / out.m1) : NAN;
    out.Tm02    = (out.m2 > 0.0) ? std::sqrt(out.m0 / out.m2) : NAN;
    out.Tm10    = (out.m0 > 0.0) ? (out.mneg1 / out.m0) : NAN;

    // Mean spectral frequency: f_mean = (m₁/m₀) / (2π)
    out.f_mean  = (out.m0 > 0.0) ? (out.m1 / out.m0) / (2.0 * M_PI) : NAN;

    return out;
}

// === Heuristic from Hs,Tp ===
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

// === Generic runner for both spectra ===
template<typename WaveModel>
void process_wave(const WaveParameters& wp,
                  size_t wave_index,
                  const std::string& type,
                  std::ofstream& file)
{
    auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
        wp.direction * M_PI / 180.0, 10.0, 42u);

    WaveModel model(wp.height, wp.period, dirDist, 0.02, 0.8, 9.81, 42u);

    const auto t_spec = compute_from_spectrum<128>(
        model.frequencies(), model.spectrum(), model.df(), wp.period);
    const auto t_heur = compute_heuristic_from_HsTp(wp.height, wp.period);

    const double tau_ratio   = (t_spec.tau     != 0.0) ? (t_heur.tau     / t_spec.tau)    : NAN;
    const double sigma_ratio = (t_spec.sigma_a != 0.0) ? (t_heur.sigma_a / t_spec.sigma_a): NAN;
    const double RS_ratio    = (t_spec.R_S     != 0.0) ? (t_heur.R_S     / t_spec.R_S)    : NAN;

    // Write to CSV
    file << wave_index << "," << type << ","
         << wp.height << "," << wp.period << ","
         << t_spec.tau << "," << t_spec.sigma_a << "," << t_spec.R_S << "," << t_spec.Hs_spec << ","
         << t_spec.m0 << "," << t_spec.m2 << "," << t_spec.m4 << "," << t_spec.Tm01 << "," << t_spec.Tm02 << "," << t_spec.Tm10 << "," << t_spec.f_mean << ","
         << t_heur.tau << "," << t_heur.sigma_a << "," << t_heur.R_S << ","
         << tau_ratio << "," << sigma_ratio << "," << RS_ratio << "\n";

    // Human-readable report
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Wave " << wave_index << " (" << type << ")\n"
              << "  Input: Hs=" << wp.height << " m, Tp=" << wp.period << " s\n"
              << "  Spectrum-derived: tau=" << t_spec.tau
              << " s, sigma_a=" << t_spec.sigma_a
              << " m/s², R_S=" << t_spec.R_S
              << " m², Hs_spec=" << t_spec.Hs_spec
              << " m, f_mean=" << t_spec.f_mean << " Hz\n"
              << "  Heuristic: tau=" << t_heur.tau
              << " s, sigma_a=" << t_heur.sigma_a
              << " m/s², R_S=" << t_heur.R_S
              << " m²\n"
              << "  Ratios (heur/spec): tau=" << tau_ratio
              << ", sigma_a=" << sigma_ratio
              << ", R_S=" << RS_ratio << "\n\n";
}

int main() {
    const std::string out_csv = "wave_tunings.csv";
    std::ofstream file(out_csv);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << out_csv << " for writing\n";
        return 1;
    }

    file << "wave_index,wave_type,Hs_input,Tp,"
         << "tau_spec,sigma_a_spec,R_S_spec,Hs_spec,"
         << "m0,m2,m4,Tm01,Tm02,Tm10,f_mean,"
         << "tau_heur,sigma_a_heur,R_S_heur,"
         << "tau_ratio,sigma_a_ratio,R_S_ratio\n";

    for (size_t idx = 0; idx < waveParamsList.size(); ++idx) {
        process_wave<Jonswap3dStokesWaves<128>>(waveParamsList[idx], idx, "JONSWAP", file);
        process_wave<PMStokesN3dWaves<128,3>>(waveParamsList[idx], idx, "PMSTOKES", file);
    }

    file.close();
    std::cout << "Wrote " << out_csv << "\n";
    return 0;
}
