#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
#include <vector>

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
    double tau;       // from spectrum
    double sigma_a;   // from spectrum
    double R_S;       // from spectrum
    double Hs_spec;
    double m0, m2, m4;
    double Tm01, Tm02, Tm10;
};

struct TuningHeur {
    double tau;       // heuristic
    double sigma_a;   // heuristic
    double R_S;       // heuristic
    double m0_from_Hs;
};

// === From-spectrum tuning (your moment-based method) ===
template<int N>
static TuningSpec compute_from_spectrum(const Eigen::Matrix<double, N, 1>& freqs,
                                        const Eigen::Matrix<double, N, 1>& S,
                                        const Eigen::Matrix<double, N, 1>& df,
                                        double Tp)
{
    TuningSpec out{};
    Eigen::Array<double, N, 1> omega = 2.0 * M_PI * freqs.array();

    // spectral moments
    double m0    = (S.cwiseProduct(df)).sum();
    double m1    = ((omega * S.array()) * df.array()).sum();
    double m2    = ((omega.square() * S.array()) * df.array()).sum();
    double m4    = ((omega.pow(4)   * S.array()) * df.array()).sum();
    double mneg1 = (((1.0 / omega)  * S.array()) * df.array()).sum();

    // Tuning from spectrum
    const double omega_p = 2.0 * M_PI / Tp;
    out.tau     = 1.0 / omega_p;   // same as Tp/(2π)
    out.sigma_a = std::sqrt(m4);   // Var[a] = ∫ ω^4 S df
    out.R_S     = 4.0 * m0;        // your heuristic scale on m0

    out.Hs_spec = 4.0 * std::sqrt(m0);
    out.m0      = m0;
    out.m2      = m2;
    out.m4      = m4;

    // Mean period estimates (optional diagnostics)
    out.Tm01 = (m1 > 0.0) ? (m0 / m1) : NAN;
    out.Tm02 = (m2 > 0.0) ? std::sqrt(m0 / m2) : NAN;
    out.Tm10 = (m0 > 0.0) ? (mneg1 / m0) : NAN;

    return out;
}

// === Heuristic tuning from (Hs, Tp) only (no spectrum integration) ===
// Uses narrowband equivalence: Var[a] ≈ ωp^4 m0 with m0 = Hs^2/16, τ = 1/ωp, R_S = c_RS m0.
static TuningHeur compute_heuristic_from_HsTp(double Hs, double Tp, double c_RS = 4.0)
{
    TuningHeur h{};
    const double m0 = (Hs*Hs)/16.0;
    const double omega_p = 2.0 * M_PI / Tp;

    h.tau      = 1.0 / omega_p;              // = Tp/(2π)
    h.sigma_a  = (omega_p*omega_p) * std::sqrt(m0); // = sqrt(ωp^4 m0)
    h.R_S      = c_RS * m0;                  // same scale used in spectrum method by default
    h.m0_from_Hs = m0;
    return h;
}

int main() {
    const std::string out_csv = "wave_tunings.csv";
    std::ofstream file(out_csv);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << out_csv << " for writing\n";
        return 1;
    }

    file << "wave_index,wave_type,Hs_input,Tp,"
         // from spectrum
         << "tau_spec,sigma_a_spec,R_S_spec,Hs_spec,"
         << "m0,m2,m4,Tm01,Tm02,Tm10,"
         // heuristic
         << "tau_heur,sigma_a_heur,R_S_heur,"
         // quick comparison ratios (heur/spec)
         << "tau_ratio, sigma_a_ratio, R_S_ratio\n";

    for (size_t idx = 0; idx < waveParamsList.size(); ++idx) {
        const auto& wp = waveParamsList[idx];

        // --- JONSWAP ---
        {
            auto dirDistJ = std::make_shared<Cosine2sRandomizedDistribution>(
                wp.direction * M_PI / 180.0, 10.0, 42u);
            Jonswap3dStokesWaves<128> jonswap(
                wp.height, wp.period, dirDistJ, 0.02, 0.8, 3.3, 9.81, 42u);

            const auto tj   = compute_from_spectrum<128>(
                jonswap.frequencies(), jonswap.spectrum(), jonswap.df(), wp.period);
            const auto th   = compute_heuristic_from_HsTp(wp.height, wp.period);

            const double tau_ratio    = (tj.tau    != 0.0) ? (th.tau    / tj.tau)    : NAN;
            const double sigma_ratio  = (tj.sigma_a!= 0.0) ? (th.sigma_a/ tj.sigma_a): NAN;
            const double RS_ratio     = (tj.R_S    != 0.0) ? (th.R_S    / tj.R_S)    : NAN;

            file << idx << ",JONSWAP,"
                 << wp.height << "," << wp.period << ","
                 << tj.tau << "," << tj.sigma_a << "," << tj.R_S << "," << tj.Hs_spec << ","
                 << tj.m0 << "," << tj.m2 << "," << tj.m4 << "," << tj.Tm01 << "," << tj.Tm02 << "," << tj.Tm10 << ","
                 << th.tau << "," << th.sigma_a << "," << th.R_S << ","
                 << tau_ratio << "," << sigma_ratio << "," << RS_ratio << "\n";
        }

        // --- PMStokes ---
        {
            auto dirDistP = std::make_shared<Cosine2sRandomizedDistribution>(
                wp.direction * M_PI / 180.0, 10.0, 42u);
            PMStokesN3dWaves<128,3> pm(
                wp.height, wp.period, dirDistP, 0.02, 0.8, 9.81, 42u);

            const auto tp   = compute_from_spectrum<128>(
                pm.frequencies(), pm.spectrum(), pm.df(), wp.period);
            const auto th   = compute_heuristic_from_HsTp(wp.height, wp.period);

            const double tau_ratio    = (tp.tau    != 0.0) ? (th.tau    / tp.tau)    : NAN;
            const double sigma_ratio  = (tp.sigma_a!= 0.0) ? (th.sigma_a/ tp.sigma_a): NAN;
            const double RS_ratio     = (tp.R_S    != 0.0) ? (th.R_S    / tp.R_S)    : NAN;

            file << idx << ",PMSTOKES,"
                 << wp.height << "," << wp.period << ","
                 << tp.tau << "," << tp.sigma_a << "," << tp.R_S << "," << tp.Hs_spec << ","
                 << tp.m0 << "," << tp.m2 << "," << tp.m4 << "," << tp.Tm01 << "," << tp.Tm02 << "," << tp.Tm10 << ","
                 << th.tau << "," << th.sigma_a << "," << th.R_S << ","
                 << tau_ratio << "," << sigma_ratio << "," << RS_ratio << "\n";
        }
    }

    file.close();
    std::cout << "Wrote " << out_csv << "\n";
    return 0;
}
