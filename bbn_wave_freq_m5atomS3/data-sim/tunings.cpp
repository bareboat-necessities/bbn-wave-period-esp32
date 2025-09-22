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

struct Tuning {
    double tau;
    double sigma_a;
    double R_S;
};

// === Compute tuning directly from spectrum ===
template<int N>
static Tuning compute_from_spectrum(const Eigen::Matrix<double, N, 1>& freqs,
                                    const Eigen::Matrix<double, N, 1>& S,
                                    const Eigen::Matrix<double, N, 1>& df,
                                    double Tp)
{
    Tuning out{};
    Eigen::Array<double, N, 1> omega = 2.0 * M_PI * freqs.array();

    // displacement variance m0 = ∫ S(f) df
    double m0 = (S.cwiseProduct(df)).sum();

    // acceleration variance = ∫ ω⁴ S(f) df
    double var_acc = ((omega.pow(4) * S.array()) * df.array()).sum();

    out.tau     = 1.0 / (2.0 * M_PI / Tp);  // ≈ Tp / (2π)
    out.sigma_a = std::sqrt(var_acc);
    out.R_S     = 4.0 * m0;                 // variance of 4η (pseudo-measurement heuristic)

    return out;
}

int main() {
    const std::string out_csv = "wave_tunings.csv";
    std::ofstream file(out_csv);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << out_csv << " for writing\n";
        return 1;
    }

    file << "wave_index,wave_type,Hs,Tp,tau,sigma_a,R_S\n";

    for (size_t idx = 0; idx < waveParamsList.size(); ++idx) {
        const auto& wp = waveParamsList[idx];

        // --- JONSWAP ---
        auto dirDistJ = std::make_shared<Cosine2sRandomizedDistribution>(
            wp.direction * M_PI / 180.0, 10.0, 42u);
        Jonswap3dStokesWaves<128> jonswap(
            wp.height, wp.period, dirDistJ, 0.02, 0.8, 3.3, 9.81, 42u);

        auto tj = compute_from_spectrum<128>(
            jonswap.frequencies(), jonswap.spectrum(), jonswap.df(), wp.period);

        file << idx << ",JONSWAP,"
             << wp.height << "," << wp.period << ","
             << tj.tau << "," << tj.sigma_a << "," << tj.R_S << "\n";

        // --- PMStokes ---
        auto dirDistP = std::make_shared<Cosine2sRandomizedDistribution>(
            wp.direction * M_PI / 180.0, 10.0, 42u);
        PMStokesN3dWaves<128,3> pm(
            wp.height, wp.period, dirDistP, 0.02, 0.8, 9.81, 42u);

        auto tp = compute_from_spectrum<128>(
            pm.frequencies(), pm.spectrum(), pm.df(), wp.period);

        file << idx << ",PMSTOKES,"
             << wp.height << "," << wp.period << ","
             << tp.tau << "," << tp.sigma_a << "," << tp.R_S << "\n";
    }

    file.close();
    std::cout << "Wrote " << out_csv << "\n";
    return 0;
}
