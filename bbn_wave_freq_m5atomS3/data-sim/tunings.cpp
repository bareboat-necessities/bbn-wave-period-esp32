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

struct TuningIMU {
    double tau_eff, sigma_a_eff, R_S_eff;
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

    // Spectral moments
    out.m0    = (S.cwiseProduct(df)).sum();
    out.m1    = ((omega * S.array()) * df.array()).sum();
    out.m2    = ((omega.square() * S.array()) * df.array()).sum();
    out.m4    = ((omega.pow(4)   * S.array()) * df.array()).sum();
    out.mneg1 = (((1.0 / omega)  * S.array()) * df.array()).sum();

    // τ = 1/ωₚ ≈ Tₚ / (2π)
    const double omega_p = 2.0 * M_PI / Tp;
    out.tau     = 1.0 / omega_p;
    out.sigma_a = std::sqrt(out.m4);
    out.R_S     = 4.0 * out.m0;

    // Hs from spectrum: Hs_spec = 4√m₀
    out.Hs_spec = 4.0 * std::sqrt(out.m0);

    // Mean periods
    out.Tm01    = (out.m1 > 0.0) ? (out.m0 / out.m1) : NAN;
    out.Tm02    = (out.m2 > 0.0) ? std::sqrt(out.m0 / out.m2) : NAN;
    out.Tm10    = (out.m0 > 0.0) ? (out.mneg1 / out.m0) : NAN;

    // Mean frequency
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

// === IMU-inflated tuning (includes bias drifts) ===
static TuningIMU compute_with_imu(const TuningSpec& base,
                                  const Eigen::Vector3d& sigma_acc,
                                  const Eigen::Vector3d& sigma_gyro,
                                  double sigma_bacc0,
                                  double q_bacc,
                                  double q_bgyr)
{
    TuningIMU t{};

    // Effective stationary accel std: ocean sigma_a + sensor white noise
    double sigma_acc_rms = sigma_acc.norm() / std::sqrt(3.0);
    double sigma_a_eff = std::sqrt(base.sigma_a * base.sigma_a +
                                   sigma_acc_rms * sigma_acc_rms +
                                   sigma_bacc0 * sigma_bacc0);
    t.sigma_a_eff = sigma_a_eff;

    // Effective tau reduced by gyro noise and bias RW
    double sigma_gyro_rms = sigma_gyro.norm() / std::sqrt(3.0);
    double tilt_var = (sigma_gyro_rms * sigma_gyro_rms + q_bgyr) * base.tau;
    t.tau_eff = base.tau / (1.0 + tilt_var);

    // R_S inflated by accelerometer noise + bias RW
    t.R_S_eff = base.R_S + 3.0 * (sigma_acc_rms * sigma_acc_rms + q_bacc);

    return t;
}

// === Generic runner ===
template<typename WaveModel>
void process_wave(const WaveParameters& wp,
                  size_t wave_index,
                  const std::string& type,
                  std::ofstream& file,
                  const Eigen::Vector3d& sigma_acc,
                  const Eigen::Vector3d& sigma_gyro,
                  double sigma_bacc0,
                  double q_bacc,
                  double q_bgyr)
{
    auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
        wp.direction * M_PI / 180.0, 10.0, 42u);

    WaveModel model(wp.height, wp.period, dirDist, 0.02, 0.8, 9.81, 42u);

    const auto t_spec = compute_from_spectrum<128>(
        model.frequencies(), model.spectrum(), model.df(), wp.period);
    const auto t_heur = compute_heuristic_from_HsTp(wp.height, wp.period);
    const auto t_imu  = compute_with_imu(t_spec, sigma_acc, sigma_gyro,
                                         sigma_bacc0, q_bacc, q_bgyr);

    // CSV
    file << wave_index << "," << type << ","
         << wp.height << "," << wp.period << ","
         << t_spec.tau << "," << t_spec.sigma_a << "," << t_spec.R_S << "," << t_spec.Hs_spec << ","
         << t_heur.tau << "," << t_heur.sigma_a << "," << t_heur.R_S << ","
         << t_imu.tau_eff << "," << t_imu.sigma_a_eff << "," << t_imu.R_S_eff << "\n";

    // Report
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Wave " << wave_index << " (" << type << ")\n"
              << "  Input: Hs=" << wp.height << " m, Tp=" << wp.period << " s\n"
              << "  Spectrum-derived: tau=" << t_spec.tau
              << " s, sigma_a=" << t_spec.sigma_a
              << " m/s², R_S=" << t_spec.R_S << " m²\n"
              << "  Heuristic: tau=" << t_heur.tau
              << " s, sigma_a=" << t_heur.sigma_a
              << " m/s², R_S=" << t_heur.R_S << " m²\n"
              << "  IMU-inflated: tau_eff=" << t_imu.tau_eff
              << " s, sigma_a_eff=" << t_imu.sigma_a_eff
              << " m/s², R_S_eff=" << t_imu.R_S_eff << " m²\n\n";
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
         << "tau_heur,sigma_a_heur,R_S_heur,"
         << "tau_imu,sigma_a_imu,R_S_imu\n";

    // Example IMU params (should match Kalman3D_Wave)
    Eigen::Vector3d sigma_acc(0.02, 0.02, 0.02);   // accel noise (m/s²)
    Eigen::Vector3d sigma_gyro(0.001, 0.001, 0.001); // gyro noise (rad/s)
    double sigma_bacc0 = 0.1;   // accel bias init std (m/s²)
    double q_bacc = 1e-8;       // accel bias RW intensity
    double q_bgyr = 1e-8;       // gyro bias RW intensity

    for (size_t idx = 0; idx < waveParamsList.size(); ++idx) {
        process_wave<Jonswap3dStokesWaves<128>>(waveParamsList[idx], idx, "JONSWAP",
                                                file, sigma_acc, sigma_gyro,
                                                sigma_bacc0, q_bacc, q_bgyr);
        process_wave<PMStokesN3dWaves<128,3>>(waveParamsList[idx], idx, "PMSTOKES",
                                              file, sigma_acc, sigma_gyro,
                                              sigma_bacc0, q_bacc, q_bgyr);
    }

    file.close();
    std::cout << "Wrote " << out_csv << "\n";
    return 0;
}
