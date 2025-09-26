#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <memory>
#include <vector>

#define EIGEN_NON_ARDUINO
#include <Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- constants ---
static constexpr double TWO_PI = 2.0 * M_PI;
static constexpr double G_STD  = 9.80665;

// ---- wave model headers ----
#include "Jonswap3dStokesWaves.h"
#include "PiersonMoskowitzStokes3D_Waves.h"
#include "DirectionalSpread.h"
#include "WaveFilesSupport.h"   // WaveParameters (Hs,Tp,phase,dir)

// ===== Input test waves =====
const std::vector<WaveParameters> waveParamsList = {
    {3.0f,   0.27f, static_cast<float>(M_PI/3.0), 25.0f},
    {5.7f,   1.5f,  static_cast<float>(M_PI/1.5), 25.0f},
    {8.5f,   4.0f,  static_cast<float>(M_PI/6.0), 25.0f},
    {11.4f,  8.5f,  static_cast<float>(M_PI/2.5), 25.0f}
};

// ===== Results structs =====
struct TuningExact {
    double tau;       // s
    double sigma_a;   // m/s^2 (RMS)
    double R_S;       // (m*s)^2
    // diagnostics
    double Hs_spec;   // m
    double m0;        // m^2
    double m4;        // m^2 s^-4
    double m_neg2;    // m^2 s^2
    double f_peak;    // Hz
};

struct TuningHeur {
    double tau;       // s
    double sigma_a;   // m/s^2
    double R_S;       // (m*s)^2
    // diagnostics
    double m0_from_Hs; // m^2
    double omega_p;    // rad/s used
};

static double rel_err(double est, double ref) {
    if (ref == 0.0) return (est == 0.0) ? 0.0 : INFINITY;
    return std::abs(est - ref) / std::abs(ref);
}

// --- exact from spectrum ---
template<int N>
static TuningExact compute_exact_from_spectrum(
    const Eigen::Matrix<double, N, 1>& f_Hz,
    const Eigen::Matrix<double, N, 1>& S_eta,
    const Eigen::Matrix<double, N, 1>& df_Hz
) {
    TuningExact out{};
    int idx_max = 0;
    S_eta.maxCoeff(&idx_max);
    out.f_peak = f_Hz(idx_max);
    const double omega_p = std::max(TWO_PI * out.f_peak, 1e-9);

    const Eigen::Array<double, N, 1> f  = f_Hz.array();
    const Eigen::Array<double, N, 1> df = df_Hz.array();
    const Eigen::Array<double, N, 1> w  = TWO_PI * f;

    const Eigen::Array<double, N, 1> S  = S_eta.array();
    const Eigen::Array<double, N, 1> w2 = w.square();
    const Eigen::Array<double, N, 1> w4 = w2.square();

    out.m0     = (S * df).sum();                               // m^2
    out.m4     = (w4 * S * df).sum();                          // m^2 s^-4
    out.m_neg2 = ((S / (w2.max(1e-18))) * df).sum();           // m^2 s^2

    out.tau     = 1.0 / omega_p;
    out.sigma_a = (out.m4 > 0.0) ? std::sqrt(out.m4) : 0.0;
    out.R_S     = std::max(out.m_neg2, 0.0);
    out.Hs_spec = (out.m0 > 0.0) ? 4.0 * std::sqrt(out.m0) : 0.0;
    return out;
}

// --- heuristic (Hs,Tp) ---
static TuningHeur compute_heur_from_HsTp(double Hs, double Tp) {
    TuningHeur h{};
    h.m0_from_Hs = (Hs*Hs)/16.0;
    h.omega_p    = TWO_PI / std::max(Tp, 1e-9);
    h.tau        = 1.0 / h.omega_p;
    h.sigma_a    = h.omega_p*h.omega_p*std::sqrt(h.m0_from_Hs);
    h.R_S        = h.m0_from_Hs / (h.omega_p*h.omega_p);
    return h;
}

// ===== Pretty print =====
static void print_report(size_t idx, const char* type, const WaveParameters& wp,
                         const TuningExact& e, const TuningHeur& h1) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Wave " << idx << " [" << type << "]  (Hs_in=" << wp.height
              << " m, Tp_in=" << wp.period << " s)\n";
    std::cout << "  Peak freq f_p     = " << e.f_peak << " Hz  (ω_p = " << (TWO_PI*e.f_peak) << " rad/s)\n";
    std::cout << "  --- Exact from spectrum ---\n";
    std::cout << "    tau_exact        = " << e.tau     << " s\n";
    std::cout << "    sigma_a_exact    = " << e.sigma_a << " m/s^2\n";
    std::cout << "    R_S_exact        = " << e.R_S     << " (m*s)^2\n";
    std::cout << "    m0               = " << e.m0      << " m^2\n";
    std::cout << "    m4               = " << e.m4      << " m^2·s^-4\n";
    std::cout << "    m_{-2}           = " << e.m_neg2  << " m^2·s^2\n";
    std::cout << "    Hs_spec          = " << e.Hs_spec << " m\n";

    std::cout << "  --- Heuristic (Hs,Tp) ---\n";
    std::cout << "    tau_h            = " << h1.tau
              << "  | rel.err vs exact = " << rel_err(h1.tau, e.tau) << "\n";
    std::cout << "    sigma_a_h        = " << h1.sigma_a
              << "  | rel.err vs exact = " << rel_err(h1.sigma_a, e.sigma_a) << "\n";
    std::cout << "    R_S_h            = " << h1.R_S
              << "  | rel.err vs exact = " << rel_err(h1.R_S, e.R_S) << "\n";
    std::cout << "    m0(Hs)           = " << h1.m0_from_Hs << " m^2,  ω_p=" << h1.omega_p << " rad/s\n\n";
}

// ===== CSV writer =====
static void write_csv_header(std::ofstream& f) {
    f << "wave_index,wave_type,Hs_input,Tp,"
      << "tau_exact,sigma_a_exact,R_S_exact,Hs_spec,m0,m4,mneg2,f_peak,"
      << "tau_heur,sigma_a_heur,R_S_heur,m0_from_Hs,omega_p,"
      << "relerr_tau,relerr_sigma_a,relerr_R_S\n";
}

static void write_csv_row(std::ofstream& f, size_t idx, const char* type,
                          const WaveParameters& wp,
                          const TuningExact& e, const TuningHeur& h1) {
    f << idx << "," << type << ","
      << wp.height << "," << wp.period << ","
      << e.tau << "," << e.sigma_a << "," << e.R_S << "," << e.Hs_spec << ","
      << e.m0 << "," << e.m4 << "," << e.m_neg2 << "," << e.f_peak << ","
      << h1.tau << "," << h1.sigma_a << "," << h1.R_S << ","
      << h1.m0_from_Hs << "," << h1.omega_p << ","
      << rel_err(h1.tau, e.tau) << ","
      << rel_err(h1.sigma_a, e.sigma_a) << ","
      << rel_err(h1.R_S, e.R_S) << "\n";
}

// ===== One runner for a model type =====
template<typename WaveModel, int N>
static void run_model_for_wave(const WaveParameters& wp, size_t idx, const char* tag, std::ofstream& csv) {
    auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
        wp.direction * M_PI / 180.0, 10.0, 42u);

    WaveModel model(wp.height, wp.period, dirDist, 0.02, 0.8, G_STD, 42u);

    const auto f   = model.frequencies();
    const auto Sη  = model.spectrum();
    const auto df  = model.df();

    const auto exact = compute_exact_from_spectrum<N>(f, Sη, df);
    const auto heur1 = compute_heur_from_HsTp(wp.height, wp.period);

    print_report(idx, tag, wp, exact, heur1);
    write_csv_row(csv, idx, tag, wp, exact, heur1);
}

// ===== Main =====
int main() {
    std::ofstream csv("wave_tunings.csv");
    if (!csv.is_open()) {
        std::cerr << "Cannot open wave_tunings.csv for writing\n";
        return 1;
    }
    write_csv_header(csv);

    std::cout << "Exact + Heuristic derivation of tau, sigma_a, R_S\n"
              << "Definitions: tau=1/ωp,  sigma_a=sqrt∫ω^4 Sη df,  R_S=∫Sη/ω^2 df\n\n";

    for (size_t i = 0; i < waveParamsList.size(); ++i) {
        run_model_for_wave<Jonswap3dStokesWaves<128>, 128>(waveParamsList[i], i, "JONSWAP", csv);
        run_model_for_wave<PMStokesN3dWaves<128,3>, 128>(waveParamsList[i], i, "PMSTOKES", csv);
    }

    csv.close();
    std::cout << "Wrote wave_tunings.csv\n";
    return 0;
}
