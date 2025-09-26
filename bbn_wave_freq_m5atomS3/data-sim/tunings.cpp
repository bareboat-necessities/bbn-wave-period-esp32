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
static constexpr double g_std  = 9.80665;

// --- new R_S law ---
constexpr float kf = 0.748f;
inline float R_S_law(float Tp, float coeff = kf) {
    return coeff * std::pow(Tp, 1.0 / 3.0);
}

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
    double tau_displ;   // s (from displacement peak)
    double tau_accel;   // s (from acceleration peak)
    double sigma_a;     // m/s^2 (RMS)
    double R_S;         // (m*s)^2 from law
    // diagnostics
    double Hs_spec;     // m
    double m0;          // m^2
    double m4;          // m^2 s^-4
    double f_peak_disp; // Hz
    double f_peak_acc;  // Hz
};

struct TuningHeur {
    double tau;       // s
    double sigma_a;   // m/s^2
    double R_S;       // (m*s)^2 from law
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
    const Eigen::Matrix<double, N, 1>& df_Hz,
    double Tp
) {
    TuningExact out{};

    // displacement spectrum peak
    int idx_disp_peak = 0;
    S_eta.maxCoeff(&idx_disp_peak);
    out.f_peak_disp = f_Hz(idx_disp_peak);
    const double omega_disp = std::max(TWO_PI * out.f_peak_disp, 1e-9);

    // acceleration spectrum = ω^4 Sη
    Eigen::Array<double, N, 1> w  = TWO_PI * f_Hz.array();
    Eigen::Array<double, N, 1> w2 = w.square();
    Eigen::Array<double, N, 1> w4 = w2.square();
    Eigen::Array<double, N, 1> S_acc = w4 * S_eta.array();

    int idx_acc_peak = 0;
    S_acc.maxCoeff(&idx_acc_peak);
    out.f_peak_acc = f_Hz(idx_acc_peak);
    const double omega_acc = std::max(w(idx_acc_peak), 1e-9);

    // moments
    const Eigen::Array<double, N, 1> df = df_Hz.array();
    out.m0     = (S_eta.array() * df).sum();                    // m^2
    out.m4     = (w4 * S_eta.array() * df).sum();               // m^2 s^-4

    // τ from both peaks
    out.tau_displ = 1.0 / omega_disp;
    out.tau_accel = 1.0 / omega_acc;

    // sigma_a, R_S, Hs
    out.sigma_a = (out.m4 > 0.0) ? std::sqrt(out.m4) : 0.0;
    out.R_S     = R_S_law(Tp);   // <--- law
    out.Hs_spec = (out.m0 > 0.0) ? 4.0 * std::sqrt(out.m0) : 0.0;

    return out;
}

// --- heuristic (Hs,Tp) using new R_S law ---
static TuningHeur compute_heur_from_HsTp(double Hs, double Tp) {
    TuningHeur h{};
    h.m0_from_Hs = (Hs*Hs)/16.0;
    h.omega_p    = TWO_PI / std::max(Tp, 1e-9);
    h.tau        = 1.0 / h.omega_p;
    h.sigma_a    = h.omega_p*h.omega_p*std::sqrt(h.m0_from_Hs);
    h.R_S        = R_S_law(Tp);   // <--- law
    return h;
}

// ===== Pretty print =====
static void print_report(size_t idx, const char* type, const WaveParameters& wp,
                         const TuningExact& e, const TuningHeur& h1) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Wave " << idx << " [" << type << "]  (Hs_in=" << wp.height
              << " m, Tp_in=" << wp.period << " s)\n";
    std::cout << "  --- Peaks ---\n";
    std::cout << "    f_peak_disp      = " << e.f_peak_disp << " Hz\n";
    std::cout << "    f_peak_acc       = " << e.f_peak_acc  << " Hz\n";
    std::cout << "  --- Exact (law applied) ---\n";
    std::cout << "    tau_from_displ   = " << e.tau_displ << " s\n";
    std::cout << "    tau_from_accel   = " << e.tau_accel << " s\n";
    std::cout << "    sigma_a_exact    = " << e.sigma_a   << " m/s^2\n";
    std::cout << "    R_S (law)        = " << e.R_S       << " (m*s)^2\n";
    std::cout << "    m0               = " << e.m0        << " m^2\n";
    std::cout << "    m4               = " << e.m4        << " m^2·s^-4\n";
    std::cout << "    Hs_spec          = " << e.Hs_spec   << " m\n";

    std::cout << "  --- Heuristic (Hs,Tp with law) ---\n";
    std::cout << "    tau_h            = " << h1.tau << "\n";
    std::cout << "    sigma_a_h        = " << h1.sigma_a << "\n";
    std::cout << "    R_S_h (law)      = " << h1.R_S << " (m*s)^2\n\n";
}

// ===== CSV writer =====
static void write_csv_header(std::ofstream& f) {
    f << "wave_index,wave_type,Hs_input,Tp,"
      << "tau_displ,tau_accel,sigma_a_exact,R_S_law,Hs_spec,"
      << "m0,m4,f_peak_disp,f_peak_acc,"
      << "tau_heur,sigma_a_heur,R_S_law_heur\n";
}

static void write_csv_row(std::ofstream& f, size_t idx, const char* type,
                          const WaveParameters& wp,
                          const TuningExact& e, const TuningHeur& h1) {
    f << idx << "," << type << ","
      << wp.height << "," << wp.period << ","
      << e.tau_displ << "," << e.tau_accel << ","
      << e.sigma_a << "," << e.R_S << "," << e.Hs_spec << ","
      << e.m0 << "," << e.m4 << ","
      << e.f_peak_disp << "," << e.f_peak_acc << ","
      << h1.tau << "," << h1.sigma_a << "," << h1.R_S << "\n";
}

// ===== One runner for a model type =====
template<typename WaveModel, int N>
static void run_model_for_wave(const WaveParameters& wp, size_t idx, const char* tag, std::ofstream& csv) {
    auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
        wp.direction * M_PI / 180.0, 10.0, 42u);

    WaveModel model(wp.height, wp.period, dirDist, 0.02, 0.8, g_std, 42u);

    const auto f   = model.frequencies();
    const auto S   = model.spectrum();
    const auto df  = model.df();

    const auto exact = compute_exact_from_spectrum<N>(f, S, df, wp.period);
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
              << "R_S law: R_S(Tp) = kf * Tp^(1/3),  kf=" << kf << "\n\n";

    for (size_t i = 0; i < waveParamsList.size(); ++i) {
        run_model_for_wave<Jonswap3dStokesWaves<128>, 128>(waveParamsList[i], i, "JONSWAP", csv);
        run_model_for_wave<PMStokesN3dWaves<128,3>, 128>(waveParamsList[i], i, "PMSTOKES", csv);
    }

    csv.close();
    std::cout << "Wrote wave_tunings.csv\n";
    return 0;
}
