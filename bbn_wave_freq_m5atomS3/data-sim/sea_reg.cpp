#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <limits>
#include <iomanip>
#include <filesystem>
#include <regex>

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float g_std = 9.80665f; // standard gravity acceleration m/s²

#include "WaveFilesSupport.h"
#include "AranovskiyFilter.h"
#include "KalmANF.h"
#include "FrequencySmoother.h"
#include "KalmanForWaveBasic.h"
#include "KalmanWaveNumStableAlt.h"
#include "SchmittTriggerFrequencyDetector.h"
#include "KalmanSmoother.h"
#include "KalmanWaveDirection.h"
#include "WaveFilters.h"
#include "SeaStateRegularity.h"
#include "SeaMetrics.h"
#include "DirectionalSpread.h"       
#include "Jonswap3dStokesWaves.h"       
#include "PiersonMoskowitzStokes3D_Waves.h" 

// Config
static constexpr float SAMPLE_RATE_HZ   = 240.0f;
static constexpr float DELTA_T          = 1.0f / SAMPLE_RATE_HZ;
static constexpr float NOISE_STDDEV     = 0.08f;
static constexpr float BIAS_MEAN        = 0.10f;
static constexpr float WARMUP_SECONDS   = 60.0f; // Warmup duration

// Trackers
AranovskiyFilter<double> arFilter;
KalmANF<double> kalmANF;
FrequencySmoother<float> freqSmoother;
KalmanSmootherVars kalman_freq;
SchmittTriggerFrequencyDetector freqDetector(
    ZERO_CROSSINGS_HYSTERESIS, ZERO_CROSSINGS_PERIODS);

static double sim_t = 0.0;
static bool kalm_smoother_first = true;
static uint32_t now_us() { return static_cast<uint32_t>(sim_t * 1e6); }

// Reset trackers for each run
static void init_tracker_backends() {
    init_filters(&arFilter, &kalman_freq);
    init_filters_alt(&kalmANF, &kalman_freq);
}

static void reset_run_state() {
    sim_t = 0.0;
    kalm_smoother_first = true;
    kalman_smoother_init(&kalman_freq, 0.25f, 2.0f, 100.0f);
    freqSmoother = FrequencySmoother<float>();
    freqDetector.reset();
}

// Run tracker once and return frequency
static double run_tracker_once(TrackerType tracker,
                               float a_norm, float dt) {
    double freq = std::numeric_limits<double>::quiet_NaN();
    if (tracker == TrackerType::ARANOVSKIY) {
        freq = estimate_freq(Aranovskiy, &arFilter, &kalmANF,
                             &freqDetector, a_norm, a_norm, dt, now_us());
    } else if (tracker == TrackerType::KALMANF) {
        freq = estimate_freq(Kalm_ANF, &arFilter, &kalmANF,
                             &freqDetector, a_norm, a_norm, dt, now_us());
    } else {
        freq = estimate_freq(ZeroCrossing, &arFilter, &kalmANF,
                             &freqDetector, a_norm, a_norm, dt, now_us());
    }
    float smooth_freq;
    if (kalm_smoother_first) {
        kalm_smoother_first = false;
        freqSmoother.setInitial(freq);
        smooth_freq = freq;
    } else {
        smooth_freq = freqSmoother.update(freq);
    }
    return smooth_freq;
}

// Write CSV header
static void write_csv_header(std::ofstream &ofs) {
    ofs << "time,omega_inst,narrowness,regularity,"
           "significant_wave_height,disp_freq_hz,disp_period_s,accel_var\n";
}

// Normalize numbers (e.g. H1.500 → H1.5, L12.340 → L12.34)
static std::string normalize_numbers(const std::string &s) {
    std::regex num_re(R"(([0-9]+\.[0-9]+))");
    std::smatch match;
    std::string result;
    std::string::const_iterator searchStart(s.cbegin());

    while (std::regex_search(searchStart, s.cend(), match, num_re)) {
        result.append(match.prefix().first, match.prefix().second);

        std::string num = match.str();
        num.erase(num.find_last_not_of('0') + 1, std::string::npos);
        if (!num.empty() && num.back() == '.') num.pop_back();

        result.append(num);
        searchStart = match.suffix().first;
    }
    result.append(searchStart, s.cend());
    return result;
}

// Struct to capture converged values
struct ConvergedStats {
    float regularity = 0.0f;
    float narrowness = 0.0f;
    float Hs = 0.0f;
    float disp_freq_hz = 0.0f;
    float disp_period_s = 0.0f;
    double tracker_freq_hz = 0.0;
};

// Per-file summary
struct FileSummary {
    std::string label; // wave type, height, period
    std::array<ConvergedStats,3> stats;
};

// Main runner from input wave_data_*.csv
static ConvergedStats run_from_csv(TrackerType tracker,
                                   const std::string &csv_file,
                                   unsigned run_seed) {
    ConvergedStats stats;
    auto parsed = WaveFileNaming::parse(csv_file);
    if (!parsed) {
        fprintf(stderr, "Could not parse metadata from %s\n", csv_file.c_str());
        return stats;
    }
    WaveFileNaming::ParsedName meta = *parsed;
    std::string waveName = EnumTraits<WaveType>::to_string(meta.type);

    std::string trackerName =
        (tracker == TrackerType::ARANOVSKIY) ? "aranovskiy" :
        (tracker == TrackerType::KALMANF)   ? "kalmanf" :
                                              "zerocross";

    std::string stem = std::filesystem::path(csv_file).filename().string();
    auto posH = stem.find("_H");
    std::string tail = (posH != std::string::npos) ? stem.substr(posH) : "";
    if (tail.size() > 4 && tail.substr(tail.size() - 4) == ".csv")
        tail = tail.substr(0, tail.size() - 4);
    tail = normalize_numbers(tail);

    char noise_bias[64];
    std::snprintf(noise_bias, sizeof(noise_bias),
                  "_N%.3f_B%.3f", NOISE_STDDEV, BIAS_MEAN);

    std::string outFile =
        "regularity_" + trackerName + "_" + waveName + tail + noise_bias + ".csv";
    std::ofstream ofs(outFile);
    if (!ofs.is_open()) {
        fprintf(stderr, "Failed to open %s\n", outFile.c_str());
        return stats;
    }
    write_csv_header(ofs);

    std::default_random_engine rng(run_seed);
    std::normal_distribution<float> gauss(0.0f, NOISE_STDDEV);
    float bias = BIAS_MEAN;

    reset_run_state();

    SeaStateRegularity regFilter;
    double last_freq = std::numeric_limits<double>::quiet_NaN();

    WaveDataCSVReader reader(csv_file);
    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        float accel_z = rec.wave.acc_z;
        float noisy_accel = accel_z + bias + gauss(rng);
        float a_norm = noisy_accel / g_std;

        double freq = run_tracker_once(tracker, a_norm, DELTA_T);
        if (std::isfinite(freq)) {
            last_freq = freq;
            if (rec.time >= WARMUP_SECONDS) {
                regFilter.update(DELTA_T, noisy_accel,
                                 static_cast<float>(2.0 * M_PI * freq));
                ofs << rec.time << ","
                    << (2.0 * M_PI * freq) << ","
                    << regFilter.getNarrowness() << ","
                    << regFilter.getRegularity() << ","
                    << regFilter.getWaveHeightEnvelopeEst() << ","
                    << regFilter.getDisplacementFrequencyHz() << ","
                    << regFilter.getDisplacementPeriodSec() << ","
                    << regFilter.getAccelerationVariance() << "\n";
            }
        }
        sim_t = rec.time;
    });
    reader.close();
    ofs.close();

    // === Write final converged *averaged* spectrum with reference built from model ===
    {
        const auto &A = regFilter.getAveragedSpectrum();

        std::string specFile = "reg_spectrum_" + trackerName + "_" +
                               waveName + tail + noise_bias + ".csv";
        std::ofstream spec(specFile);
        if (!spec.is_open()) {
            fprintf(stderr, "Failed to open %s\n", specFile.c_str());
            return stats;
        }

        // Build theoretical reference spectrum from parsed wave parameters
        std::vector<double> f_ref_hz_vec, S_ref_hz_vec;
        if (auto parsed_params = WaveFileNaming::parse_to_params(csv_file)) {
            auto [kind_ref, type_ref, wp] = *parsed_params;
            auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
                wp.direction * M_PI / 180.0, 10.0, 42u);

            Eigen::Matrix<double,128,1> f_ref, S_ref;
            if (type_ref == WaveType::JONSWAP) {
                Jonswap3dStokesWaves<128> refModel(
                    wp.height, wp.period, dirDist,
                    0.03, 1.5, 3.3, g_std, 42u);
                f_ref = refModel.frequencies();
                S_ref = refModel.spectrum();
            } else if (type_ref == WaveType::PMSTOKES) {
                PMStokesN3dWaves<128,3> refModel(
                    wp.height, wp.period, dirDist,
                    0.03, 1.5, g_std, 42u);
                f_ref = refModel.frequencies();
                S_ref = refModel.spectrum();
            }

            if (f_ref.size() > 0) {
                f_ref_hz_vec.resize(f_ref.size());
                S_ref_hz_vec.resize(S_ref.size());
                for (int i = 0; i < f_ref.size(); ++i) {
                    f_ref_hz_vec[i] = f_ref(i);
                    S_ref_hz_vec[i] = S_ref(i);
                }
            }
        }

        // CSV header
        spec << "freq_hz,S_eta_hz,S_ref_interp,S_ratio,"
                 "A_eta_est,A_eta_ref,E_eta_est,E_eta_ref,"
                 "CumVar_est,CumVar_ref\n";

        double cum_est = 0.0, cum_ref = 0.0;

        for (int k = 0; k < A.size(); ++k) {
            double f_est = A.freq_hz[k];
            double S_eta_hz = A.valueHz(k);
            double delta_f = A.domega[k] / (2.0 * M_PI);

            // Interpolate theoretical reference S_eta
            double s_ref_interp = 0.0;
            if (f_ref_hz_vec.size() >= 2) {
                if (f_est <= f_ref_hz_vec.front())
                    s_ref_interp = S_ref_hz_vec.front();
                else if (f_est >= f_ref_hz_vec.back())
                    s_ref_interp = S_ref_hz_vec.back();
                else {
                    size_t j = 0;
                    while (j + 1 < f_ref_hz_vec.size() &&
                           f_ref_hz_vec[j + 1] < f_est)
                        ++j;
                    double f0 = f_ref_hz_vec[j], f1 = f_ref_hz_vec[j + 1];
                    double s0 = S_ref_hz_vec[j], s1 = S_ref_hz_vec[j + 1];
                    double t = (f_est - f0) / (f1 - f0);
                    s_ref_interp = (1.0 - t) * s0 + t * s1;
                }
            }

            double ratio = (s_ref_interp > 0.0) ? (S_eta_hz / s_ref_interp) : 0.0;

            // Log-bin aware amplitudes
            double A_eta_est = std::sqrt(std::max(0.0, 2.0 * S_eta_hz * delta_f * f_est));
            double A_eta_ref = std::sqrt(std::max(0.0, 2.0 * s_ref_interp * delta_f * f_est));

            // Energy per log frequency
            double E_eta_est = f_est * S_eta_hz;
            double E_eta_ref = f_est * s_ref_interp;

            // Cumulative integrals
            cum_est += S_eta_hz * delta_f;
            cum_ref += s_ref_interp * delta_f;

            spec << f_est << "," << S_eta_hz << ","
                 << s_ref_interp << "," << ratio << ","
                 << A_eta_est << "," << A_eta_ref << ","
                 << E_eta_est << "," << E_eta_ref << ","
                 << cum_est << "," << cum_ref << "\n";
        }

        // Footer
        spec << "\n# Hs=" << regFilter.getWaveHeightEnvelopeEst()
             << ", nu=" << regFilter.getNarrowness()
             << ", f_disp=" << regFilter.getDisplacementFrequencyHz() << " Hz\n";

        spec.close();
        printf("Wrote %s\n", specFile.c_str());
    }
    
    // Capture final convergence stats
    stats.regularity    = regFilter.getRegularity();
    stats.narrowness    = regFilter.getNarrowness();
    stats.Hs            = regFilter.getWaveHeightEnvelopeEst();
    stats.disp_freq_hz  = regFilter.getDisplacementFrequencyHz();
    stats.disp_period_s = regFilter.getDisplacementPeriodSec();
    stats.tracker_freq_hz = last_freq;

    printf("Wrote %s\n", outFile.c_str());
    return stats;
}

int main() {
    init_tracker_backends();

    std::vector<std::string> files;
    for (const auto &entry : std::filesystem::directory_iterator(".")) {
        if (!entry.is_regular_file()) continue;
        auto fname = entry.path().string();
        if (fname.find("wave_data_") == std::string::npos) continue;
        files.push_back(fname);
    }
    std::sort(files.begin(), files.end());

    unsigned run_idx = 0;
    std::vector<FileSummary> all_summaries;

    for (const auto &fname : files) {
        auto parsed = WaveFileNaming::parse(fname);
        if (!parsed) {
            std::cout << "Skipping (unparsed): " << fname << "\n";
            continue;
        }

        const auto meta = *parsed;

        // only keep JONSWAP and PMSTOKES
        if (meta.type != WaveType::JONSWAP && meta.type != WaveType::PMSTOKES) {
            continue;
        }

        std::ostringstream oss;
        oss << EnumTraits<WaveType>::to_string(meta.type);
        if (meta.height > 0) oss << " H" << meta.height;
        if (meta.length > 0) oss << " L" << meta.length;
        const std::string label = oss.str();

        FileSummary summary;
        summary.label = label;

        for (int tr = 0; tr < 3; ++tr) {
            summary.stats[tr] = run_from_csv(static_cast<TrackerType>(tr), fname, run_idx++);
        }

        all_summaries.push_back(summary);
    }

    // === Final summary ===
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "\n=== Final Comparison Summary ===\n";
    std::cout << std::setw(10) << "Reg(Aran)"
              << std::setw(10) << "Reg(Kalm)"
              << std::setw(10) << "Reg(Zero)"
              << std::setw(9)  << "Hs(Aran)"
              << std::setw(9)  << "Hs(Kalm)"
              << std::setw(9)  << "Hs(Zero)"
              << std::setw(11) << "Freq(Aran)"
              << std::setw(11) << "Freq(Kalm)"
              << std::setw(11) << "Freq(Zero)"
              << std::setw(9)  << "Tp(Aran)"
              << std::setw(9)  << "Tp(Kalm)"
              << std::setw(9)  << "Tp(Zero)"
              << std::setw(9)  << "Nu(Aran)"
              << std::setw(9)  << "Nu(Kalm)"
              << std::setw(9)  << "Nu(Zero)"
              << std::setw(11) << "TrkF(Aran)"
              << std::setw(11) << "TrkF(Kalm)"
              << std::setw(11) << "TrkF(Zero)"
              << std::setw(23) << "Wave"
              << "\n";

    for (const auto& s : all_summaries) {
        std::cout << std::setw(10) << s.stats[0].regularity
                  << std::setw(10) << s.stats[1].regularity
                  << std::setw(10) << s.stats[2].regularity
                  << std::setw(9)  << s.stats[0].Hs
                  << std::setw(9)  << s.stats[1].Hs
                  << std::setw(9)  << s.stats[2].Hs
                  << std::setw(11) << s.stats[0].disp_freq_hz
                  << std::setw(11) << s.stats[1].disp_freq_hz
                  << std::setw(11) << s.stats[2].disp_freq_hz
                  << std::setw(9)  << s.stats[0].disp_period_s
                  << std::setw(9)  << s.stats[1].disp_period_s
                  << std::setw(9)  << s.stats[2].disp_period_s
                  << std::setw(9)  << s.stats[0].narrowness
                  << std::setw(9)  << s.stats[1].narrowness
                  << std::setw(9)  << s.stats[2].narrowness
                  << std::setw(11) << s.stats[0].tracker_freq_hz
                  << std::setw(11) << s.stats[1].tracker_freq_hz
                  << std::setw(11) << s.stats[2].tracker_freq_hz
                  << std::left << " " << std::setw(23) << s.label
                  << std::right << "\n";
    }

    std::cout << "All SeaStateRegularity runs complete.\n";
    return 0;
}
