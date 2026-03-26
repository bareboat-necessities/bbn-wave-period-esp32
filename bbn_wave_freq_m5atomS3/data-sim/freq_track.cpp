#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <iomanip>
#include <filesystem>

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float g_std = 9.80665f; // standard gravity acceleration m/s²

#include "WaveFilesSupport.h"
#include "FrequencyTrackerPolicy.h"
#include "FrequencySmoother.h"
#include "PLLFreqTracker.h"

// Config
static constexpr float SAMPLE_RATE_HZ = 200.0f;
static constexpr float DELTA_T        = 1.0f / SAMPLE_RATE_HZ;
static constexpr float NOISE_STDDEV   = 0.08f;
static constexpr float BIAS_MEAN      = 0.10f;
static constexpr float FREQ_MIN_HZ    = 0.04f;
static constexpr float FREQ_MAX_HZ    = 2.0f;
static constexpr double QUALITY_WINDOW_SECONDS = 5.0;
static constexpr double QUALITY_GATE_REL_MAX = 0.15;

// Trackers
TrackerPolicy<TrackerType::ARANOVSKIY> aranovskiyTracker;
TrackerPolicy<TrackerType::KALMANF> kalmanfTracker;
TrackerPolicy<TrackerType::PLL> pllTracker;
TrackerPolicy<TrackerType::ZEROCROSS> zcTracker;
FrequencySmoother<float> freqSmoother;

static bool kalm_smoother_first = true;
static double sim_t = 0.0;

// Helpers
static void init_kalmanf_backend() {
    kalmanfTracker.t.init(0.985f, 1e-6f, 1e+5f, 1.0f, 0.0f, 0.0f, 1.9999f);
}

static void init_tracker_backends() {
    init_kalmanf_backend();

    TrackerPolicy<TrackerType::PLL>::Config cfg;
    cfg.f_min_hz = FREQ_MIN_HZ;
    cfg.f_max_hz = FREQ_MAX_HZ;
    cfg.f_init_hz = FREQ_GUESS;
    cfg.pre_hp_hz = 0.03;
    cfg.pre_lp_hz = 2.5;
    cfg.demod_lp_hz = 0.12;
    cfg.loop_bandwidth_hz = 0.05;
    cfg.output_smooth_tau_s = 1.0;
    cfg.lock_rms_min = 0.002;
    pllTracker.configure(cfg);
}

static void reset_run_state() {
    sim_t = 0.0;
    kalm_smoother_first = true;
    aranovskiyTracker = TrackerPolicy<TrackerType::ARANOVSKIY>();
    kalmanfTracker = TrackerPolicy<TrackerType::KALMANF>();
    init_kalmanf_backend();
    zcTracker = TrackerPolicy<TrackerType::ZEROCROSS>();
    freqSmoother = FrequencySmoother<float>();
    pllTracker.reset(FREQ_GUESS);
}

static double clamp_freq(double v) {
    return std::clamp(v, static_cast<double>(FREQ_MIN_HZ), static_cast<double>(FREQ_MAX_HZ));
}

static void write_csv_header(std::ofstream &ofs) {
    ofs << "time,true_freq,est_freq,smooth_freq,error,smooth_error,"
           "abs_error,abs_smooth_error,update_flag\n";
}

static void write_csv_line(std::ofstream &ofs,
                           double time, double true_freq,
                           double est_freq, double smooth_freq,
                           double error, double smooth_error,
                           double abs_error, double abs_smooth_error,
                           bool updated) {
    ofs << std::fixed << std::setprecision(6)
        << time << ","
        << true_freq << ",";
    if (std::isnan(est_freq)) ofs << "nan,";
    else ofs << est_freq << ",";
    if (std::isnan(smooth_freq)) ofs << "nan,";
    else ofs << smooth_freq << ",";
    if (std::isnan(error)) ofs << "nan,";
    else ofs << error << ",";
    if (std::isnan(smooth_error)) ofs << "nan,";
    else ofs << smooth_error << ",";
    if (std::isnan(abs_error)) ofs << "nan,";
    else ofs << abs_error << ",";
    if (std::isnan(abs_smooth_error)) ofs << "nan,";
    else ofs << abs_smooth_error << ",";
    ofs << (updated ? 1 : 0) << "\n";
}

static std::pair<double,double> run_tracker_once(TrackerType tracker,
                                               float a_raw, float dt) {
    double freq = FREQ_GUESS;
    double smooth_freq = std::numeric_limits<double>::quiet_NaN();
    if (tracker == TrackerType::ARANOVSKIY) {
        freq = aranovskiyTracker.run(a_raw, dt);
    } else if (tracker == TrackerType::KALMANF) {
        freq = kalmanfTracker.run(a_raw, dt);
    } else if (tracker == TrackerType::PLL) {
        pllTracker.update(a_raw, dt);
        freq = pllTracker.getRawFrequencyHz();
        smooth_freq = pllTracker.getFrequencyHz();
    } else {
        freq = zcTracker.run(a_raw, dt);
    }
    return {freq, smooth_freq};
}

struct TrackerRunSummary {
    std::string tracker_name;
    std::string output_file;
    double tail_mean_smooth_hz = std::numeric_limits<double>::quiet_NaN();
    std::size_t tail_samples = 0;
};

// Main runner
static TrackerRunSummary run_from_csv(TrackerType tracker,
                                      const std::string &csv_file,
                                      unsigned run_seed) {
    // Parse metadata
    auto parsed = WaveFileNaming::parse(csv_file);
    if (!parsed) {
        fprintf(stderr, "Could not parse metadata from %s\n", csv_file.c_str());
        return {};
    }
    WaveFileNaming::ParsedName meta = *parsed;
    std::string waveName = EnumTraits<WaveType>::to_string(meta.type);

    // Tracker prefix
    std::string trackerName =
        (tracker == TrackerType::ARANOVSKIY) ? "aranovskiy" :
        (tracker == TrackerType::KALMANF)   ? "kalmanf" :
        (tracker == TrackerType::PLL) ? "pll" :
                                                    "zerocross";

    // Grab "_H..._L..._A..._P..." tail
    std::string stem = std::filesystem::path(csv_file).filename().string();
    auto posH = stem.find("_H");
    std::string tail = (posH != std::string::npos) ? stem.substr(posH) : "";

    // Remove .csv extension
    if (tail.size() > 4 && tail.substr(tail.size()-4) == ".csv") {
        tail = tail.substr(0, tail.size()-4);
    }

    // Encode noise & bias
    char nb[64];
    std::snprintf(nb, sizeof(nb), "_N%.2f_B%.2f", NOISE_STDDEV, BIAS_MEAN);

    // Output file
    std::string outFile = "freq_track_" + trackerName + "_" + waveName + tail + nb + ".csv";
    std::ofstream ofs(outFile);
    if (!ofs.is_open()) {
        fprintf(stderr, "Failed to open %s\n", outFile.c_str());
        return {};
    }
    write_csv_header(ofs);

    // Noise
    std::default_random_engine rng(run_seed);
    std::normal_distribution<float> gauss(0.0f, NOISE_STDDEV);
    float bias = BIAS_MEAN;
    std::vector<std::pair<double, double>> smooth_series;

    reset_run_state();

    // Process records
    WaveDataCSVReader reader(csv_file);
    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        float accel_z = (meta.type == WaveType::JONSWAP || meta.type == WaveType::PMSTOKES) ? rec.imu.acc_bz - g_std : rec.wave.acc_z;
        float noisy_accel = accel_z + bias + gauss(rng);
        auto [est_freq, tracker_smooth_freq] = run_tracker_once(tracker, noisy_accel, DELTA_T);
        const bool updated = !std::isnan(est_freq);

        double smooth_freq = std::numeric_limits<double>::quiet_NaN();
        if (tracker == TrackerType::PLL) {
            smooth_freq = tracker_smooth_freq;
        } else if (!std::isnan(est_freq) && updated) {
            if (kalm_smoother_first) {
                kalm_smoother_first = false;
                freqSmoother.setInitial(est_freq);
                smooth_freq = est_freq;
            } else {
                smooth_freq = freqSmoother.update(est_freq);
            }
        }
        if (!std::isnan(smooth_freq)) smooth_freq = clamp_freq(smooth_freq);
        if (!std::isnan(smooth_freq)) {
            smooth_series.emplace_back(rec.time, smooth_freq);
        }

        // True freq if length available
        double true_f = std::numeric_limits<double>::quiet_NaN();
        if (meta.length > 0.0) {
            double T = std::sqrt(meta.length / g_std * 2 * M_PI);
            true_f = 1.0 / T;
        }

        double error            = std::isnan(est_freq) ? NAN : (est_freq - true_f);
        double smooth_error     = std::isnan(smooth_freq) ? NAN : (smooth_freq - true_f);
        double abs_error        = std::isnan(error) ? NAN : std::fabs(error);
        double abs_smooth_error = std::isnan(smooth_error) ? NAN : std::fabs(smooth_error);

        write_csv_line(ofs, rec.time, true_f, est_freq, smooth_freq,
                       error, smooth_error, abs_error, abs_smooth_error, updated);

        sim_t = rec.time;
    });
    reader.close();

    ofs.close();
    printf("Wrote %s\n", outFile.c_str());

    TrackerRunSummary summary;
    summary.tracker_name = trackerName;
    summary.output_file = outFile;
    if (!smooth_series.empty()) {
        const double end_t = smooth_series.back().first;
        const double start_t = end_t - QUALITY_WINDOW_SECONDS;
        double sum = 0.0;
        for (const auto &sample : smooth_series) {
            if (sample.first >= start_t) {
                sum += sample.second;
                summary.tail_samples++;
            }
        }
        if (summary.tail_samples > 0) {
            summary.tail_mean_smooth_hz = sum / static_cast<double>(summary.tail_samples);
        }
    }
    return summary;
}

// Main
int main() {
    init_tracker_backends();

    // Gather candidate files
    std::vector<std::string> files;
    for (const auto &entry : std::filesystem::directory_iterator(".")) {
        if (!entry.is_regular_file()) continue;
        auto fname = entry.path().string();
        if (fname.find("wave_data_") == std::string::npos) continue;
        files.push_back(fname);
    }

    // Sort lexicographically for stable, repeatable order
    std::sort(files.begin(), files.end());

    // Run each file with each tracker
    unsigned run_idx = 0;
    for (const auto &fname : files) {
        std::vector<TrackerRunSummary> summaries;
        for (int tr = 0; tr < 4; ++tr) {
            summaries.push_back(run_from_csv(static_cast<TrackerType>(tr), fname, run_idx++));
        }

        printf("\nSummary for %s (last %.1f s smoothed frequency):\n",
               fname.c_str(), QUALITY_WINDOW_SECONDS);
        for (const auto &s : summaries) {
            if (std::isnan(s.tail_mean_smooth_hz)) {
                printf("  %-10s mean=nan (samples=%zu)\n", s.tracker_name.c_str(), s.tail_samples);
            } else {
                printf("  %-10s mean=%.6f Hz (samples=%zu)\n",
                       s.tracker_name.c_str(), s.tail_mean_smooth_hz, s.tail_samples);
            }
        }

        bool quality_ok = true;
        for (std::size_t i = 0; i < summaries.size(); ++i) {
            for (std::size_t j = i + 1; j < summaries.size(); ++j) {
                const double fi = summaries[i].tail_mean_smooth_hz;
                const double fj = summaries[j].tail_mean_smooth_hz;
                if (std::isnan(fi) || std::isnan(fj) || fi <= 0.0 || fj <= 0.0) {
                    quality_ok = false;
                    printf("  Pair %-10s vs %-10s: N/A (missing tail mean)\n",
                           summaries[i].tracker_name.c_str(), summaries[j].tracker_name.c_str());
                    continue;
                }
                const double denom = std::max(fi, fj);
                const double rel_diff = std::fabs(fi - fj) / denom;
                const bool pass = rel_diff <= QUALITY_GATE_REL_MAX;
                quality_ok = quality_ok && pass;
                printf("  Pair %-10s vs %-10s: rel_diff=%.2f%% -> %s (<= %.0f%%)\n",
                       summaries[i].tracker_name.c_str(), summaries[j].tracker_name.c_str(),
                       rel_diff * 100.0, pass ? "PASS" : "FAIL", QUALITY_GATE_REL_MAX * 100.0);
            }
        }

        printf("Quality gate for %s: %s\n\n", fname.c_str(), quality_ok ? "PASS" : "FAIL");
    }

    printf("All runs complete.\n");
    return 0;
}
