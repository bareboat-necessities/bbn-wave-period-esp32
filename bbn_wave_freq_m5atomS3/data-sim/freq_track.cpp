#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <limits>
#include <iomanip>
#include <filesystem>

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float g_std = 9.80665; // standard gravity acceleration m/s²

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

// Config
static constexpr float SAMPLE_RATE_HZ = 240.0f;
static constexpr float DELTA_T        = 1.0f / SAMPLE_RATE_HZ;
static constexpr float NOISE_STDDEV   = 0.08f;
static constexpr float BIAS_MEAN      = 0.10f;

// Trackers
AranovskiyFilter<double> arFilter;
KalmANF<double> kalmANF;
FrequencySmoother<float> freqSmoother;
KalmanSmootherVars kalman_freq;
SchmittTriggerFrequencyDetector freqDetector(ZERO_CROSSINGS_HYSTERESIS, ZERO_CROSSINGS_PERIODS);

static bool kalm_smoother_first = true;
static double sim_t = 0.0;
static uint32_t now_us() { return static_cast<uint32_t>(sim_t * 1e6); }

// ---------- Helpers ----------
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

static double clamp_freq(double v) {
    return clamp(v, (double)FREQ_LOWER, (double)FREQ_UPPER);
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

static std::pair<double,bool> run_tracker_once(TrackerType tracker,
                                               float a_norm, float a_raw, float dt) {
    double freq = FREQ_GUESS;
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
    return {freq, !std::isnan(freq)};
}

// ---------- Main runner ----------
static void run_from_csv(TrackerType tracker,
                         const std::string &csv_file,
                         unsigned run_seed) {
    // Parse metadata from filename
    auto parsed = WaveFileNaming::parse(csv_file);
    std::string waveName = parsed ? EnumTraits<WaveType>::to_string(parsed->type)
                                  : "unknown";
    double height = parsed ? parsed->height : 0.0;

    // output file
    std::string trackerName = (tracker == TrackerType::ARANOVSKIY) ? "aranovskiy" :
                              (tracker == TrackerType::KALMANF) ? "kalmanf" :
                              "zerocrossing";
    char buf[128];
    std::snprintf(buf, sizeof(buf), "tracker_%s_%s_H%.3f.csv",
                  trackerName.c_str(), waveName.c_str(), height);
    std::ofstream ofs(buf);
    if (!ofs.is_open()) { fprintf(stderr, "Failed to open %s\n", buf); return; }
    write_csv_header(ofs);

    // noise generator
    std::default_random_engine rng(run_seed);
    std::normal_distribution<float> gauss(0.0f, NOISE_STDDEV);
    float bias = BIAS_MEAN;

    reset_run_state();

    // use WaveDataCSVReader instead of manual parsing ✅
    WaveDataCSVReader reader(csv_file);
    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        float accel_z = rec.wave.acc_z;

        float noise_val   = gauss(rng);
        float noisy_accel = accel_z + bias + noise_val;
        float a_norm      = noisy_accel / g_std;

        auto [est_freq, updated] = run_tracker_once(tracker, a_norm, noisy_accel, DELTA_T);

        double smooth_freq = std::numeric_limits<double>::quiet_NaN();
        if (!std::isnan(est_freq) && updated) {
            if (kalm_smoother_first) {
                kalm_smoother_first = false;
                freqSmoother.setInitial(est_freq);
                smooth_freq = est_freq;
            } else {
                smooth_freq = freqSmoother.update(est_freq);
            }
        }
        if (!std::isnan(smooth_freq)) smooth_freq = clamp_freq(smooth_freq);

        // no true freq available in file
        double true_f = std::numeric_limits<double>::quiet_NaN();

        double error = std::isnan(est_freq) ? NAN : (est_freq - true_f);
        double smooth_error = std::isnan(smooth_freq) ? NAN : (smooth_freq - true_f);
        double abs_error = std::isnan(error) ? NAN : std::fabs(error);
        double abs_smooth_error = std::isnan(smooth_error) ? NAN : std::fabs(smooth_error);

        write_csv_line(ofs, rec.time, true_f, est_freq, smooth_freq,
                       error, smooth_error, abs_error, abs_smooth_error, updated);

        sim_t = rec.time;
    });
    reader.close();

    ofs.close();
    printf("Wrote %s\n", buf);
}

// ---------- Main ----------
int main() {
    init_tracker_backends();

    unsigned run_idx = 0;
    for (const auto &entry : std::filesystem::directory_iterator(".")) {
        if (!entry.is_regular_file()) continue;
        auto fname = entry.path().string();
        if (fname.find("wave_data_") == std::string::npos) continue;

        for (int tr = 0; tr < 3; ++tr) {
            run_from_csv(static_cast<TrackerType>(tr), fname, run_idx++);
        }
    }

    printf("All runs complete.\n");
    return 0;
}
