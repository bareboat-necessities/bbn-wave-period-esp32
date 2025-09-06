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

// Config
static constexpr float SAMPLE_RATE_HZ = 240.0f;
static constexpr float DELTA_T        = 1.0f / SAMPLE_RATE_HZ;
static constexpr float NOISE_STDDEV   = 0.08f;
static constexpr float BIAS_MEAN      = 0.10f;

// Trackers
AranovskiyFilter<double> arFilter;
KalmANF<double> kalmANF;
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
    return freq;
}

// Write CSV header
static void write_csv_header(std::ofstream &ofs) {
    ofs << "time,omega_inst,narrowness,regularity,"
           "significant_wave_height,disp_freq_hz\n";
}

// Normalize numbers (e.g. H1.500 → H1.5, L12.340 → L12.34)
static std::string normalize_numbers(const std::string &s) {
    std::regex num_re(R"(([0-9]+\.[0-9]+))");
    std::smatch match;
    std::string result;
    std::string::const_iterator searchStart(s.cbegin());

    while (std::regex_search(searchStart, s.cend(), match, num_re)) {
        // Append before number
        result.append(match.prefix().first, match.prefix().second);

        // Normalize number
        std::string num = match.str();
        num.erase(num.find_last_not_of('0') + 1, std::string::npos);
        if (!num.empty() && num.back() == '.') num.pop_back();

        result.append(num);
        searchStart = match.suffix().first;
    }
    result.append(searchStart, s.cend());
    return result;
}

// Main runner from input wave_data_*.csv
static void run_from_csv(TrackerType tracker,
                         const std::string &csv_file,
                         unsigned run_seed) {
    // Parse metadata
    auto parsed = WaveFileNaming::parse(csv_file);
    if (!parsed) {
        fprintf(stderr, "Could not parse metadata from %s\n", csv_file.c_str());
        return;
    }
    WaveFileNaming::ParsedName meta = *parsed;
    std::string waveName = EnumTraits<WaveType>::to_string(meta.type);

    // Tracker prefix
    std::string trackerName =
        (tracker == TrackerType::ARANOVSKIY) ? "aranovskiy" :
        (tracker == TrackerType::KALMANF)   ? "kalmanf" :
                                              "zerocross";

    // Grab "_H..._L..._A..._P..." tail
    std::string stem = std::filesystem::path(csv_file).filename().string();
    auto posH = stem.find("_H");
    std::string tail = (posH != std::string::npos) ? stem.substr(posH) : "";

    // Remove trailing ".csv"
    if (tail.size() > 4 && tail.substr(tail.size() - 4) == ".csv") {
        tail = tail.substr(0, tail.size() - 4);
    }

    // Normalize numbers inside the tail
    tail = normalize_numbers(tail);

    // Append noise and bias into filename
    char noise_bias[64];
    std::snprintf(noise_bias, sizeof(noise_bias),
                  "_N%.3f_B%.3f", NOISE_STDDEV, BIAS_MEAN);

    // Output file
    std::string outFile = "regularity_" + trackerName + "_" + waveName + tail + noise_bias + ".csv";
    std::ofstream ofs(outFile);
    if (!ofs.is_open()) {
        fprintf(stderr, "Failed to open %s\n", outFile.c_str());
        return;
    }
    write_csv_header(ofs);

    // Noise
    std::default_random_engine rng(run_seed);
    std::normal_distribution<float> gauss(0.0f, NOISE_STDDEV);
    float bias = BIAS_MEAN;

    reset_run_state();

    SeaStateRegularity regFilter;

    // Process records
    WaveDataCSVReader reader(csv_file);
    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        float accel_z = rec.wave.acc_z;
        float noisy_accel = accel_z + bias + gauss(rng);
        float a_norm = noisy_accel / g_std;

        double freq = run_tracker_once(tracker, a_norm, DELTA_T);
        if (std::isfinite(freq)) {
            regFilter.update(DELTA_T, noisy_accel,
                             static_cast<float>(2.0 * M_PI * freq));
            ofs << rec.time << ","
                << (2.0 * M_PI * freq) << ","
                << regFilter.getNarrowness() << ","
                << regFilter.getRegularity() << ","
                << regFilter.getWaveHeightEnvelopeEst() << ","
                << regFilter.getDisplacementFrequencyHz() << "\n";
        }

        sim_t = rec.time;
    });
    reader.close();

    ofs.close();
    printf("Wrote %s\n", outFile.c_str());
}

// Main
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

    printf("All SeaStateRegularity runs complete.\n");
    return 0;
}
