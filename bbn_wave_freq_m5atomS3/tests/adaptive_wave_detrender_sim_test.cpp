#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define EIGEN_NON_ARDUINO

#include "AdaptiveWaveDetrender.h"
#include "PiersonMoskowitzStokes3D_Waves.h"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kGravity = 9.80665;
constexpr double kSampleRateHz = 200.0;
constexpr double kDtS = 1.0 / kSampleRateHz;
constexpr double kDurationS = 300.0;
constexpr double kWarmupMinS = 60.0;
constexpr double kRmsGateFractionOfHeight = 0.16;

struct Scenario {
  const char* tag;
  double height_m;
  double period_s;
  double phase_rad;
  double direction_deg;
};

struct RMSAccumulator {
  double sum_sq = 0.0;
  std::size_t count = 0U;

  void add(double value) {
    sum_sq += value * value;
    ++count;
  }

  double rms() const {
    return (count > 0U) ? std::sqrt(sum_sq / static_cast<double>(count)) : 0.0;
  }
};

const std::vector<Scenario> kScenarios = {
    {"low", 0.27, 3.0,  kPi / 3.0, 25.0},
    {"medium", 1.5, 5.7, kPi / 1.5, 25.0},
    {"large", 4.0, 8.5,  kPi / 6.0, 25.0},
    {"extreme", 8.5, 11.4, kPi / 2.5, 25.0},
};

double drift_signal(const Scenario& scenario, double wave_freq_hz, double time_s) {
  const double primary_freq_hz = std::max(0.0035, wave_freq_hz * 0.085);
  const double secondary_freq_hz = std::max(0.0020, primary_freq_hz * 0.41);
  const double primary_amp_m = 0.38 * scenario.height_m;
  const double secondary_amp_m = 0.14 * scenario.height_m;
  const double ramp_amp_m = 0.20 * scenario.height_m;
  const double normalized_time = (time_s / kDurationS) - 0.5;

  return primary_amp_m * std::sin((2.0 * kPi * primary_freq_hz * time_s) + 0.35) +
         secondary_amp_m * std::sin((2.0 * kPi * secondary_freq_hz * time_s) - 0.80) +
         ramp_amp_m * normalized_time;
}

std::string resolve_output_path(int argc, char* argv[], int arg_index, const char* fallback) {
  return (argc > arg_index) ? argv[arg_index] : std::string(fallback);
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const std::string samples_path = resolve_output_path(argc, argv, 1, "adaptive_wave_detrender_sim_output.csv");
    const std::string summary_path = resolve_output_path(argc, argv, 2, "adaptive_wave_detrender_sim_summary.csv");

    std::ofstream samples_ofs(samples_path);
    if (!samples_ofs.is_open()) {
      throw std::runtime_error("Unable to open output CSV: " + samples_path);
    }

    std::ofstream summary_ofs(summary_path);
    if (!summary_ofs.is_open()) {
      throw std::runtime_error("Unable to open summary CSV: " + summary_path);
    }

    samples_ofs << "scenario,height_m,period_s,wave_freq_hz,time_s,reference_z_m,drift_m,drifted_z_m,detrended_z_m,baseline_slow_m,error_z_m\n";
    samples_ofs << std::fixed << std::setprecision(9);

    summary_ofs << "scenario,height_m,period_s,wave_freq_hz,warmup_s,drift_rms_m,detrended_rms_m,gate_rms_m,improvement_ratio\n";
    summary_ofs << std::fixed << std::setprecision(9);

    bool all_ok = true;

    for (const Scenario& scenario : kScenarios) {
      const double wave_freq_hz = 1.0 / scenario.period_s;
      const double warmup_s = std::max(kWarmupMinS, 6.0 * scenario.period_s);
      const double gate_rms_m = kRmsGateFractionOfHeight * scenario.height_m;

      auto dir_dist = std::make_shared<Cosine2sRandomizedDistribution>(
          scenario.direction_deg * kPi / 180.0, 10.0, 42u);
      PMStokesN3dWaves<128, 3> wave_model(
          scenario.height_m,
          scenario.period_s,
          dir_dist,
          0.02,
          0.8,
          kGravity,
          42u);

      AdaptiveWaveDetrender detrender;

      RMSAccumulator drift_rms;
      RMSAccumulator detrended_rms;

      bool initialized = false;
      const std::size_t total_steps = static_cast<std::size_t>(std::ceil(kDurationS * kSampleRateHz));

      for (std::size_t step = 0; step < total_steps; ++step) {
        const double time_s = static_cast<double>(step) * kDtS;
        const auto state = wave_model.getLagrangianState(time_s);
        const double reference_z_m = state.displacement.z();
        const double drift_m = drift_signal(scenario, wave_freq_hz, time_s);
        const double drifted_z_m = reference_z_m + drift_m;

        AdaptiveWaveDetrender::Output output;
        if (!initialized) {
          detrender.reset(static_cast<float>(drifted_z_m));
          output = detrender.lastOutput();
          initialized = true;
        } else {
          output = detrender.update(static_cast<float>(drifted_z_m),
                                    static_cast<float>(kDtS),
                                    static_cast<float>(wave_freq_hz),
                                    true);
        }

        const double detrended_z_m = static_cast<double>(output.wave_clean);
        const double error_z_m = detrended_z_m - reference_z_m;

        if (time_s >= warmup_s) {
          drift_rms.add(drifted_z_m - reference_z_m);
          detrended_rms.add(error_z_m);
        }

        samples_ofs << scenario.tag << ','
                    << scenario.height_m << ','
                    << scenario.period_s << ','
                    << wave_freq_hz << ','
                    << time_s << ','
                    << reference_z_m << ','
                    << drift_m << ','
                    << drifted_z_m << ','
                    << detrended_z_m << ','
                    << static_cast<double>(output.baseline_slow) << ','
                    << error_z_m << '\n';
      }

      const double drift_rms_m = drift_rms.rms();
      const double detrended_rms_m = detrended_rms.rms();
      const double improvement_ratio = (detrended_rms_m > 0.0)
          ? (drift_rms_m / detrended_rms_m)
          : std::numeric_limits<double>::infinity();

      summary_ofs << scenario.tag << ','
                  << scenario.height_m << ','
                  << scenario.period_s << ','
                  << wave_freq_hz << ','
                  << warmup_s << ','
                  << drift_rms_m << ','
                  << detrended_rms_m << ','
                  << gate_rms_m << ','
                  << improvement_ratio << '\n';

      std::cout << scenario.tag
                << ": drift RMS=" << drift_rms_m
                << " m, detrended RMS=" << detrended_rms_m
                << " m, gate=" << gate_rms_m
                << " m, improvement=" << improvement_ratio << "x\n";

      if (!(detrended_rms_m <= gate_rms_m) || !(detrended_rms_m < drift_rms_m)) {
        all_ok = false;
        std::cerr << "Quality gate failed for scenario '" << scenario.tag << "'\n";
      }
    }

    std::cout << "Wrote simulation samples to " << samples_path << '\n';
    std::cout << "Wrote simulation summary to " << summary_path << '\n';

    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& ex) {
    std::cerr << "adaptive_wave_detrender_sim_test failed: " << ex.what() << '\n';
    return EXIT_FAILURE;
  }
}
