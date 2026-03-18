#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "AdaptiveWaveDetrender.h"

namespace {

struct ReferenceRow {
  double x_axis = 0.0;
  double original_from_screenshot_cm = 0.0;
  double baseline_slow = 0.0;
  double wave_raw = 0.0;
  double wave_clean = 0.0;
  double wave_freq_hz = 0.0;
  double wave_period_s = 0.0;
  double baseline_cutoff_hz = 0.0;
  double baseline_tau_s = 0.0;
  double cleanup_cutoff_hz = 0.0;
  double slope_rms = 0.0;
  double slope_threshold = 0.0;
  bool freq_valid = false;
  int schmitt_state = 0;
};

struct ErrorStats {
  const char* name = nullptr;
  double tolerance = 0.0;
  double max_abs_error = 0.0;
};

std::string trim(const std::string& value) {
  std::size_t start = 0;
  while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
    ++start;
  }

  std::size_t end = value.size();
  while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    --end;
  }

  return value.substr(start, end - start);
}

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, ',')) {
    fields.push_back(trim(field));
  }
  return fields;
}

bool parse_bool(const std::string& value) {
  std::string lowered;
  lowered.reserve(value.size());
  for (char ch : value) {
    lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }

  if (lowered == "true" || lowered == "1") {
    return true;
  }
  if (lowered == "false" || lowered == "0") {
    return false;
  }

  throw std::runtime_error("Unexpected boolean value: " + value);
}

ReferenceRow parse_row(const std::vector<std::string>& fields) {
  if (fields.size() != 14U) {
    throw std::runtime_error("Expected 14 CSV columns, got " + std::to_string(fields.size()));
  }

  ReferenceRow row;
  row.x_axis = std::stod(fields[0]);
  row.original_from_screenshot_cm = std::stod(fields[1]);
  row.baseline_slow = std::stod(fields[2]);
  row.wave_raw = std::stod(fields[3]);
  row.wave_clean = std::stod(fields[4]);
  row.wave_freq_hz = std::stod(fields[5]);
  row.wave_period_s = std::stod(fields[6]);
  row.baseline_cutoff_hz = std::stod(fields[7]);
  row.baseline_tau_s = std::stod(fields[8]);
  row.cleanup_cutoff_hz = std::stod(fields[9]);
  row.slope_rms = std::stod(fields[10]);
  row.slope_threshold = std::stod(fields[11]);
  row.freq_valid = parse_bool(fields[12]);
  row.schmitt_state = std::stoi(fields[13]);
  return row;
}

std::vector<ReferenceRow> load_reference_rows(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("Unable to open input CSV: " + path);
  }

  std::string line;
  if (!std::getline(ifs, line)) {
    throw std::runtime_error("Input CSV is empty: " + path);
  }

  std::vector<ReferenceRow> rows;
  while (std::getline(ifs, line)) {
    if (trim(line).empty()) {
      continue;
    }
    rows.push_back(parse_row(split_csv_line(line)));
  }

  if (rows.empty()) {
    throw std::runtime_error("No data rows found in input CSV: " + path);
  }

  return rows;
}

void update_error(ErrorStats& stats, double error_value) {
  stats.max_abs_error = std::max(stats.max_abs_error, std::abs(error_value));
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const std::string input_path = (argc > 1) ? argv[1] : "../data-sim/detrend.csv";
    const std::string output_path = (argc > 2) ? argv[2] : "adaptive_wave_detrender_test_output.csv";

    const std::vector<ReferenceRow> reference_rows = load_reference_rows(input_path);

    AdaptiveWaveDetrender detrender;

    std::ofstream ofs(output_path);
    if (!ofs.is_open()) {
      throw std::runtime_error("Unable to open output CSV: " + output_path);
    }

    ofs << "sample_idx,time_s,x_axis,input_cm,";
    ofs << "ref_baseline_slow,actual_baseline_slow,baseline_abs_error,";
    ofs << "ref_wave_raw,actual_wave_raw,wave_raw_abs_error,";
    ofs << "ref_wave_clean,actual_wave_clean,wave_clean_abs_error,";
    ofs << "ref_wave_freq_hz,actual_wave_freq_hz,wave_freq_abs_error,";
    ofs << "ref_wave_period_s,actual_wave_period_s,wave_period_abs_error,";
    ofs << "ref_baseline_cutoff_hz,actual_baseline_cutoff_hz,baseline_cutoff_abs_error,";
    ofs << "ref_baseline_tau_s,actual_baseline_tau_s,baseline_tau_abs_error,";
    ofs << "ref_cleanup_cutoff_hz,actual_cleanup_cutoff_hz,cleanup_cutoff_abs_error,";
    ofs << "ref_slope_rms,actual_slope_rms,slope_rms_abs_error,";
    ofs << "ref_slope_threshold,actual_slope_threshold,slope_threshold_abs_error,";
    ofs << "ref_freq_valid,actual_freq_valid,ref_schmitt_state,actual_schmitt_state\n";
    ofs << std::fixed << std::setprecision(9);

    ErrorStats baseline_stats{"baseline_slow", 1.0e-3, 0.0};
    ErrorStats wave_raw_stats{"wave_raw", 1.0e-3, 0.0};
    ErrorStats wave_clean_stats{"wave_clean", 1.0e-3, 0.0};
    ErrorStats wave_freq_stats{"wave_freq_hz", 1.0e-5, 0.0};
    ErrorStats wave_period_stats{"wave_period_s", 5.0e-4, 0.0};
    ErrorStats baseline_cutoff_stats{"baseline_cutoff_hz", 1.0e-6, 0.0};
    ErrorStats baseline_tau_stats{"baseline_tau_s", 5.0e-4, 0.0};
    ErrorStats cleanup_cutoff_stats{"cleanup_cutoff_hz", 1.0e-6, 0.0};
    ErrorStats slope_rms_stats{"slope_rms", 1.0e-4, 0.0};
    ErrorStats slope_threshold_stats{"slope_threshold", 1.0e-4, 0.0};

    bool bool_mismatch = false;
    bool schmitt_mismatch = false;
    double current_time_s = 0.0;

    for (std::size_t i = 0; i < reference_rows.size(); ++i) {
      const ReferenceRow& row = reference_rows[i];

      AdaptiveWaveDetrender::Output output;
      if (i == 0U) {
        detrender.reset(static_cast<float>(row.original_from_screenshot_cm));
        output = detrender.lastOutput();
      } else {
        const double dt_s = row.x_axis - reference_rows[i - 1U].x_axis;
        if (!(dt_s > 0.0)) {
          throw std::runtime_error("Non-positive dt derived from x_axis at row " + std::to_string(i));
        }
        current_time_s += dt_s;
        output = detrender.update(static_cast<float>(row.original_from_screenshot_cm), static_cast<float>(dt_s));
      }

      const double baseline_error = std::abs(static_cast<double>(output.baseline_slow) - row.baseline_slow);
      const double wave_raw_error = std::abs(static_cast<double>(output.wave_raw) - row.wave_raw);
      const double wave_clean_error = std::abs(static_cast<double>(output.wave_clean) - row.wave_clean);
      const double wave_freq_error = std::abs(static_cast<double>(output.wave_freq_hz) - row.wave_freq_hz);
      const double wave_period_error = std::abs(static_cast<double>(output.wave_period_s) - row.wave_period_s);
      const double baseline_cutoff_error = std::abs(static_cast<double>(output.baseline_cutoff_hz) - row.baseline_cutoff_hz);
      const double baseline_tau_error = std::abs(static_cast<double>(output.baseline_tau_s) - row.baseline_tau_s);
      const double cleanup_cutoff_error = std::abs(static_cast<double>(output.cleanup_cutoff_hz) - row.cleanup_cutoff_hz);
      const double slope_rms_error = std::abs(static_cast<double>(output.slope_rms) - row.slope_rms);
      const double slope_threshold_error = std::abs(static_cast<double>(output.slope_threshold) - row.slope_threshold);

      update_error(baseline_stats, baseline_error);
      update_error(wave_raw_stats, wave_raw_error);
      update_error(wave_clean_stats, wave_clean_error);
      update_error(wave_freq_stats, wave_freq_error);
      update_error(wave_period_stats, wave_period_error);
      update_error(baseline_cutoff_stats, baseline_cutoff_error);
      update_error(baseline_tau_stats, baseline_tau_error);
      update_error(cleanup_cutoff_stats, cleanup_cutoff_error);
      update_error(slope_rms_stats, slope_rms_error);
      update_error(slope_threshold_stats, slope_threshold_error);

      if (output.freq_valid != row.freq_valid) {
        bool_mismatch = true;
      }
      if (output.schmitt_state != row.schmitt_state) {
        schmitt_mismatch = true;
      }

      ofs << i << ','
          << current_time_s << ','
          << row.x_axis << ','
          << row.original_from_screenshot_cm << ','
          << row.baseline_slow << ',' << output.baseline_slow << ',' << baseline_error << ','
          << row.wave_raw << ',' << output.wave_raw << ',' << wave_raw_error << ','
          << row.wave_clean << ',' << output.wave_clean << ',' << wave_clean_error << ','
          << row.wave_freq_hz << ',' << output.wave_freq_hz << ',' << wave_freq_error << ','
          << row.wave_period_s << ',' << output.wave_period_s << ',' << wave_period_error << ','
          << row.baseline_cutoff_hz << ',' << output.baseline_cutoff_hz << ',' << baseline_cutoff_error << ','
          << row.baseline_tau_s << ',' << output.baseline_tau_s << ',' << baseline_tau_error << ','
          << row.cleanup_cutoff_hz << ',' << output.cleanup_cutoff_hz << ',' << cleanup_cutoff_error << ','
          << row.slope_rms << ',' << output.slope_rms << ',' << slope_rms_error << ','
          << row.slope_threshold << ',' << output.slope_threshold << ',' << slope_threshold_error << ','
          << (row.freq_valid ? 1 : 0) << ',' << (output.freq_valid ? 1 : 0) << ','
          << row.schmitt_state << ',' << static_cast<int>(output.schmitt_state) << '\n';
    }

    const std::vector<ErrorStats> stats = {
      baseline_stats,
      wave_raw_stats,
      wave_clean_stats,
      wave_freq_stats,
      wave_period_stats,
      baseline_cutoff_stats,
      baseline_tau_stats,
      cleanup_cutoff_stats,
      slope_rms_stats,
      slope_threshold_stats,
    };

    bool ok = true;
    for (const ErrorStats& stat : stats) {
      std::cout << stat.name << " max abs error: " << stat.max_abs_error
                << " (tolerance " << stat.tolerance << ")\n";
      if (stat.max_abs_error > stat.tolerance) {
        ok = false;
      }
    }

    if (bool_mismatch) {
      std::cerr << "freq_valid mismatch detected against detrend.csv reference\n";
      ok = false;
    }
    if (schmitt_mismatch) {
      std::cerr << "schmitt_state mismatch detected against detrend.csv reference\n";
      ok = false;
    }

    std::cout << "Wrote regression output to " << output_path << '\n';

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& ex) {
    std::cerr << "adaptive_wave_detrender_test failed: " << ex.what() << '\n';
    return EXIT_FAILURE;
  }
}
