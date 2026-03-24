#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define EIGEN_NON_ARDUINO

#include "CalibrateIMU.h"

namespace {
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
constexpr double kPi = 3.14159265358979323846;
constexpr double kG = 9.80665;
constexpr double kT0 = 25.0;
constexpr int kSamples = 320;

struct Gates {
  double accel_rms_norm_mps2_max = 0.12;
  double accel_bias_fit_mps2_max = 0.07;
  double mag_rms_uT_max = 1.1;
  double mag_norm_rms_uT_max = 1.2;
  double gyro_bias_rms_rads_max = 0.012;
};

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); }
void expect(bool cond, const std::string& msg) { if (!cond) fail(msg); }

Vec3 spherical_direction(double u, double v) {
  const double z = 2.0 * u - 1.0;
  const double phi = 2.0 * kPi * v;
  const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
  return Vec3(r * std::cos(phi), r * std::sin(phi), z);
}

double rms(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  double sse = 0.0;
  for (double v : values) sse += v * v;
  return std::sqrt(sse / static_cast<double>(values.size()));
}
}  // namespace

int main(int argc, char* argv[]) {
  try {
    const std::string csv_path = (argc > 1) ? argv[1] : "calibrate_imu_test_output.csv";
    const std::string summary_path = (argc > 2) ? argv[2] : "calibrate_imu_test_summary.csv";
    std::ofstream csv(csv_path);
    std::ofstream summary(summary_path);
    if (!csv.is_open()) fail("unable to open csv output");
    if (!summary.is_open()) fail("unable to open summary output");

    csv << "sensor,sample,tempC,raw_x,raw_y,raw_z,cal_x,cal_y,cal_z,cal_norm,target_norm,error_norm\n";
    csv << std::fixed << std::setprecision(9);
    summary << "sensor,metric,value,gate,pass\n";
    summary << std::fixed << std::setprecision(9);

    std::mt19937 rng(42u);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_real_distribution<double> temp_dist(5.0, 45.0);
    std::normal_distribution<double> noise_acc(0.0, 0.015);
    std::normal_distribution<double> noise_mag(0.0, 0.08);
    std::normal_distribution<double> noise_gyro(0.0, 0.0012);

    const Gates gates{};
    bool all_ok = true;

    imu_cal::AccelCalibrator<double, 400, 8> accel;
    accel.g = kG;
    accel.T0 = kT0;
    accel.accel_mag_tol = 1.20;

    const Mat3 accel_S_true = (Mat3() << 1.08, 0.0, 0.0, 0.0, 0.94, 0.0, 0.0, 0.0, 1.03).finished();
    const Mat3 accel_S_true_inv = accel_S_true.inverse();
    const Vec3 accel_b0(0.09, -0.07, 0.05);
    const Vec3 accel_k(0.0012, -0.0008, 0.0005);
    std::vector<Vec3> accel_raw;
    std::vector<double> accel_temp;

    for (int i = 0; i < kSamples; ++i) {
      const double tC = temp_dist(rng);
      const Vec3 a_true = kG * spherical_direction(uni01(rng), uni01(rng));
      Vec3 raw = accel_S_true_inv * a_true + (accel_b0 + accel_k * (tC - kT0));
      raw += Vec3(noise_acc(rng), noise_acc(rng), noise_acc(rng));
      expect(accel.addSample(raw, Vec3::Zero(), tC), "accel sample rejected unexpectedly");
      accel_raw.push_back(raw);
      accel_temp.push_back(tC);
    }

    imu_cal::AccelCalibration<double> accel_out;
    imu_cal::FitFail accel_reason = imu_cal::FitFail::OK;
    expect(accel.fit(accel_out, 4, 0.14, &accel_reason), std::string("accel fit failed: ") + imu_cal::fitFailStr(accel_reason));

    std::vector<double> accel_norm_err;
    std::vector<double> accel_bias_err;
    for (int i = 0; i < kSamples; ++i) {
      const double tC = accel_temp[i];
      const Vec3 cal = accel_out.apply(accel_raw[i], tC);
      const double nerr = cal.norm() - kG;
      accel_norm_err.push_back(nerr);
      accel_bias_err.push_back((accel_out.biasT.bias(tC) - (accel_b0 + accel_k * (tC - kT0))).norm());
      csv << "accel," << i << ',' << tC << ',' << accel_raw[i].x() << ',' << accel_raw[i].y() << ',' << accel_raw[i].z() << ','
          << cal.x() << ',' << cal.y() << ',' << cal.z() << ',' << cal.norm() << ',' << kG << ',' << nerr << '\n';
    }

    const double accel_norm_rms = rms(accel_norm_err);
    const double accel_bias_rms = rms(accel_bias_err);
    const bool accel_gate_1 = accel_norm_rms <= gates.accel_rms_norm_mps2_max;
    const bool accel_gate_2 = accel_bias_rms <= gates.accel_bias_fit_mps2_max;
    all_ok = all_ok && accel_gate_1 && accel_gate_2;
    summary << "accel,norm_rms_mps2," << accel_norm_rms << ',' << gates.accel_rms_norm_mps2_max << ',' << (accel_gate_1 ? 1 : 0) << '\n';
    summary << "accel,bias_fit_rms_mps2," << accel_bias_rms << ',' << gates.accel_bias_fit_mps2_max << ',' << (accel_gate_2 ? 1 : 0) << '\n';

    imu_cal::MagCalibrator<double, 400> mag;
    const double field_uT = 47.0;
    const Mat3 mag_A_true = (Mat3() << 1.18, 0.02, -0.01, 0.01, 0.86, 0.03, -0.02, 0.01, 1.05).finished();
    const Mat3 mag_A_true_inv = mag_A_true.inverse();
    const Vec3 mag_b_true(8.0, -5.0, 3.0);
    std::vector<Vec3> mag_raw;

    for (int i = 0; i < kSamples; ++i) {
      const Vec3 m_true = field_uT * spherical_direction(uni01(rng), uni01(rng));
      Vec3 raw = mag_A_true_inv * m_true + mag_b_true + Vec3(noise_mag(rng), noise_mag(rng), noise_mag(rng));
      expect(mag.addSample(raw), "mag sample rejected unexpectedly");
      mag_raw.push_back(raw);
    }

    imu_cal::MagCalibration<double> mag_out;
    imu_cal::FitFail mag_reason = imu_cal::FitFail::OK;
    expect(mag.fit(mag_out, 4, 0.12, 1e-6, &mag_reason), std::string("mag fit failed: ") + imu_cal::fitFailStr(mag_reason));

    std::vector<double> mag_norm_err;
    for (int i = 0; i < kSamples; ++i) {
      const Vec3 cal = mag_out.apply(mag_raw[i]);
      const double nerr = cal.norm() - field_uT;
      mag_norm_err.push_back(nerr);
      csv << "mag," << i << ",0," << mag_raw[i].x() << ',' << mag_raw[i].y() << ',' << mag_raw[i].z() << ','
          << cal.x() << ',' << cal.y() << ',' << cal.z() << ',' << cal.norm() << ',' << field_uT << ',' << nerr << '\n';
    }

    const double mag_norm_rms = rms(mag_norm_err);
    const bool mag_gate_1 = mag_out.rms <= gates.mag_rms_uT_max;
    const bool mag_gate_2 = mag_norm_rms <= gates.mag_norm_rms_uT_max;
    all_ok = all_ok && mag_gate_1 && mag_gate_2;
    summary << "mag,fit_rms_uT," << mag_out.rms << ',' << gates.mag_rms_uT_max << ',' << (mag_gate_1 ? 1 : 0) << '\n';
    summary << "mag,norm_rms_uT," << mag_norm_rms << ',' << gates.mag_norm_rms_uT_max << ',' << (mag_gate_2 ? 1 : 0) << '\n';

    imu_cal::GyroCalibrator<double, 400, 8> gyro;
    gyro.T0 = kT0;
    const Vec3 gyro_b0(0.015, -0.009, 0.006);
    const Vec3 gyro_k(0.00035, -0.00021, 0.00017);
    std::vector<Vec3> gyro_raw;
    std::vector<double> gyro_temp;

    for (int i = 0; i < kSamples; ++i) {
      const double tC = temp_dist(rng);
      Vec3 w = gyro_b0 + gyro_k * (tC - kT0) + Vec3(noise_gyro(rng), noise_gyro(rng), noise_gyro(rng));
      Vec3 a(0.0, 0.0, kG);
      a += Vec3(noise_acc(rng) * 0.2, noise_acc(rng) * 0.2, noise_acc(rng) * 0.2);
      expect(gyro.addSample(w, a, tC), "gyro sample rejected unexpectedly");
      gyro_raw.push_back(w);
      gyro_temp.push_back(tC);
    }

    imu_cal::GyroCalibration<double> gyro_out;
    imu_cal::FitFail gyro_reason = imu_cal::FitFail::OK;
    expect(gyro.fit(gyro_out, &gyro_reason), std::string("gyro fit failed: ") + imu_cal::fitFailStr(gyro_reason));

    std::vector<double> gyro_bias_err;
    for (int i = 0; i < kSamples; ++i) {
      const Vec3 cal = gyro_out.apply(gyro_raw[i], gyro_temp[i]);
      gyro_bias_err.push_back((gyro_out.biasT.bias(gyro_temp[i]) - (gyro_b0 + gyro_k * (gyro_temp[i] - kT0))).norm());
      csv << "gyro," << i << ',' << gyro_temp[i] << ',' << gyro_raw[i].x() << ',' << gyro_raw[i].y() << ',' << gyro_raw[i].z() << ','
          << cal.x() << ',' << cal.y() << ',' << cal.z() << ',' << cal.norm() << ",0," << cal.norm() << '\n';
    }

    const double gyro_bias_rms = rms(gyro_bias_err);
    const bool gyro_gate = gyro_bias_rms <= gates.gyro_bias_rms_rads_max;
    all_ok = all_ok && gyro_gate;
    summary << "gyro,bias_fit_rms_rads," << gyro_bias_rms << ',' << gates.gyro_bias_rms_rads_max << ',' << (gyro_gate ? 1 : 0) << '\n';

    std::cout << "calibrate_imu_test outputs: " << csv_path << ", " << summary_path << "\n";
    std::cout << "accel norm RMS=" << accel_norm_rms << " (gate " << gates.accel_rms_norm_mps2_max << ")\n";
    std::cout << "accel bias-fit RMS=" << accel_bias_rms << " (gate " << gates.accel_bias_fit_mps2_max << ")\n";
    std::cout << "mag fit RMS=" << mag_out.rms << " (gate " << gates.mag_rms_uT_max << ")\n";
    std::cout << "mag norm RMS=" << mag_norm_rms << " (gate " << gates.mag_norm_rms_uT_max << ")\n";
    std::cout << "gyro bias-fit RMS=" << gyro_bias_rms << " (gate " << gates.gyro_bias_rms_rads_max << ")\n";

    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& ex) {
    std::cerr << "calibrate_imu_test: FAIL: " << ex.what() << '\n';
    return EXIT_FAILURE;
  }
}
