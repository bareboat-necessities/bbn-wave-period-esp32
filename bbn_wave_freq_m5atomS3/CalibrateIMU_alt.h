#pragma once

// Works with ArduinoEigenDense (Eigen subset). Use float to keep it light.
#include <stdint.h>
#include <math.h>

#include <Eigen/Dense>

namespace imu_cal {

// -----------------------------
// Small helpers
// -----------------------------
template <typename T>
static inline T clamp(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }

template <typename T>
static inline bool isfinite3(const Eigen::Matrix<T,3,1>& v) {
  return isfinite(v.x()) && isfinite(v.y()) && isfinite(v.z());
}

// -----------------------------
// Fixed-capacity sample buffer
// -----------------------------
template <typename T, int N>
struct Vec3SampleBuffer {
  using Vec3 = Eigen::Matrix<T,3,1>;

  Vec3 samples[N];
  T    temps[N];     // optional; can be 0 if unused
  int  size = 0;

  void clear() { size = 0; }

  bool push(const Vec3& v, T temp = T(0)) {
    if (!isfinite3(v)) return false;
    if (size >= N) return false;
    samples[size] = v;
    temps[size]   = temp;
    ++size;
    return true;
  }
};

// -----------------------------
// Ellipsoid fitter (x^T Q x + q^T x + c = 0)
// Design row: [x^2, y^2, z^2, 2xy, 2xz, 2yz, 2x, 2y, 2z, 1]
// We solve D p ≈ 1 (or -1) via least squares; scale is arbitrary.
// -----------------------------
template <typename T>
struct EllipsoidFitResult {
  bool ok = false;

  Eigen::Matrix<T,3,3> A = Eigen::Matrix<T,3,3>::Identity(); // "whitening" transform
  Eigen::Matrix<T,3,1> b = Eigen::Matrix<T,3,1>::Zero();     // center (hard-iron / accel bias proxy)
  T target_radius = T(1);                                     // scale output to this radius if desired

  // Quality metrics
  T rms_radius_error = T(0);
  T mean_radius = T(0);
  int n = 0;
};

template <typename T>
static EllipsoidFitResult<T> fit_ellipsoid_to_sphere(
    const Eigen::Matrix<T,3,1>* x, int n,
    T target_radius = T(1),
    T ridge = T(1e-6))   // small regularization for stability
{
  EllipsoidFitResult<T> out;
  out.target_radius = target_radius;
  out.n = n;

  if (n < 10) return out;

  using Mat10 = Eigen::Matrix<T,10,10>;
  using Vec10 = Eigen::Matrix<T,10,1>;

  Mat10 H = Mat10::Zero();
  Vec10 g = Vec10::Zero();

  for (int i = 0; i < n; ++i) {
    const T X = x[i].x();
    const T Y = x[i].y();
    const T Z = x[i].z();

    Vec10 d;
    d << X*X, Y*Y, Z*Z,
         T(2)*X*Y, T(2)*X*Z, T(2)*Y*Z,
         T(2)*X, T(2)*Y, T(2)*Z,
         T(1);

    // Normal equations: minimize ||D p - 1||^2
    H.noalias() += d * d.transpose();
    g.noalias() += d; // D^T * 1
  }

  // Ridge regularization
  H.diagonal().array() += ridge;

  // Solve H p = g
  Vec10 p = H.ldlt().solve(g);
  if (!(p.array().isFinite().all())) return out;

  // Build Q, q, c
  Eigen::Matrix<T,3,3> Q;
  Q << p(0), p(3), p(4),
       p(3), p(1), p(5),
       p(4), p(5), p(2);

  Eigen::Matrix<T,3,1> q;
  q << p(6), p(7), p(8);

  const T c = p(9);

  // Center b = -0.5 Q^{-1} q
  Eigen::FullPivLU<Eigen::Matrix<T,3,3>> lu(Q);
  if (!lu.isInvertible()) return out;

  Eigen::Matrix<T,3,1> b = T(-0.5) * lu.solve(q);

  // Compute translated constant: k = c + b^T Q b + q^T b
  const T k = c + b.dot(Q*b) + q.dot(b);

  // For ellipsoid: Q SPD-ish and k should be negative (so that (x-b)^T(Q/-k)(x-b) = 1)
  if (!isfinite(k) || k >= T(0)) return out;

  Eigen::Matrix<T,3,3> M = Q / (-k); // should be SPD

  // Ensure symmetry (numeric hygiene)
  M = T(0.5) * (M + M.transpose());

  Eigen::LLT<Eigen::Matrix<T,3,3>> llt(M);
  if (llt.info() != Eigen::Success) return out;

  // Whitening transform. For any raw sample:
  // y = A (x - b)  => ||y|| ≈ 1
  Eigen::Matrix<T,3,3> A = llt.matrixU(); // upper-triangular

  // Compute radius stats after scaling to target_radius
  T sum_r = 0;
  T sum_e2 = 0;
  for (int i = 0; i < n; ++i) {
    Eigen::Matrix<T,3,1> y = A * (x[i] - b);
    T r = y.norm();
    if (!isfinite(r)) continue;
    sum_r += r;
  }
  T mean_r = sum_r / T(n);

  for (int i = 0; i < n; ++i) {
    Eigen::Matrix<T,3,1> y = A * (x[i] - b);
    T r = y.norm();
    if (!isfinite(r)) continue;
    T e = (r - T(1));
    sum_e2 += e*e;
  }
  T rms = sqrt(sum_e2 / T(n));

  // Scale A to target radius (so output has magnitude ~target_radius)
  A *= target_radius;

  out.ok = true;
  out.A = A;
  out.b = b;
  out.mean_radius = mean_r * target_radius;
  out.rms_radius_error = rms * target_radius;
  return out;
}

// -----------------------------
// Temperature regression: fit b(T) = b0 + k*(T - T0)
// Simple least squares per-axis.
// -----------------------------
template <typename T>
struct TempBiasModel3 {
  bool ok = false;
  T T0 = T(25); // reference temp (deg C)

  Eigen::Matrix<T,3,1> b0 = Eigen::Matrix<T,3,1>::Zero();
  Eigen::Matrix<T,3,1> k  = Eigen::Matrix<T,3,1>::Zero(); // per-degree

  Eigen::Matrix<T,3,1> bias(T T) const {
    return b0 + k * (T - T0);
  }
};

template <typename T, int N>
static TempBiasModel3<T> fit_temp_bias_from_centers(
    const Eigen::Matrix<T,3,1> (&centers)[N],
    const T (&temps)[N],
    int n,
    T T0)
{
  TempBiasModel3<T> out;
  out.T0 = T0;
  if (n < 2) return out;

  // Fit each axis: center_i = b0 + k*(T - T0)
  // => y = b0 + k*x where x=(T-T0)
  T Sx = 0, Sxx = 0;
  Eigen::Matrix<T,3,1> Sy = Eigen::Matrix<T,3,1>::Zero();
  Eigen::Matrix<T,3,1> Sxy = Eigen::Matrix<T,3,1>::Zero();

  for (int i = 0; i < n; ++i) {
    const T x = temps[i] - T0;
    Sx  += x;
    Sxx += x*x;
    Sy  += centers[i];
    Sxy += centers[i] * x;
  }

  const T det = T(n) * Sxx - Sx * Sx;
  if (fabs(det) < T(1e-9)) return out;

  // b0 = (Sy*Sxx - Sxy*Sx)/det
  // k  = (n*Sxy - Sy*Sx)/det
  out.b0 = (Sy * Sxx - Sxy * Sx) / det;
  out.k  = (Sxy * T(n) - Sy * Sx) / det;
  out.ok = true;
  return out;
}

// -----------------------------
// Calibration models
// -----------------------------
template <typename T>
struct AccelCalibration {
  bool ok = false;
  T g = T(9.80665);

  // a_cal = S * (a_raw - bias(T))
  Eigen::Matrix<T,3,3> S = Eigen::Matrix<T,3,3>::Identity();

  TempBiasModel3<T> biasT; // b0 + k*(T-T0)

  // Diagnostics
  T rms_mag_error = T(0);

  Eigen::Matrix<T,3,1> apply(const Eigen::Matrix<T,3,1>& a_raw, T tempC) const {
    const Eigen::Matrix<T,3,1> b = biasT.bias(tempC);
    return S * (a_raw - b);
  }
};

template <typename T>
struct MagCalibration {
  bool ok = false;

  // m_cal = A * (m_raw - b)
  Eigen::Matrix<T,3,3> A = Eigen::Matrix<T,3,3>::Identity(); // soft-iron correction (whitening * field strength)
  Eigen::Matrix<T,3,1> b = Eigen::Matrix<T,3,1>::Zero();     // hard-iron

  // estimated field strength magnitude after calibration (optional)
  T field_uT = T(0);
  T rms_mag_error = T(0);

  Eigen::Matrix<T,3,1> apply(const Eigen::Matrix<T,3,1>& m_raw) const {
    return A * (m_raw - b);
  }
};

template <typename T>
struct GyroCalibration {
  bool ok = false;

  // w_cal = S * (w_raw - bias(T))
  Eigen::Matrix<T,3,3> S = Eigen::Matrix<T,3,3>::Identity(); // keep diag unless you have a rate table
  TempBiasModel3<T> biasT;

  Eigen::Matrix<T,3,1> apply(const Eigen::Matrix<T,3,1>& w_raw, T tempC) const {
    return S * (w_raw - biasT.bias(tempC));
  }
};

// -----------------------------
// Builders / fitters
// -----------------------------
template <typename T, int N>
struct AccelCalibrator {
  using Vec3 = Eigen::Matrix<T,3,1>;

  // Collect samples while rotating the IMU slowly in many orientations.
  // Samples should be "quasi-static" so |a_true|≈g.
  Vec3SampleBuffer<T, N> buf;

  // Optional: group ellipsoid centers vs temperature for bias(T)
  static constexpr int MAX_TEMP_BINS = 8;
  Vec3 bin_center[MAX_TEMP_BINS];
  T    bin_temp[MAX_TEMP_BINS];
  int  bin_count = 0;

  T g = T(9.80665);
  T T0 = T(25);

  void clear() { buf.clear(); bin_count = 0; }

  bool addSample(const Vec3& a_raw, T tempC) {
    return buf.push(a_raw, tempC);
  }

  // Fit scale/misalignment + a "center". Then store center per temp-bin.
  // Later call fitTempModel() once you have several bins across temperature.
  bool fitPerBatchCenter(T tempC_center, AccelCalibration<T>& out, T ridge = T(1e-6)) {
    auto r = fit_ellipsoid_to_sphere<T>(buf.samples, buf.size, g, ridge);
    if (!r.ok) return false;

    // Here r.b is "center" in raw space. r.A is a whitening+scale.
    // We want a_cal = S * (a_raw - b(T)).
    out.S = r.A;
    out.g = g;
    out.ok = true;
    out.rms_mag_error = r.rms_radius_error;

    // Store this batch's center for temp regression
    if (bin_count < MAX_TEMP_BINS) {
      bin_center[bin_count] = r.b;
      bin_temp[bin_count] = tempC_center;
      ++bin_count;
    }

    // If no temp model yet, at least set b0 to this center.
    if (!out.biasT.ok) {
      out.biasT.T0 = T0;
      out.biasT.b0 = r.b;
      out.biasT.k.setZero();
      out.biasT.ok = true;
    }

    return true;
  }

  bool fitTempModel(AccelCalibration<T>& out) {
    if (bin_count < 2) return false;
    TempBiasModel3<T> m = fit_temp_bias_from_centers<T, MAX_TEMP_BINS>(bin_center, bin_temp, bin_count, T0);
    if (!m.ok) return false;
    out.biasT = m;
    out.ok = true;
    return true;
  }
};

template <typename T, int N>
struct MagCalibrator {
  using Vec3 = Eigen::Matrix<T,3,1>;
  Vec3SampleBuffer<T, N> buf;

  void clear() { buf.clear(); }

  bool addSample(const Vec3& m_raw) { return buf.push(m_raw, T(0)); }

  bool fit(MagCalibration<T>& out, T ridge = T(1e-6)) {
    // Fit to unit sphere first, then scale to estimated field magnitude
    auto r = fit_ellipsoid_to_sphere<T>(buf.samples, buf.size, T(1), ridge);
    if (!r.ok) return false;

    // Estimate field magnitude from calibrated samples
    T sum = 0;
    for (int i = 0; i < buf.size; ++i) {
      Vec3 y = r.A * (buf.samples[i] - r.b);
      sum += y.norm();
    }
    T field = (buf.size > 0) ? (sum / T(buf.size)) : T(1);

    // Scale so output magnitude ~ field (keeps µT-ish scale if input was µT)
    // If input is raw counts, this still yields consistent normalized magnitude.
    out.A = r.A * field;
    out.b = r.b;
    out.field_uT = field;
    out.rms_mag_error = r.rms_radius_error * field;
    out.ok = true;
    return true;
  }
};

template <typename T, int N>
struct GyroCalibrator {
  using Vec3 = Eigen::Matrix<T,3,1>;
  Vec3SampleBuffer<T, N> buf_stationary; // collect while IMU is stationary

  static constexpr int MAX_TEMP_BINS = 8;
  Vec3 bin_bias[MAX_TEMP_BINS];
  T    bin_temp[MAX_TEMP_BINS];
  int  bin_count = 0;

  T T0 = T(25);

  void clear() { buf_stationary.clear(); bin_count = 0; }

  bool addStationarySample(const Vec3& w_raw, T tempC) {
    return buf_stationary.push(w_raw, tempC);
  }

  // Compute mean gyro bias for this batch (stationary), store vs temp.
  bool finishStationaryBatch(T tempC_center, GyroCalibration<T>& out) {
    if (buf_stationary.size < 20) return false;

    Vec3 mean = Vec3::Zero();
    for (int i = 0; i < buf_stationary.size; ++i) mean += buf_stationary.samples[i];
    mean /= T(buf_stationary.size);

    if (bin_count < MAX_TEMP_BINS) {
      bin_bias[bin_count] = mean;
      bin_temp[bin_count] = tempC_center;
      ++bin_count;
    }

    // set default model immediately
    out.biasT.T0 = T0;
    out.biasT.b0 = mean;
    out.biasT.k.setZero();
    out.biasT.ok = true;

    out.S = Eigen::Matrix<T,3,3>::Identity(); // scale requires known-rate calibration; leave as identity
    out.ok = true;

    return true;
  }

  bool fitTempModel(GyroCalibration<T>& out) {
    if (bin_count < 2) return false;
    TempBiasModel3<T> m = fit_temp_bias_from_centers<T, MAX_TEMP_BINS>(bin_bias, bin_temp, bin_count, T0);
    if (!m.ok) return false;
    out.biasT = m;
    out.ok = true;
    return true;
  }
};

} // namespace imu_cal
