// imu_ellipsoid_calibration_rates.h
// Header-only, heap-free ellipsoid calibration with configurable sample rates
// for ESP32-S3 + Eigen.
//
// Calibrates:
//   - Accelerometer: bias + 3x3 soft matrix by fitting ||M(a-b)|| = g (auto-detect units)
//   - Magnetometer: hard-iron + soft-iron by fitting ||M(m-b)|| = r (r estimated)
//   - Gyro: stillness-based bias (recommended for embedded)
//
// Key feature requested: configurable rates
//   - accel+gyro effective calibration sampling rate (acc_gyro_cal_hz)
//   - magnetometer effective calibration sampling rate (mag_cal_hz)
// You can call update() at your high-rate loop; internally it down-samples.
//
// Apply:
//   a_cal = acc.M * (a_raw - acc.bias)
//   m_cal = mag.M * (m_raw - mag.bias)
//   w_cal = w_raw - gyro_bias
//
// Notes:
//   - No heap allocations (fixed MaxN buffers, fixed-size matrices).
//   - Includes coverage gating so it won’t fit from a “flat” dataset.
//   - Auto accel unit detect (g, m/s^2, mg). If unknown: targets median norm.

#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

#ifdef EIGEN_NON_ARDUINO
  #include <Eigen/Dense>
#else
  #include <ArduinoEigenDense.h>
#endif

namespace imu_cal {

template <typename T>
static inline T clamp(T x, T lo, T hi) { return std::min(hi, std::max(lo, x)); }

template <typename T>
static inline T huber_weight(T r, T k) {
  const T a = std::abs(r);
  if (a <= k) return T(1);
  return k / a;
}

template <typename T, int MaxN>
struct SampleBuffer3 {
  using Vec3 = Eigen::Matrix<T,3,1>;
  Vec3 data[MaxN];
  int  n = 0;

  void clear() { n = 0; }
  int  size() const { return n; }

  bool push(const Vec3& v) {
    if (n >= MaxN) return false;
    data[n++] = v;
    return true;
  }
  const Vec3& operator[](int i) const { return data[i]; }
};

template <typename T, int MaxN>
struct NormStats {
  T norms[MaxN];
  int n = 0;

  void clear() { n = 0; }
  int  size() const { return n; }
  bool add_norm(T x) {
    if (n >= MaxN) return false;
    norms[n++] = x;
    return true;
  }

  template <typename Vec3>
  bool add_vec(const Vec3& v) { return add_norm(T(v.norm())); }

  static T quickselect(T* a, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
      const T pivot = a[(lo + hi) >> 1];
      int i = lo, j = hi;
      while (i <= j) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) { std::swap(a[i], a[j]); i++; j--; }
      }
      if (k <= j) hi = j;
      else if (k >= i) lo = i;
      else return a[k];
    }
    return a[lo];
  }

  T median() const {
    if (n <= 0) return T(0);
    T tmp[MaxN];
    for (int i=0;i<n;i++) tmp[i] = norms[i];
    const int k = n >> 1;
    T m = quickselect(tmp, n, k);
    if ((n & 1) == 0) {
      T m2 = quickselect(tmp, n, k - 1);
      return T(0.5) * (m + m2);
    }
    return m;
  }

  T mad() const {
    if (n <= 0) return T(0);
    const T m = median();
    T tmp[MaxN];
    for (int i=0;i<n;i++) tmp[i] = std::abs(norms[i] - m);
    const int k = n >> 1;
    T d = quickselect(tmp, n, k);
    if ((n & 1) == 0) {
      T d2 = quickselect(tmp, n, k - 1);
      return T(0.5) * (d + d2);
    }
    return d;
  }
};

// Cheap spherical coverage bins: quantize direction (theta/phi)
template <typename T, int ElevBins, int AzimBins>
struct CoverageBins {
  bool bins[ElevBins * AzimBins];

  void clear() { for (int i=0;i<ElevBins*AzimBins;i++) bins[i] = false; }

  static inline T rad2deg(T r) { return r * T(57.29577951308232); }

  void add_dir(const Eigen::Matrix<T,3,1>& v) {
    const T n = v.norm();
    if (n < T(1e-6)) return;
    const Eigen::Matrix<T,3,1> u = v / n;

    const T theta = rad2deg(std::acos(clamp(u.z(), T(-1), T(1)))); // 0..180
    const T phi   = rad2deg(std::atan2(u.y(), u.x()));             // -180..180

    int ti = int((theta / T(180)) * T(ElevBins));
    ti = std::min(std::max(ti, 0), ElevBins-1);

    T phi01 = (phi + T(180)) / T(360); // 0..1
    int pi = int(phi01 * T(AzimBins));
    pi = std::min(std::max(pi, 0), AzimBins-1);

    bins[ti*AzimBins + pi] = true;
  }

  int count() const {
    int c = 0;
    for (int i=0;i<ElevBins*AzimBins;i++) if (bins[i]) c++;
    return c;
  }
};

enum class AccelUnits : uint8_t { G, MPS2, MG, UNKNOWN };
enum class MagUnits   : uint8_t { UT_LIKE, NORMALIZED, COUNTS, UNKNOWN };

template <typename T>
static inline AccelUnits detect_accel_units(T med_norm) {
  if (med_norm > T(0.6) && med_norm < T(1.6))   return AccelUnits::G;
  if (med_norm > T(6.0) && med_norm < T(13.0))  return AccelUnits::MPS2;
  if (med_norm > T(600) && med_norm < T(1600))  return AccelUnits::MG;
  return AccelUnits::UNKNOWN;
}
template <typename T>
static inline T accel_radius_for_units(AccelUnits u) {
  switch (u) {
    case AccelUnits::G:    return T(1);
    case AccelUnits::MPS2: return T(9.80665);
    case AccelUnits::MG:   return T(1000);
    default:               return T(-1);
  }
}
template <typename T>
static inline MagUnits detect_mag_units(T med_norm) {
  if (med_norm > T(5) && med_norm < T(200))      return MagUnits::UT_LIKE;
  if (med_norm > T(0.2) && med_norm < T(2.5))    return MagUnits::NORMALIZED;
  if (med_norm > T(200) && med_norm < T(100000)) return MagUnits::COUNTS;
  return MagUnits::UNKNOWN;
}

// -----------------------------------------------------------------------------
// Ellipsoid fitter: ||L(x-b)|| ~= r, with L lower-triangular diag>0 via exp(logd).
// -----------------------------------------------------------------------------
template <typename T, int MaxN>
class EllipsoidFitter {
public:
  using Vec3 = Eigen::Matrix<T,3,1>;
  using Mat3 = Eigen::Matrix<T,3,3>;

  struct Result {
    bool ok = false;
    Vec3 bias = Vec3::Zero();
    Mat3 M = Mat3::Identity();
    T radius = T(1);
    T rms = T(0);
    int iters = 0;
  };

  void clear() {
    samples_.clear();
    b_.setZero(); off_.setZero(); logd_.setZero();
    estimate_radius_ = false;
    r_fixed_ = T(1);
    r_ = T(1);
    max_iters_ = 15;
    lambda0_ = T(1e-3);
    huber_k_ = T(0.05);
  }

  bool addSample(const Vec3& x) { return samples_.push(x); }
  int  size() const { return samples_.size(); }

  void setEstimateRadius(bool en) { estimate_radius_ = en; }
  void setFixedRadius(T r) { r_fixed_ = r; r_ = r; estimate_radius_ = false; }
  void setMaxIters(int it) { max_iters_ = it; }
  void setLambda0(T lam) { lambda0_ = lam; }
  void setHuberK(T k) { huber_k_ = std::max(T(1e-9), k); }

  void initializeFromData() {
    const int N = samples_.size();
    if (N <= 0) return;

    Vec3 mean = Vec3::Zero();
    for (int i=0;i<N;i++) mean += samples_[i];
    mean /= T(N);
    b_ = mean;

    T avg_norm = T(0);
    for (int i=0;i<N;i++) avg_norm += (samples_[i] - b_).norm();
    avg_norm = (avg_norm > T(1e-9)) ? (avg_norm / T(N)) : T(1);

    if (estimate_radius_) {
      r_ = avg_norm;
      logd_.setZero();
    } else {
      r_ = r_fixed_;
      const T s = std::max(T(1e-6), r_fixed_ / avg_norm);
      logd_ = Vec3::Constant(std::log(s));
    }
    off_.setZero();
  }

  Result fit() {
    if (samples_.size() < (estimate_radius_ ? 12 : 12)) return Result{};
    return estimate_radius_ ? fit10_() : fit9_();
  }

private:
  Mat3 buildL_() const {
    Mat3 L = Mat3::Zero();
    L(0,0) = std::exp(logd_(0));
    L(1,0) = off_(0);
    L(1,1) = std::exp(logd_(1));
    L(2,0) = off_(1);
    L(2,1) = off_(2);
    L(2,2) = std::exp(logd_(2));
    return L;
  }

  Result finalize_() const {
    Result out;
    out.ok = true;
    out.bias = b_;
    out.M = buildL_();
    out.radius = r_;

    const Mat3 L = out.M;
    T sum2 = T(0);
    int used = 0;
    for (int i=0;i<samples_.size();i++) {
      const Vec3 y = L * (samples_[i] - out.bias);
      const T s = y.norm();
      if (s < T(1e-9)) continue;
      const T e = s - out.radius;
      sum2 += e*e;
      used++;
    }
    out.rms = (used>0) ? std::sqrt(sum2 / T(used)) : T(0);
    return out;
  }

  Result fit9_() {
    using Mat9 = Eigen::Matrix<T,9,9>;
    using Vec9 = Eigen::Matrix<T,9,1>;

    T lambda = lambda0_;
    int last_it = 0;

    for (int it=0; it<max_iters_; ++it) {
      Mat9 JTJ = Mat9::Zero();
      Vec9 JTr = Vec9::Zero();
      T cost = T(0);
      int used = 0;

      const Mat3 L = buildL_();

      for (int i=0;i<samples_.size();i++) {
        const Vec3 x = samples_[i];
        const Vec3 v = x - b_;
        const Vec3 y = L * v;
        const T s = y.norm();
        if (s < T(1e-9)) continue;

        const T e = s - r_fixed_;
        const T w = huber_weight(e, huber_k_);
        cost += (w*e)*(w*e);
        used++;

        const Vec3 u = y / s;

        const Vec3 j_b = -(L.transpose() * u);
        const T j_l10 = u(1) * v(0);
        const T j_l20 = u(2) * v(0);
        const T j_l21 = u(2) * v(1);
        const T j_d0  = (u(0) * v(0)) * L(0,0);
        const T j_d1  = (u(1) * v(1)) * L(1,1);
        const T j_d2  = (u(2) * v(2)) * L(2,2);

        Vec9 J;
        J << j_b(0), j_b(1), j_b(2),
             j_l10, j_l20, j_l21,
             j_d0, j_d1, j_d2;

        const T ww = w*w;
        JTJ.noalias() += ww * (J * J.transpose());
        JTr.noalias() += ww * (J * e);
      }
      if (used < 9) break;

      Mat9 A = JTJ;
      for (int k=0;k<9;k++) A(k,k) += lambda;

      Eigen::LDLT<Mat9> ldlt(A);
      if (ldlt.info() != Eigen::Success) break;
      const Vec9 dx = ldlt.solve(-JTr);
      if (ldlt.info() != Eigen::Success) break;

      const Vec3 b0 = b_;
      const Vec3 off0 = off_;
      const Vec3 logd0 = logd_;

      b_ += dx.template segment<3>(0);
      off_(0) += dx(3); off_(1) += dx(4); off_(2) += dx(5);
      logd_(0) += dx(6); logd_(1) += dx(7); logd_(2) += dx(8);
      for (int k=0;k<3;k++) logd_(k) = clamp(logd_(k), T(-6), T(6));
      r_ = r_fixed_;

      // evaluate quickly
      const Mat3 Lnew = buildL_();
      T new_cost = T(0);
      int used2 = 0;
      for (int i=0;i<samples_.size();i++) {
        const Vec3 y = Lnew * (samples_[i] - b_);
        const T s = y.norm();
        if (s < T(1e-9)) continue;
        const T e = s - r_fixed_;
        const T w = huber_weight(e, huber_k_);
        new_cost += (w*e)*(w*e);
        used2++;
      }
      if (used2 < 9) new_cost = std::numeric_limits<T>::infinity();

      if (new_cost < cost) {
        lambda = std::max(lambda * T(0.3), T(1e-12));
        last_it = it+1;
        if (dx.norm() < T(1e-6)) break;
      } else {
        b_ = b0; off_ = off0; logd_ = logd0;
        lambda = std::min(lambda * T(10), T(1e6));
      }
    }

    Result out = finalize_();
    out.radius = r_fixed_;
    out.iters = last_it;
    return out;
  }

  Result fit10_() {
    using Mat10 = Eigen::Matrix<T,10,10>;
    using Vec10 = Eigen::Matrix<T,10,1>;

    T lambda = lambda0_;
    int last_it = 0;

    for (int it=0; it<max_iters_; ++it) {
      Mat10 JTJ = Mat10::Zero();
      Vec10 JTr = Vec10::Zero();
      T cost = T(0);
      int used = 0;

      const Mat3 L = buildL_();

      for (int i=0;i<samples_.size();i++) {
        const Vec3 x = samples_[i];
        const Vec3 v = x - b_;
        const Vec3 y = L * v;
        const T s = y.norm();
        if (s < T(1e-9)) continue;

        const T e = s - r_;
        const T w = huber_weight(e, huber_k_);
        cost += (w*e)*(w*e);
        used++;

        const Vec3 u = y / s;

        const Vec3 j_b = -(L.transpose() * u);
        const T j_l10 = u(1) * v(0);
        const T j_l20 = u(2) * v(0);
        const T j_l21 = u(2) * v(1);
        const T j_d0  = (u(0) * v(0)) * L(0,0);
        const T j_d1  = (u(1) * v(1)) * L(1,1);
        const T j_d2  = (u(2) * v(2)) * L(2,2);

        Vec10 J;
        J << j_b(0), j_b(1), j_b(2),
             j_l10, j_l20, j_l21,
             j_d0, j_d1, j_d2,
             T(-1);

        const T ww = w*w;
        JTJ.noalias() += ww * (J * J.transpose());
        JTr.noalias() += ww * (J * e);
      }
      if (used < 10) break;

      Mat10 A = JTJ;
      for (int k=0;k<10;k++) A(k,k) += lambda;

      Eigen::LDLT<Mat10> ldlt(A);
      if (ldlt.info() != Eigen::Success) break;
      const Vec10 dx = ldlt.solve(-JTr);
      if (ldlt.info() != Eigen::Success) break;

      const Vec3 b0 = b_;
      const Vec3 off0 = off_;
      const Vec3 logd0 = logd_;
      const T r0 = r_;

      b_ += dx.template segment<3>(0);
      off_(0) += dx(3); off_(1) += dx(4); off_(2) += dx(5);
      logd_(0) += dx(6); logd_(1) += dx(7); logd_(2) += dx(8);
      for (int k=0;k<3;k++) logd_(k) = clamp(logd_(k), T(-6), T(6));
      r_ += dx(9);
      r_ = std::max(T(1e-6), r_);

      const Mat3 Lnew = buildL_();
      T new_cost = T(0);
      int used2 = 0;
      for (int i=0;i<samples_.size();i++) {
        const Vec3 y = Lnew * (samples_[i] - b_);
        const T s = y.norm();
        if (s < T(1e-9)) continue;
        const T e = s - r_;
        const T w = huber_weight(e, huber_k_);
        new_cost += (w*e)*(w*e);
        used2++;
      }
      if (used2 < 10) new_cost = std::numeric_limits<T>::infinity();

      if (new_cost < cost) {
        lambda = std::max(lambda * T(0.3), T(1e-12));
        last_it = it+1;
        if (dx.norm() < T(1e-6)) break;
      } else {
        b_ = b0; off_ = off0; logd_ = logd0; r_ = r0;
        lambda = std::min(lambda * T(10), T(1e6));
      }
    }

    Result out = finalize_();
    out.iters = last_it;
    return out;
  }

  SampleBuffer3<T, MaxN> samples_;
  bool estimate_radius_ = false;
  T r_fixed_ = T(1);
  int max_iters_ = 15;
  T lambda0_ = T(1e-3);
  T huber_k_ = T(0.05);

  Vec3 b_ = Vec3::Zero();
  Vec3 off_ = Vec3::Zero();
  Vec3 logd_ = Vec3::Zero();
  T    r_ = T(1);
};

// Gyro bias estimator (stillness-only)
template <typename T>
class GyroBiasEstimator {
public:
  using Vec3 = Eigen::Matrix<T,3,1>;
  void reset() { n_ = 0; mean_.setZero(); m2_.setZero(); }

  void addStillSample(const Vec3& w) {
    n_++;
    const Vec3 delta = w - mean_;
    mean_ += delta / T(n_);
    const Vec3 delta2 = w - mean_;
    m2_ += delta.cwiseProduct(delta2);
  }

  int count() const { return n_; }
  Vec3 bias() const { return mean_; }
  Vec3 variance() const { return (n_ >= 2) ? (m2_ / T(n_ - 1)) : Vec3::Zero(); }
  bool ready(int min_samples) const { return n_ >= min_samples; }

private:
  int n_ = 0;
  Vec3 mean_ = Vec3::Zero();
  Vec3 m2_   = Vec3::Zero();
};

// Auto config: unit detect + thresholds + coverage gates
template <typename T, int MaxN>
struct AutoConfig {
  using Vec3 = Eigen::Matrix<T,3,1>;

  NormStats<T, MaxN> acc_stats;
  NormStats<T, MaxN> mag_stats;

  AccelUnits acc_units = AccelUnits::UNKNOWN;
  MagUnits   mag_units = MagUnits::UNKNOWN;

  T acc_radius = T(1);
  T acc_huber_k = T(0.05);
  T mag_huber_k = T(0.5);

  // Coverage gate defaults
  int min_samples_acc = 80;
  int min_samples_mag = 60;

  int min_bins_acc = 36; // 9x18 bins -> require >=36
  int min_bins_mag = 54; // require more mag diversity

  T acc_norm_gate_sigma = T(4);
  T mag_norm_gate_sigma = T(4);

  void finalize() {
    const T acc_med = acc_stats.median();
    const T mag_med = mag_stats.median();

    acc_units = detect_accel_units(acc_med);
    mag_units = detect_mag_units(mag_med);

    T r = accel_radius_for_units<T>(acc_units);
    if (r <= T(0)) r = (acc_med > T(1e-6)) ? acc_med : T(1);
    acc_radius = r;

    const T acc_mad = acc_stats.mad();
    acc_huber_k = std::max(T(0.02) * acc_radius, T(3) * acc_mad);

    const T mag_mad = mag_stats.mad();
    const T mag_scale = (mag_med > T(1e-6)) ? mag_med : T(1);
    mag_huber_k = std::max(T(0.03) * mag_scale, T(3) * mag_mad);
  }

  template <typename Fitter>
  void apply_accel(Fitter& f) const {
    f.setEstimateRadius(false);
    f.setFixedRadius(acc_radius);
    f.setHuberK(acc_huber_k);
  }

  template <typename Fitter>
  void apply_mag(Fitter& f) const {
    f.setEstimateRadius(true);
    f.setHuberK(mag_huber_k);
  }

  template <int ElevBins, int AzimBins>
  bool accel_has_coverage(const SampleBuffer3<T, MaxN>& acc_samples) const {
    if (acc_samples.size() < min_samples_acc) return false;

    CoverageBins<T, ElevBins, AzimBins> cov; cov.clear();
    const T med = acc_stats.median();
    const T mad = acc_stats.mad();
    const T gate = acc_norm_gate_sigma * mad + T(0.08) * med;

    for (int i=0;i<acc_samples.size();i++) {
      const Vec3 a = acc_samples[i];
      const T n = a.norm();
      if (std::abs(n - med) > gate) continue;
      cov.add_dir(a);
    }
    return cov.count() >= min_bins_acc;
  }

  template <int ElevBins, int AzimBins>
  bool mag_has_coverage(const SampleBuffer3<T, MaxN>& mag_samples) const {
    if (mag_samples.size() < min_samples_mag) return false;

    CoverageBins<T, ElevBins, AzimBins> cov; cov.clear();
    const T med = mag_stats.median();
    const T mad = mag_stats.mad();
    const T gate = mag_norm_gate_sigma * mad + T(0.15) * med;

    for (int i=0;i<mag_samples.size();i++) {
      const Vec3 m = mag_samples[i];
      const T n = m.norm();
      if (std::abs(n - med) > gate) continue;
      cov.add_dir(m);
    }
    return cov.count() >= min_bins_mag;
  }
};

// -----------------------------------------------------------------------------
// Rate-controlled IMU calibrator
// -----------------------------------------------------------------------------
template <typename T, int MaxN>
class ImuEllipsoidCalibrator {
public:
  using Vec3 = Eigen::Matrix<T,3,1>;
  using Mat3 = Eigen::Matrix<T,3,3>;

  struct Rates {
    // Desired *calibration sampling* rates (not your sensor ODR).
    // You can call update() at any frequency; internally it downsamples.
    T acc_gyro_cal_hz = T(50); // e.g., collect 50 Hz worth of unique orientations
    T mag_cal_hz      = T(20); // mag often 20-100 Hz ODR; you may want 10-25 Hz for cal

    // Minimum dt guard (avoid division by 0)
    T min_dt = T(1e-4);
  };

  struct Stillness {
    // Used only for gyro bias accumulation:
    // - require accel norm near expected (computed from auto-detected units once ready)
    // - require gyro magnitude small
    T gyro_max_norm = T(0.08);     // rad/s (tune)
    T accel_norm_tol_frac = T(0.06); // fraction of expected |a| (6%)
    int min_still_samples = 300;   // ~ few seconds worth
  };

  struct CalResult {
    bool ok_acc = false;
    bool ok_mag = false;
    bool ok_gyro = false;

    typename EllipsoidFitter<T, MaxN>::Result accel;
    typename EllipsoidFitter<T, MaxN>::Result mag;

    Vec3 gyro_bias = Vec3::Zero();

    AccelUnits accel_units = AccelUnits::UNKNOWN;
    MagUnits   mag_units   = MagUnits::UNKNOWN;

    // Coverage diagnostics (bins hit)
    int acc_bins = 0;
    int mag_bins = 0;
  };

  void reset() {
    acc_fit_.clear();
    mag_fit_.clear();
    gyro_bias_.reset();
    cfg_ = AutoConfig<T, MaxN>();
    acc_samples_.clear();
    mag_samples_.clear();
    t_acc_ = T(0);
    t_mag_ = T(0);
    last_expected_acc_norm_ = T(0);
  }

  Rates& rates() { return rates_; }
  Stillness& stillness() { return still_; }
  AutoConfig<T, MaxN>& config() { return cfg_; }

  // Call this in your main loop.
  // dt_sec: elapsed time since last call in seconds.
  // mag_valid: true only when you actually have a new mag sample (if you run mag at lower ODR).
  void update(const Vec3& accel_raw,
              const Vec3& gyro_raw,
              bool mag_valid,
              const Vec3& mag_raw,
              T dt_sec)
  {
    dt_sec = std::max(dt_sec, rates_.min_dt);

    // --- rate-controlled accel/gyro sampling ---
    t_acc_ += dt_sec;
    const T acc_period = (rates_.acc_gyro_cal_hz > T(1e-6)) ? (T(1) / rates_.acc_gyro_cal_hz) : T(1e9);

    while (t_acc_ >= acc_period && acc_samples_.size() < MaxN) {
      t_acc_ -= acc_period;

      // store accel sample
      cfg_.acc_stats.add_vec(accel_raw);
      acc_samples_.push(accel_raw);
      acc_fit_.addSample(accel_raw);

      // try to accumulate gyro bias if "still"
      maybe_add_gyro_still_(accel_raw, gyro_raw);
    }

    // --- rate-controlled magnetometer sampling ---
    if (mag_valid) {
      t_mag_ += dt_sec;
      const T mag_period = (rates_.mag_cal_hz > T(1e-6)) ? (T(1) / rates_.mag_cal_hz) : T(1e9);

      while (t_mag_ >= mag_period && mag_samples_.size() < MaxN) {
        t_mag_ -= mag_period;

        cfg_.mag_stats.add_vec(mag_raw);
        mag_samples_.push(mag_raw);
        mag_fit_.addSample(mag_raw);
      }
    }
  }

  int accel_count() const { return acc_samples_.size(); }
  int mag_count() const { return mag_samples_.size(); }
  int gyro_still_count() const { return gyro_bias_.count(); }

  CalResult calibrate() {
    CalResult out;

    // 1) finalize auto units + thresholds (uses norms from collected samples)
    cfg_.finalize();
    out.accel_units = cfg_.acc_units;
    out.mag_units = cfg_.mag_units;

    // expected accel norm for stillness check (now meaningful)
    last_expected_acc_norm_ = cfg_.acc_radius;

    // 2) coverage gating + diagnostics (9x18)
    out.acc_bins = compute_bins_<9,18>(acc_samples_, cfg_.acc_stats, /*is_accel=*/true);
    out.mag_bins = compute_bins_<9,18>(mag_samples_, cfg_.mag_stats, /*is_accel=*/false);

    const bool acc_ok = cfg_.template accel_has_coverage<9,18>(acc_samples_);
    const bool mag_ok = cfg_.template mag_has_coverage<9,18>(mag_samples_);

    // 3) fit accel
    if (acc_ok) {
      cfg_.apply_accel(acc_fit_);
      acc_fit_.initializeFromData();
      out.accel = acc_fit_.fit();
      out.ok_acc = out.accel.ok;
    }

    // 4) fit mag
    if (mag_ok) {
      cfg_.apply_mag(mag_fit_);
      mag_fit_.initializeFromData();
      out.mag = mag_fit_.fit();
      out.ok_mag = out.mag.ok;
    }

    // 5) gyro bias
    if (gyro_bias_.ready(still_.min_still_samples)) {
      out.gyro_bias = gyro_bias_.bias();
      out.ok_gyro = true;
    }

    return out;
  }

  // Apply helper
  static inline Vec3 apply_cal(const typename EllipsoidFitter<T, MaxN>::Result& r, const Vec3& x) {
    return r.M * (x - r.bias);
  }

private:
  // Minimal stillness detector for gyro bias:
  // - accel norm near expected (once expected known; else use current median proxy)
  // - gyro magnitude small
  void maybe_add_gyro_still_(const Vec3& accel_raw, const Vec3& gyro_raw) {
    const T wnorm = gyro_raw.norm();
    if (wnorm > still_.gyro_max_norm) return;

    // Determine expected accel norm:
    // If we haven't finalized, use running median if available; else accept none.
    T expected = last_expected_acc_norm_;
    if (expected <= T(0)) {
      if (cfg_.acc_stats.size() < 20) return;
      expected = cfg_.acc_stats.median();
      if (expected <= T(1e-6)) return;
    }

    const T an = accel_raw.norm();
    const T tol = std::max(T(0.02) * expected, still_.accel_norm_tol_frac * expected);
    if (std::abs(an - expected) > tol) return;

    gyro_bias_.addStillSample(gyro_raw);
  }

  template <int EB, int AB>
  int compute_bins_(const SampleBuffer3<T, MaxN>& samples, const NormStats<T, MaxN>& stats, bool is_accel) const {
    CoverageBins<T, EB, AB> cov; cov.clear();
    const T med = stats.median();
    const T mad = stats.mad();
    if (stats.size() < 10) return 0;

    const T gate = (is_accel ? (cfg_.acc_norm_gate_sigma * mad + T(0.08) * med)
                             : (cfg_.mag_norm_gate_sigma * mad + T(0.15) * med));

    for (int i=0;i<samples.size();i++) {
      const Vec3 v = samples[i];
      const T n = v.norm();
      if (std::abs(n - med) > gate) continue;
      cov.add_dir(v);
    }
    return cov.count();
  }

  Rates rates_;
  Stillness still_;
  AutoConfig<T, MaxN> cfg_;

  SampleBuffer3<T, MaxN> acc_samples_;
  SampleBuffer3<T, MaxN> mag_samples_;

  EllipsoidFitter<T, MaxN> acc_fit_;
  EllipsoidFitter<T, MaxN> mag_fit_;
  GyroBiasEstimator<T>     gyro_bias_;

  // time accumulators for rate-controlled sampling
  T t_acc_ = T(0);
  T t_mag_ = T(0);

  // once calibrate() has run, we keep expected accel norm for stillness gate
  T last_expected_acc_norm_ = T(0);
};

} // namespace imu_cal

// -----------------------------------------------------------------------------
// Example usage:
//
// imu_cal::ImuEllipsoidCalibrator<float, 300> cal;
// cal.reset();
//
// // Configure your desired *calibration sampling* rates:
// cal.rates().acc_gyro_cal_hz = 60.0f;  // downsample accel+gyro samples used for cal
// cal.rates().mag_cal_hz      = 20.0f;  // downsample mag samples used for cal
//
// // Stillness tuning (gyro bias):
// cal.stillness().gyro_max_norm = 0.06f;      // rad/s
// cal.stillness().accel_norm_tol_frac = 0.05f; // 5%
//
// loop:
//   dt = seconds since last loop
//   accel_raw (Vec3), gyro_raw (Vec3)
//   mag_valid = true only when you got a fresh mag sample (if mag ODR lower)
//   mag_raw (Vec3) if mag_valid
//   cal.update(accel_raw, gyro_raw, mag_valid, mag_raw, dt);
//
// When you think you have enough coverage (e.g. accel_count ~ 250, mag_count ~ 200):
//   auto res = cal.calibrate();
//
// Apply:
//   if (res.ok_acc) a_cal = ImuEllipsoidCalibrator<float,300>::apply_cal(res.accel, accel_raw);
//   if (res.ok_mag) m_cal = ImuEllipsoidCalibrator<float,300>::apply_cal(res.mag, mag_raw);
//   if (res.ok_gyro) w_cal = gyro_raw - res.gyro_bias;
// -----------------------------------------------------------------------------

