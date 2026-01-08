#pragma once

/*
  Copyright 2026, Mikhail Grushinskiy

  Modern embedded-friendly IMU calibration for Arduino + Eigen.
  Units:
    accel: m/s^2
    mag:   uT
    gyro:  rad/s
*/

#include <stdint.h>
#include <cmath>

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#else
#include <ArduinoEigenDense.h>
#endif

namespace imu_cal {

// Config / limits
static constexpr int IMU_CAL_MAX_SAMPLES = 400;

// Utilities
template <typename T>
static inline T clamp(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }

template <typename T>
static inline bool finiteT(T x) { return std::isfinite((double)x); }

template <typename T>
static inline bool isfinite3(const Eigen::Matrix<T,3,1>& v) {
  return finiteT(v.x()) && finiteT(v.y()) && finiteT(v.z());
}

// SPD projection for 3x3 symmetric matrices
// Clamps eigenvalues to enforce positive-definiteness.
// Returns false if eigendecomposition fails or result is non-finite.
template <typename T>
static inline bool project_spd_3x3(
    Eigen::Matrix<T,3,3>& M,
    T eps_rel = T(1e-6),
    T eps_abs = T(1e-9))
{
  using Mat3 = Eigen::Matrix<T,3,3>;
  // Symmetrize first
  M = T(0.5) * (M + M.transpose());

  Eigen::SelfAdjointEigenSolver<Mat3> es(M);
  if (es.info() != Eigen::Success) return false;

  Eigen::Matrix<T,3,1> eval = es.eigenvalues();
  Mat3 evec = es.eigenvectors();
  if (!(eval.array().isFinite().all()) || !(evec.array().isFinite().all())) return false;

  const T max_e = eval.maxCoeff();
  const T floor_e = (max_e > T(0)) ? (eps_rel * max_e) : eps_abs;
  const T min_e = (floor_e > eps_abs) ? floor_e : eps_abs;

  eval.x() = (eval.x() < min_e ? min_e : eval.x());
  eval.y() = (eval.y() < min_e ? min_e : eval.y());
  eval.z() = (eval.z() < min_e ? min_e : eval.z());

  M = evec * eval.asDiagonal() * evec.transpose();
  M = T(0.5) * (M + M.transpose());
  return (M.array().isFinite().all());
}

template <typename T>
static inline bool degeneracy_check_coverage3(
    const Eigen::Matrix<T,3,1>* x, int n,
    T expected_radius,                 // e.g. g for accel, ~B for mag
    T min_axis_span_mult = T(0.90),    // require span >= 0.90*R on each axis (after centering)
    T min_sign_frac      = T(0.08),    // require >=8% of samples on both +/- side per axis (after centering)
    T min_cov_det        = T(1e-3))    // covariance determinant threshold for unit directions
{
  if (!x || n < 12) return false;
  if (!(expected_radius > T(0))) return false;

  using Vec3 = Eigen::Matrix<T,3,1>;
  using Mat3 = Eigen::Matrix<T,3,3>;

  // Mean-center to remove bias offsets.
  Vec3 mu = Vec3::Zero();
  for (int i = 0; i < n; ++i) mu += x[i];
  mu *= (T(1) / (T)n);

  // Axis span and sign balance checks (centered).
  Vec3 mn = Vec3::Constant( T(1e30));
  Vec3 mx = Vec3::Constant(-T(1e30));
  int pos[3] = {0,0,0};
  int neg[3] = {0,0,0};

  // Direction covariance on unit vectors u = (x-mu)/||x-mu||
  Mat3 C = Mat3::Zero();
  int m = 0;

  int valid = 0; // count of finite centered samples used for sign/span tests

  for (int i = 0; i < n; ++i) {
    Vec3 d = x[i] - mu;
    if (!isfinite3(d)) continue;

    ++valid;

    // track min/max
    mn.x() = (d.x() < mn.x() ? d.x() : mn.x());
    mn.y() = (d.y() < mn.y() ? d.y() : mn.y());
    mn.z() = (d.z() < mn.z() ? d.z() : mn.z());
    mx.x() = (d.x() > mx.x() ? d.x() : mx.x());
    mx.y() = (d.y() > mx.y() ? d.y() : mx.y());
    mx.z() = (d.z() > mx.z() ? d.z() : mx.z());

    // sign counts
    if (d.x() > T(0)) pos[0]++; else if (d.x() < T(0)) neg[0]++;
    if (d.y() > T(0)) pos[1]++; else if (d.y() < T(0)) neg[1]++;
    if (d.z() > T(0)) pos[2]++; else if (d.z() < T(0)) neg[2]++;

    const T dn = d.norm();
    if (!(dn > T(1e-9))) continue;

    Vec3 u = d / dn;
    if (!isfinite3(u)) continue;

    C.noalias() += u * u.transpose();
    ++m;
  }

  if (valid < 12) return false;
  if (m < 12) return false;

  // Axis span must be large enough (requires coverage in each direction).
  const Vec3 span = mx - mn;
  const T min_span = min_axis_span_mult * expected_radius;

  if (!(span.x() > min_span && span.y() > min_span && span.z() > min_span)) return false;

  // Must have both positive and negative samples on each axis (after centering).
  const int min_side = (int)ceil((double)valid * (double)min_sign_frac);
  for (int a = 0; a < 3; ++a) {
    if (pos[a] < min_side || neg[a] < min_side) return false;
  }

  // Normalize covariance; for uniform sphere C ~ (1/3)I, det ~ 1/27 ≈ 0.037.
  C *= (T(1) / (T)m);

  // Determinant collapses to ~0 if samples are planar/linear in direction space.
  const T detC = C.determinant();
  if (!finiteT(detC) || detC < min_cov_det) return false;

  return true;
}

template <typename T>
static inline T sqr(T x) { return x*x; }

template <typename T>
static inline void sort_small(T* a, int n) {
  // Simple insertion sort (n <= ~400)
  for (int i = 1; i < n; ++i) {
    T key = a[i];
    int j = i - 1;
    while (j >= 0 && a[j] > key) { a[j+1] = a[j]; --j; }
    a[j+1] = key;
  }
}

template <typename T>
static inline T median_of_array(T* a, int n) {
  if (n <= 0) return T(0);
  sort_small(a, n);
  if (n & 1) return a[n/2];
  return T(0.5) * (a[n/2 - 1] + a[n/2]);
}

template <typename T>
static inline T robust_mad(T* residuals, int n) {
  // MAD = median(|r - median(r)|)
  if (n <= 0) return T(0);
  T med = median_of_array(residuals, n);
  for (int i = 0; i < n; ++i) residuals[i] = (T)fabs((double)(residuals[i] - med));
  T mad = median_of_array(residuals, n);
  return mad;
}

// Fixed-capacity sample buffer
template <typename T, int N>
struct SampleBuffer3 {
  static_assert(N > 0, "N must be positive");
  static_assert(N <= IMU_CAL_MAX_SAMPLES, "N exceeds IMU_CAL_MAX_SAMPLES (400)");

  using Vec3 = Eigen::Matrix<T,3,1>;

  Vec3 v[N];
  T    tempC[N];
  int  n = 0;

  void clear() { n = 0; }

  bool push(const Vec3& x, T tC = T(0)) {
    if (n >= N) return false;
    if (!isfinite3(x) || !finiteT(tC)) return false;
    v[n] = x;
    tempC[n] = tC;
    ++n;
    return true;
  }
};

// Ellipsoid -> sphere robust fit
// Implicit model matching build_row_d():
//    x^T Q x + 2 q^T x + c = 1
// Row d = [x^2 y^2 z^2 2xy 2xz 2yz 2x 2y 2z 1]
// Solve LS: D p ≈ 1 (i.e., minimize ||D p - 1||).
//
// Recover:
//   b = -Q^{-1} q
//   s = 1 + b^T Q b - c   (must be > 0)
//   Then: (x-b)^T (Q/s) (x-b) = 1
// Whitening A_unit from chol(Q/s): ||A_unit(x-b)|| ≈ 1
// Then scale A = A_unit * R_target so ||A(x-b)|| ≈ R_target.
template <typename T>
struct EllipsoidSphereFit {
  bool ok = false;
  Eigen::Matrix<T,3,1> b = Eigen::Matrix<T,3,1>::Zero();
  Eigen::Matrix<T,3,3> A = Eigen::Matrix<T,3,3>::Identity(); // scaled whitening

  // Diagnostics
  T rms = T(0);     // RMS of (||A(x-b)|| - R_target)
  T median_r = T(0);
  int used = 0;
};

template <typename T>
static inline void build_row_d(const Eigen::Matrix<T,3,1>& x, Eigen::Matrix<T,10,1>& d) {
  const T X=x.x(), Y=x.y(), Z=x.z();
  d << X*X, Y*Y, Z*Z,
       T(2)*X*Y, T(2)*X*Z, T(2)*Y*Z,
       T(2)*X, T(2)*Y, T(2)*Z,
       T(1);
}

template <typename T>
static bool solve_ellipsoid_params_trimmed(
    const Eigen::Matrix<T,3,1>* x, int n,
    const bool* inlier,
    Eigen::Matrix<T,10,1>& p_out,
    T ridge_rel)
{
  using Mat10 = Eigen::Matrix<T,10,10>;
  using Vec10 = Eigen::Matrix<T,10,1>;

  if (!x || n <= 0) return false;

  Mat10 H = Mat10::Zero();
  Vec10 g = Vec10::Zero();

  int used = 0;
  for (int i = 0; i < n; ++i) {
    if (inlier && !inlier[i]) continue;
    if (!isfinite3<T>(x[i])) continue;              // <-- DROP-IN: skip non-finite

    Vec10 d;
    build_row_d(x[i], d);
    if (!(d.array().isFinite().all())) continue;    // paranoia

    H.noalias() += d * d.transpose();
    g.noalias() += d; // D^T * 1
    ++used;
  }
  if (used < 10) return false;

  // Scale ridge by feature energy for better invariance
  const T tr = H.trace() / T(10);
  const T ridge = ridge_rel * (tr + T(1e-12));
  H.diagonal().array() += ridge;

  Eigen::LDLT<Mat10> ldlt(H);
  if (ldlt.info() != Eigen::Success) return false;

  Vec10 p = ldlt.solve(g);
  if (!(p.array().isFinite().all())) return false;

  p_out = p;
  return true;
}

template <typename T>
static EllipsoidSphereFit<T> ellipsoid_to_sphere_robust(
    const Eigen::Matrix<T,3,1>* x, int n,
    T R_target,
    int robust_iters = 3,
    T trim_frac = T(0.15),     // drop worst ~15% by |e| each iteration (now implemented)
    T ridge_rel = T(1e-6),
    T expected_radius_for_checks = T(0)) // optional early coverage check
{
  EllipsoidSphereFit<T> out;
  if (!x || n < 12) return out;
  if (n > IMU_CAL_MAX_SAMPLES) return out;
  if (!(R_target > T(0))) return out;

  // Optional early degeneracy check (coverage/planarity)
  if (expected_radius_for_checks > T(0)) {
    if (!degeneracy_check_coverage3<T>(
          x, n,
          expected_radius_for_checks,
          T(0.90),   // min axis span ~0.90*R
          T(0.08),   // require both sides ~8%
          T(1e-3)))  // direction covariance determinant threshold
    {
      return out;
    }
  }

  bool inlier[IMU_CAL_MAX_SAMPLES];
  for (int i = 0; i < n; ++i) inlier[i] = true;

  // Working state
  Eigen::Matrix<T,10,1> p;
  Eigen::Matrix<T,3,1>  b = Eigen::Matrix<T,3,1>::Zero();
  Eigen::Matrix<T,3,3>  A_unit = Eigen::Matrix<T,3,3>::Identity();

  // Robust params
  robust_iters = (robust_iters <= 0 ? 1 : robust_iters);
  trim_frac = clamp<T>(trim_frac, T(0), T(0.45));

  // Residual buffers (unit-sphere space): e = r - 1
  T e[IMU_CAL_MAX_SAMPLES];
  T abs_e[IMU_CAL_MAX_SAMPLES];
  T tmp[IMU_CAL_MAX_SAMPLES];

  // Helper: rebuild b/A_unit from p and validate
  auto build_model_from_p = [&](const Eigen::Matrix<T,10,1>& p_in) -> bool {
    Eigen::Matrix<T,3,3> Q;
    Q << p_in(0), p_in(3), p_in(4),
         p_in(3), p_in(1), p_in(5),
         p_in(4), p_in(5), p_in(2);
    Eigen::Matrix<T,3,1> q;
    q << p_in(6), p_in(7), p_in(8);
    const T c = p_in(9);

    // b = -Q^{-1} q
    Eigen::FullPivLU<Eigen::Matrix<T,3,3>> lu(Q);
    if (!lu.isInvertible()) return false;
    b = -lu.solve(q);
    if (!isfinite3(b)) return false;

    // s = 1 + b^T Q b - c  (must be > 0)
    const T s = T(1) + b.dot(Q * b) - c;
    if (!finiteT(s) || s <= T(0)) return false;

    // M = Q / s, symmetrize
    Eigen::Matrix<T,3,3> M = Q / s;
    M = T(0.5) * (M + M.transpose());

    // project to SPD to avoid borderline LLT failures
    if (!project_spd_3x3<T>(M, T(1e-6), T(1e-9))) return false;

    // Cholesky: M = U^T U
    Eigen::LLT<Eigen::Matrix<T,3,3>> llt(M);
    if (llt.info() != Eigen::Success) return false;
    A_unit = llt.matrixU();
    return true;    
  };

  for (int it = 0; it < robust_iters; ++it) {
    if (!solve_ellipsoid_params_trimmed<T>(x, n, inlier, p, ridge_rel)) return out;
    if (!build_model_from_p(p)) return out;

    // Compute residuals for ALL points: e = ||A_unit(x-b)|| - 1
    for (int i = 0; i < n; ++i) {
      const T r = (A_unit * (x[i] - b)).norm();
      T ei = r - T(1);
      if (!finiteT(ei)) {
        // mark as huge outlier
        ei = T(1e9);
      }
      e[i] = ei;
      abs_e[i] = (T)fabs((double)ei);
    }

    // Robust scale via MAD
    for (int i = 0; i < n; ++i) tmp[i] = e[i];
    const T mad = robust_mad(tmp, n);

    // MAD floor to avoid mad->0 making thr ~ 1e-6
    // (units are "fractional radius" since r is near 1)
    const T mad_floor = T(1e-4);
    const T mad_eff   = (mad > mad_floor ? mad : mad_floor);

    // MAD threshold
    const T thr_mad = T(5.2) * mad_eff + T(1e-6);

    // Quantile trim threshold (drop worst trim_frac)
    T thr_trim = thr_mad; // default if trim_frac==0
    if (trim_frac > T(0)) {
      for (int i = 0; i < n; ++i) tmp[i] = abs_e[i];
      sort_small(tmp, n);
      int keep = (int)floor((double)((T(1) - trim_frac) * (T)n));
      keep = clamp<int>(keep, 10, n);
      thr_trim = tmp[keep - 1];
    }

    // Combine: be conservative (tighter) first
    T thr = (thr_trim < thr_mad ? thr_trim : thr_mad);

    // Update inliers
    int used = 0;
    for (int i = 0; i < n; ++i) {
      inlier[i] = (abs_e[i] <= thr);
      if (inlier[i]) ++used;
    }

    // If we got too aggressive, relax to the looser of the two thresholds
    if (used < 10) {
      thr = (thr_trim > thr_mad ? thr_trim : thr_mad);
      used = 0;
      for (int i = 0; i < n; ++i) {
        inlier[i] = (abs_e[i] <= thr);
        if (inlier[i]) ++used;
      }
      if (used < 10) return out;
    }
  }

  // IMPORTANT: final solve using FINAL inlier mask,
  // so the model matches the mask we score with.
  if (!solve_ellipsoid_params_trimmed<T>(x, n, inlier, p, ridge_rel)) return out;
  if (!build_model_from_p(p)) return out;

  // Final scale A to target radius
  const Eigen::Matrix<T,3,3> A = A_unit * R_target;

  int used = 0;
  T sum_e2 = 0;
  T radii[IMU_CAL_MAX_SAMPLES];

  for (int i = 0; i < n; ++i) {
    if (!inlier[i]) continue;
    const T r = (A * (x[i] - b)).norm();
    if (!finiteT(r)) continue;
    radii[used] = r;
    const T er = r - R_target;
    sum_e2 += er * er;
    ++used;
  }
  if (used < 10) return out;

  const T medr = median_of_array(radii, used);

  out.ok = true;
  out.b = b;
  out.A = A;
  out.rms = (T)sqrt((double)(sum_e2 / (T)used));
  out.median_r = medr;
  out.used = used;
  return out;
}

// Temperature model: bias(T) = b0 + k*(T - T0)
// Fit per-axis with simple LS.
template <typename T>
struct TempBias3 {
  bool ok = false;
  T T0 = T(25);
  Eigen::Matrix<T,3,1> b0 = Eigen::Matrix<T,3,1>::Zero();
  Eigen::Matrix<T,3,1> k  = Eigen::Matrix<T,3,1>::Zero();

  Eigen::Matrix<T,3,1> bias(T tempC) const { return b0 + k * (tempC - T0); }
};

template <typename T, int N>
static TempBias3<T> fit_temp_bias3(const Eigen::Matrix<T,3,1>(&b)[N], const T(&t)[N], int n, T T0) {
  TempBias3<T> out;
  out.T0 = T0;
  if (n < 2) return out;

  T Sx = 0, Sxx = 0;
  Eigen::Matrix<T,3,1> Sy = Eigen::Matrix<T,3,1>::Zero();
  Eigen::Matrix<T,3,1> Sxy = Eigen::Matrix<T,3,1>::Zero();

  for (int i = 0; i < n; ++i) {
    const T x = t[i] - T0;
    Sx += x;
    Sxx += x*x;
    Sy += b[i];
    Sxy += b[i] * x;
  }
  const T det = (T)n * Sxx - Sx*Sx;
  if (fabs((double)det) < 1e-9) return out;

  out.b0 = (Sy * Sxx - Sxy * Sx) / det;
  out.k  = (Sxy * (T)n - Sy * Sx) / det;
  out.ok = true;
  return out;
}

// Calibration outputs
template <typename T>
struct AccelCalibration {
  bool ok = false;
  T g = T(9.80665);
  Eigen::Matrix<T,3,3> S = Eigen::Matrix<T,3,3>::Identity(); // a_cal = S*(a_raw - bias(T))
  TempBias3<T> biasT;
  T rms_mag = T(0);

  Eigen::Matrix<T,3,1> apply(const Eigen::Matrix<T,3,1>& a_raw, T tempC) const {
    return S * (a_raw - biasT.bias(tempC));
  }
};

template <typename T>
struct MagCalibration {
  bool ok = false;
  Eigen::Matrix<T,3,3> A = Eigen::Matrix<T,3,3>::Identity(); // m_cal = A*(m_raw - b)
  Eigen::Matrix<T,3,1> b = Eigen::Matrix<T,3,1>::Zero();
  T field_uT = T(0);
  T rms = T(0);

  Eigen::Matrix<T,3,1> apply(const Eigen::Matrix<T,3,1>& m_raw) const {
    return A * (m_raw - b);
  }
};

template <typename T>
struct GyroCalibration {
  bool ok = false;
  Eigen::Matrix<T,3,3> S = Eigen::Matrix<T,3,3>::Identity(); // keep identity unless you have a known-rate rig
  TempBias3<T> biasT;

  Eigen::Matrix<T,3,1> apply(const Eigen::Matrix<T,3,1>& w_raw, T tempC) const {
    return S * (w_raw - biasT.bias(tempC));
  }
};

// Calibrator: Accel
// Strategy:
//  - Collect ~400 quasi-static accel samples across orientations.
//  - Bin by temperature into up to K bins.
//  - For each bin: robust ellipsoid->sphere fit => center b_bin and S_bin.
//  - Choose S from best bin (low rms, enough samples).
//  - Regress b_bin vs temperature => bias(T).
template <typename T, int N, int K_TBINS = 8>
struct AccelCalibrator {
  static_assert(N <= IMU_CAL_MAX_SAMPLES, "N exceeds IMU_CAL_MAX_SAMPLES (400)");
  static_assert(K_TBINS > 0 && K_TBINS <= 16, "K_TBINS unreasonable");

  using Vec3 = Eigen::Matrix<T,3,1>;
  SampleBuffer3<T, N> buf;

  T g = T(9.80665);
  T T0 = T(25);

  // acceptance gates
  T max_gyro_for_static = T(0.35);  // rad/s
  T accel_mag_tol = T(1.0);         // m/s^2, accept if | |a| - g | < tol

  // Workspace to reduce stack use in fit()
  struct Workspace {
    uint16_t idx[K_TBINS][N];
    int      nbin[K_TBINS];
    T        tsum[K_TBINS];
    Vec3     xscratch[N];
    Vec3     centers[K_TBINS];
    T        temps[K_TBINS];
  };

  // fit() is const, so workspace must be mutable
  mutable Workspace ws_;

  void clear() { buf.clear(); }

  bool addSample(const Vec3& a_raw, const Vec3& w_raw, T tempC) {
    if (!isfinite3(a_raw) || !isfinite3(w_raw) || !finiteT(tempC)) return false;
    const T amag = a_raw.norm();
    const T wmag = w_raw.norm();
    if ((T)fabs((double)(amag - g)) > accel_mag_tol) return false;
    if (wmag > max_gyro_for_static) return false;
    return buf.push(a_raw, tempC);
  }

  bool fit(AccelCalibration<T>& out, int robust_iters = 3, T trim_frac = T(0.15)) const {
    if (buf.n < 80) return false;

    // Temperature range
    T tmin = buf.tempC[0], tmax = buf.tempC[0];
    for (int i = 1; i < buf.n; ++i) {
      tmin = (buf.tempC[i] < tmin ? buf.tempC[i] : tmin);
      tmax = (buf.tempC[i] > tmax ? buf.tempC[i] : tmax);
    }
    const T trange = tmax - tmin;
    const T binW = (trange > T(1e-3)) ? (trange / (T)K_TBINS) : T(1);

    // Use reusable workspace (reduces stack)
    auto& idx   = ws_.idx;
    auto& nbin  = ws_.nbin;
    auto& tsum  = ws_.tsum;
    auto& xs    = ws_.xscratch;
    auto& centers = ws_.centers;
    auto& temps   = ws_.temps;

    for (int k = 0; k < K_TBINS; ++k) { nbin[k] = 0; tsum[k] = T(0); }

    for (int i = 0; i < buf.n; ++i) {
      int k = 0;
      if (trange > T(1e-3)) {
        k = (int)floor((double)((buf.tempC[i] - tmin) / binW));
        k = clamp<int>(k, 0, K_TBINS - 1);
      }
      int& nk = nbin[k];
      if (nk < N) {
        idx[k][nk] = (uint16_t)i;
        tsum[k] += buf.tempC[i];
        ++nk;
      }
    }

    int nb = 0;

    // Choose S from best bin
    T best_score = T(1e30);
    Eigen::Matrix<T,3,3> bestS = Eigen::Matrix<T,3,3>::Identity();
    T best_rms = T(0);
    Vec3 best_center = Vec3::Zero();    

    for (int k = 0; k < K_TBINS; ++k) {
      if (nbin[k] < 30) continue;

      for (int j = 0; j < nbin[k]; ++j) {
        xs[j] = buf.v[(int)idx[k][j]];
      }

      const T tmean = tsum[k] / (T)nbin[k];

      auto fitk = ellipsoid_to_sphere_robust<T>(
        xs, nbin[k], g,
        robust_iters, trim_frac,
        T(1e-6), // ridge_rel (matches default)
        g);      // expected_radius_for_checks
      if (!fitk.ok) continue;

      centers[nb] = fitk.b;
      temps[nb]   = tmean;
      nb++;      

      // score: rms + small penalty for fewer used points
      T score = fitk.rms + T(0.2) * (T(50) / (T)fitk.used);
      if (score < best_score) {
        best_score = score;
        bestS = fitk.A;
        best_rms = fitk.rms;
        best_center = fitk.b;
      }
    }

    if (nb < 1) return false;

    // Fit bias(T) from bin centers
    TempBias3<T> biasT;
    if (nb >= 2) {
      Eigen::Matrix<T,3,1> btmp[K_TBINS];
      T ttmp[K_TBINS];
      for (int i = 0; i < nb; ++i) { btmp[i] = centers[i]; ttmp[i] = temps[i]; }
      biasT = fit_temp_bias3<T, K_TBINS>(btmp, ttmp, nb, T0);
    } else {
      biasT.ok = true;
      biasT.T0 = T0;
      biasT.b0 = best_center;
      biasT.k.setZero();
    }

    out.ok = true;
    out.g = g;
    out.S = bestS;
    out.biasT = biasT;
    out.rms_mag = best_rms;
    return true;
  }
};

// Calibrator: Magnetometer
// Strategy:
//  - Collect mag samples across orientations.
//  - Robust ellipsoid->unit sphere gives (b, A_unit).
//  - Preserve µT magnitude by scaling so median(||m_raw-b||) is maintained.
template <typename T, int N>
struct MagCalibrator {
  static_assert(N <= IMU_CAL_MAX_SAMPLES, "N exceeds IMU_CAL_MAX_SAMPLES (400)");
  using Vec3 = Eigen::Matrix<T,3,1>;
  SampleBuffer3<T, N> buf;

  // sanity gates
  T min_norm_uT = T(5);
  T max_norm_uT = T(200);

  void clear() { buf.clear(); }

  bool addSample(const Vec3& m_raw_uT) {
    if (!isfinite3(m_raw_uT)) return false;
    const T nrm = m_raw_uT.norm();
    if (nrm < min_norm_uT || nrm > max_norm_uT) return false;
    return buf.push(m_raw_uT, T(0));
  }

  bool fit(MagCalibration<T>& out, int robust_iters = 3, T trim_frac = T(0.15)) const {
    if (buf.n < 80) return false;

    // Degeneracy check radius estimate in µT space: median ||m - mean(m)||
    Vec3 mu = Vec3::Zero();
    for (int i = 0; i < buf.n; ++i) mu += buf.v[i];
    mu *= (T(1) / (T)buf.n);

    T rad_mu[IMU_CAL_MAX_SAMPLES];
    for (int i = 0; i < buf.n; ++i) rad_mu[i] = (buf.v[i] - mu).norm();
    T B_est = median_of_array(rad_mu, buf.n);

    auto fit0 = ellipsoid_to_sphere_robust<T>(
    buf.v, buf.n, T(1),
    robust_iters, trim_frac,
    T(1e-6), // ridge_rel (matches default)
    B_est);  // expected_radius_for_checks in µT space    
    if (!fit0.ok) return false;

    // Estimate field magnitude from median radius in raw µT space
    T radii[IMU_CAL_MAX_SAMPLES];
    int m = 0;
    for (int i = 0; i < buf.n; ++i) {
      radii[m++] = (buf.v[i] - fit0.b).norm();
    }
    T B_med = median_of_array(radii, m);

    // Typical Earth field is ~25-65 uT; allow broad
    if (B_med < T(12) || B_med > T(120)) return false;

    out.ok = true;
    out.b = fit0.b;
    out.A = fit0.A * B_med;     // map to µT-like magnitude
    out.field_uT = B_med;
    out.rms = fit0.rms * B_med;
    return true;
  }
};

// Calibrator: Gyroscope bias(T)
// Strategy:
//  - Collect stationary gyro samples while not moving.
//  - Bin by temperature and estimate mean bias per bin.
//  - Fit bias(T) via regression.
template <typename T, int N, int K_TBINS = 8>
struct GyroCalibrator {
  static_assert(N <= IMU_CAL_MAX_SAMPLES, "N exceeds IMU_CAL_MAX_SAMPLES (400)");
  static_assert(K_TBINS > 0 && K_TBINS <= 16, "K_TBINS unreasonable");

  using Vec3 = Eigen::Matrix<T,3,1>;
  SampleBuffer3<T, N> buf;

  T T0 = T(25);

  // stationary gate thresholds
  T max_gyro_norm = T(0.08);   // rad/s
  T max_accel_dev = T(1.0);    // m/s^2
  T g = T(9.80665);

  void clear() { buf.clear(); }

  bool addSample(const Vec3& w_raw, const Vec3& a_raw, T tempC) {
    if (!isfinite3(w_raw) || !isfinite3(a_raw) || !finiteT(tempC)) return false;
    if (w_raw.norm() > max_gyro_norm) return false;
    if ((T)fabs((double)(a_raw.norm() - g)) > max_accel_dev) return false;
    return buf.push(w_raw, tempC);
  }

  bool fit(GyroCalibration<T>& out) const {
    if (buf.n < 80) return false;

    // Temperature range
    T tmin = buf.tempC[0], tmax = buf.tempC[0];
    for (int i = 1; i < buf.n; ++i) {
      tmin = (buf.tempC[i] < tmin ? buf.tempC[i] : tmin);
      tmax = (buf.tempC[i] > tmax ? buf.tempC[i] : tmax);
    }
    const T trange = tmax - tmin;
    const T binW = (trange > T(1e-3)) ? (trange / (T)K_TBINS) : T(1);

    Eigen::Matrix<T,3,1> sumB[K_TBINS];
    T sumT[K_TBINS];
    int cnt[K_TBINS];
    for (int k = 0; k < K_TBINS; ++k) { sumB[k].setZero(); sumT[k]=T(0); cnt[k]=0; }

    for (int i = 0; i < buf.n; ++i) {
      int k = 0;
      if (trange > T(1e-3)) {
        k = (int)floor((double)((buf.tempC[i] - tmin) / binW));
        k = clamp<int>(k, 0, K_TBINS - 1);
      }
      sumB[k] += buf.v[i];
      sumT[k] += buf.tempC[i];
      cnt[k]  += 1;
    }

    Eigen::Matrix<T,3,1> bcenters[K_TBINS];
    T tcenters[K_TBINS];
    int nb = 0;
    for (int k = 0; k < K_TBINS; ++k) {
      if (cnt[k] < 20) continue;
      bcenters[nb] = sumB[k] / (T)cnt[k];
      tcenters[nb] = sumT[k] / (T)cnt[k];
      nb++;
    }
    if (nb < 1) return false;

    TempBias3<T> biasT;
    if (nb >= 2) {
      biasT = fit_temp_bias3<T, K_TBINS>(bcenters, tcenters, nb, T0);
    } else {
      biasT.ok = true;
      biasT.T0 = T0;
      biasT.b0 = bcenters[0];
      biasT.k.setZero();
    }

    out.ok = true;
    out.S = Eigen::Matrix<T,3,3>::Identity();
    out.biasT = biasT;
    return true;
  }
};

} // namespace imu_cal
