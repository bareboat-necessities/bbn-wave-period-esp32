#pragma once

/*

  Copyright 2026, Mikhail Grushinskiy

  AtomS3R reusable IMU calibration plumbing (NVS blob + CRC, runtime apply, axis mapping,
  and M5Unified-calibration clearing).

  This header is intentionally UI-agnostic: you can reuse it in:
    - the calibration wizard sketch
    - your "real" application firmware

  Example boot flow:

    #include <M5Unified.h>
    #include "AtomS3R_ImuCal.h"

    atoms3r_ical::ImuCalStoreNvs store;
    atoms3r_ical::ImuCalBlobV1   blob;
    atoms3r_ical::RuntimeCals    cals;

    void setup() {
      Serial.begin(115200);
      delay(150);
      Serial.println();

      auto cfg = M5.config();
      M5.begin(cfg);

      // 1) Clear M5Unified's own IMU calibration/offset data so it can't "stack"
      //    with our calibration (prevents two different calibrations colliding).
      atoms3r_ical::clearM5UnifiedImuCalibration();

      if (!M5.Imu.isEnabled()) {
        Serial.println("[BOOT] IMU not found");
        while (true) delay(100);
      }

      // 2) Load our calibration from NVS
      bool have = store.load(blob);

      if (have) {
        // 3) Display it at startup on Serial
        Serial.println("[BOOT] Found saved AtomS3R calibration:");
        atoms3r_ical::printBlobSummary(Serial, blob);
        atoms3r_ical::printBlobDetail(Serial, blob);

        // 4) Build runtime calibration objects for fast apply()
        cals.rebuildFromBlob(blob);
      } else {
        Serial.println("[BOOT] No saved AtomS3R calibration.");
        Serial.println("[BOOT] Starting calibration UI...");

        // 5) Start calibration UI / wizard
        // run_my_calibration_ui_and_save_blob(store);

        // After wizard saves:
        // store.load(blob); cals.rebuildFromBlob(blob);
      }
    }

    void loop() {
      atoms3r_ical::ImuSample s;
      if (!atoms3r_ical::readImuMapped(M5.Imu, s)) return;

      const auto a_cal = cals.applyAccel(s.a, s.tempC);
      const auto w_cal = cals.applyGyro (s.w, s.tempC);
      const auto m_cal = cals.applyMag  (s.m);

      // Use a_cal / w_cal / m_cal everywhere in the application
      // ...
    }

*/

#include <Arduino.h>
#include <M5Unified.h>
#include <Preferences.h>

#include <stdint.h>
#include <stddef.h>   // offsetof
#include <string.h>
#include <math.h>

#ifndef EIGEN_STACK_ALLOCATION_LIMIT
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#endif
#include <ArduinoEigenDense.h>

// Requires existing calibration types (imu_cal::AccelCalibration, etc.)
#include "CalibrateIMU.h"

namespace atoms3r_ical {

using Vector3f = Eigen::Matrix<float,3,1>;
using Matrix3f = Eigen::Matrix<float,3,3>;

// Config/constants (shared)
struct ImuCalCfg {
  static constexpr float g_std   = 9.80665f;
  static constexpr float DEG2RAD = 3.14159265358979323846f / 180.0f;
};

// Axis mapping (AtomS3R)
// acc_body = ( ay, ax, -az ) * g
// gyr_body = ( gy, gx, -gz ) * deg2rad
// mag_body = ( my, mx, -mz ) * (1/10)
static inline Vector3f map_acc_to_body_ned_(const m5::imu_3d_t& a_g) {
  return Vector3f(a_g.y * ImuCalCfg::g_std,
                  a_g.x * ImuCalCfg::g_std,
                 -a_g.z * ImuCalCfg::g_std);
}
static inline Vector3f map_gyr_to_body_ned_(const m5::imu_3d_t& w_deg_s) {
  return Vector3f(w_deg_s.y * ImuCalCfg::DEG2RAD,
                  w_deg_s.x * ImuCalCfg::DEG2RAD,
                 -w_deg_s.z * ImuCalCfg::DEG2RAD);
}
static inline Vector3f map_mag_to_body_uT_(const m5::imu_3d_t& m_raw) {
  return Vector3f(m_raw.y / 10.0f,
                  m_raw.x / 10.0f,
                 -m_raw.z / 10.0f);
}

// Blob + CRC utilities
struct ImuCalBlobV1 {
  static constexpr uint32_t IMU_CAL_MAGIC   = 0x434C554D; // 'MULC'
  static constexpr uint16_t IMU_CAL_VERSION = 1;

  uint32_t magic = IMU_CAL_MAGIC;
  uint16_t version = IMU_CAL_VERSION;
  uint16_t size_bytes = sizeof(ImuCalBlobV1);

  uint8_t  accel_ok = 0;
  float    accel_g = ImuCalCfg::g_std;
  float    accel_S[9]{};
  float    accel_T0 = 25.0f;
  float    accel_b0[3]{};
  float    accel_k[3]{};
  float    accel_rms_mag = 0.0f;

  uint8_t  gyro_ok = 0;
  float    gyro_T0 = 25.0f;
  float    gyro_b0[3]{};
  float    gyro_k[3]{};

  uint8_t  mag_ok = 0;
  float    mag_A[9]{};
  float    mag_b[3]{};
  float    mag_field_uT = 0.0f;
  float    mag_rms = 0.0f;

  uint32_t crc = 0;
};
static constexpr size_t IMU_CAL_CRC_LEN = offsetof(ImuCalBlobV1, crc);

static inline uint32_t crc32_ieee_(const uint8_t* data, size_t n) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) {
    crc ^= (uint32_t)data[i];
    for (int k = 0; k < 8; ++k) {
      uint32_t mask = -(crc & 1u);
      crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
  }
  return ~crc;
}

static inline Matrix3f mat_from_rowmajor9_(const float a[9]) {
  Matrix3f M;
  M(0,0)=a[0]; M(0,1)=a[1]; M(0,2)=a[2];
  M(1,0)=a[3]; M(1,1)=a[4]; M(1,2)=a[5];
  M(2,0)=a[6]; M(2,1)=a[7]; M(2,2)=a[8];
  return M;
}

static inline void mat_to_rowmajor9_(const Matrix3f& M, float a[9]) {
  a[0]=M(0,0); a[1]=M(0,1); a[2]=M(0,2);
  a[3]=M(1,0); a[4]=M(1,1); a[5]=M(1,2);
  a[6]=M(2,0); a[7]=M(2,1); a[8]=M(2,2);
}

static inline uint32_t computeBlobCrc(const ImuCalBlobV1& in) {
  ImuCalBlobV1 tmp = in;
  tmp.crc = 0;
  return crc32_ieee_((const uint8_t*)&tmp, IMU_CAL_CRC_LEN);
}

static inline bool validateBlob(const ImuCalBlobV1& b) {
  if (b.magic != ImuCalBlobV1::IMU_CAL_MAGIC) return false;
  if (b.version != ImuCalBlobV1::IMU_CAL_VERSION) return false;
  if (b.size_bytes != sizeof(ImuCalBlobV1)) return false;
  const uint32_t want = b.crc;
  return (computeBlobCrc(b) == want);
}

// NVS store (Preferences)
class ImuCalStoreNvs {
public:
  // Namespace/key kept stable so different sketches share the same saved cal.
  // If you want per-app separation, change these strings.
  static constexpr const char* kNamespace = "imu_cal";
  static constexpr const char* kKey       = "blob";

  bool load(ImuCalBlobV1& out) {
    Preferences prefs;
    prefs.begin(kNamespace, true);
    size_t n = prefs.getBytesLength(kKey);
    if (n != sizeof(ImuCalBlobV1)) { prefs.end(); return false; }

    ImuCalBlobV1 tmp;
    size_t got = prefs.getBytes(kKey, &tmp, sizeof(tmp));
    prefs.end();
    if (got != sizeof(tmp)) return false;

    if (!validateBlob(tmp)) return false;
    out = tmp;
    return true;
  }

  bool save(const ImuCalBlobV1& in) {
    ImuCalBlobV1 tmp = in;
    tmp.magic = ImuCalBlobV1::IMU_CAL_MAGIC;
    tmp.version = ImuCalBlobV1::IMU_CAL_VERSION;
    tmp.size_bytes = sizeof(ImuCalBlobV1);
    tmp.crc = 0;
    tmp.crc = computeBlobCrc(tmp);

    Preferences prefs;
    prefs.begin(kNamespace, false);
    size_t wrote = prefs.putBytes(kKey, &tmp, sizeof(tmp));
    prefs.end();
    return (wrote == sizeof(tmp));
  }

  void erase() {
    Preferences prefs;
    prefs.begin(kNamespace, false);
    prefs.remove(kKey);
    prefs.end();
  }
};

// Runtime calibration objects
struct RuntimeCals {
  imu_cal::AccelCalibration<float> acc{};
  imu_cal::GyroCalibration<float>  gyr{};
  imu_cal::MagCalibration<float>   mag{};

  void rebuildFromBlob(const ImuCalBlobV1& b) {
    acc.ok = (b.accel_ok != 0);
    acc.g  = b.accel_g;
    acc.S  = mat_from_rowmajor9_(b.accel_S);
    acc.biasT.ok = acc.ok;
    acc.biasT.T0 = b.accel_T0;
    acc.biasT.b0 = Vector3f(b.accel_b0[0], b.accel_b0[1], b.accel_b0[2]);
    acc.biasT.k  = Vector3f(b.accel_k[0],  b.accel_k[1],  b.accel_k[2]);
    acc.rms_mag  = b.accel_rms_mag;

    gyr.ok = (b.gyro_ok != 0);
    gyr.S  = Matrix3f::Identity();
    gyr.biasT.ok = gyr.ok;
    gyr.biasT.T0 = b.gyro_T0;
    gyr.biasT.b0 = Vector3f(b.gyro_b0[0], b.gyro_b0[1], b.gyro_b0[2]);
    gyr.biasT.k  = Vector3f(b.gyro_k[0],  b.gyro_k[1],  b.gyro_k[2]);

    mag.ok = (b.mag_ok != 0);
    mag.A  = mat_from_rowmajor9_(b.mag_A);
    mag.b  = Vector3f(b.mag_b[0], b.mag_b[1], b.mag_b[2]);
    mag.field_uT = b.mag_field_uT;
    mag.rms      = b.mag_rms;
  }

  Vector3f applyAccel(const Vector3f& a_raw, float tempC) const { return acc.ok ? acc.apply(a_raw, tempC) : a_raw; }
  Vector3f applyGyro (const Vector3f& w_raw, float tempC) const { return gyr.ok ? gyr.apply(w_raw, tempC) : w_raw; }
  Vector3f applyMag  (const Vector3f& m_raw) const { return mag.ok ? mag.apply(m_raw) : m_raw; }
};

// Pretty 3x3 print from row-major float[9].
static inline void printMat3RowMajor(Print& out, const float a[9], int prec = 9) {
  for (int r = 0; r < 3; ++r) {
    out.print("      [");
    for (int c = 0; c < 3; ++c) {
      out.print(a[3 * r + c], prec);
      if (c < 2) out.print(", ");
    }
    out.println("]");
  }
}

// Print quick diag + off-diagonal RMS, so you can instantly see "not identity".
static inline void printMatDiagOffDiagRms(Print& out, const float a[9]) {
  const float d0 = a[0], d1 = a[4], d2 = a[8];
  const float off2 =
      a[1]*a[1] + a[2]*a[2] +
      a[3]*a[3] + a[5]*a[5] +
      a[6]*a[6] + a[7]*a[7];
  const float off_rms = sqrtf(off2 / 6.0f);
  out.printf("      diag=[%.6f %.6f %.6f], offdiag_rms=%.6f\n", (double)d0, (double)d1, (double)d2, (double)off_rms);
}

// Identity matrix in row-major storage.
static inline const float* mat3_identity_rowmajor_() {
  static const float I[9] = {1,0,0, 0,1,0, 0,0,1};
  return I;
}

// Optional: print a matrix header line with a consistent style.
static inline void printMatHeader(Print& out, const char* name, const char* meaning) {
  out.print("    ");
  out.print(name);
  if (meaning && meaning[0]) {
    out.print(" (");
    out.print(meaning);
    out.print(")");
  }
  out.println(":");
}

// Print helpers (startup serial)
static inline void printBlobSummary(Print& out, const ImuCalBlobV1& b) {
  out.printf("  ok: A=%d G=%d M=%d\n", (int)b.accel_ok, (int)b.gyro_ok, (int)b.mag_ok);
}

static inline void printBlobDetail(Print& out, const ImuCalBlobV1& b) {
  // ACCEL 
  out.printf("  accel: g=%.6f T0=%.2f rms_mag=%.4f\n", (double)b.accel_g, (double)b.accel_T0, (double)b.accel_rms_mag);
  out.printf("    b0=[%.5f %.5f %.5f]\n", (double)b.accel_b0[0], (double)b.accel_b0[1], (double)b.accel_b0[2]);
  out.printf("    k =[%.6f %.6f %.6f]\n", (double)b.accel_k[0], (double)b.accel_k[1], (double)b.accel_k[2]);

  printMatHeader(out, "S", "a_cal = S*(a_raw - bias(T))");
  printMat3RowMajor(out, b.accel_S, 9);
  printMatDiagOffDiagRms(out, b.accel_S);

  // GYRO
  out.printf("  gyro:  T0=%.2f\n", (double)b.gyro_T0);
  out.printf("    b0=[%.6f %.6f %.6f]\n", (double)b.gyro_b0[0], (double)b.gyro_b0[1], (double)b.gyro_b0[2]);
  out.printf("    k =[%.6f %.6f %.6f]\n", (double)b.gyro_k[0], (double)b.gyro_k[1], (double)b.gyro_k[2]);

  // Gyro calibrator *always* sets S=I (stationary-only).
  printMatHeader(out, "S", "w_cal = S*(w_raw - bias(T)) ; S=I (stationary bias-only fit)");
  const float* I = mat3_identity_rowmajor_();
  printMat3RowMajor(out, I, 3);
  printMatDiagOffDiagRms(out, I);

  // MAG
  out.printf("  mag: field_uT=%.3f rms=%.4f\n", (double)b.mag_field_uT, (double)b.mag_rms);
  out.printf("    b=[%.3f %.3f %.3f]\n", (double)b.mag_b[0], (double)b.mag_b[1], (double)b.mag_b[2]);

  printMatHeader(out, "A", "m_cal = A*(m_raw - b)");
  printMat3RowMajor(out, b.mag_A, 9);
  printMatDiagOffDiagRms(out, b.mag_A);
}

// IMU sample + mapped read
struct ImuSample {
  Vector3f a;     // m/s^2 (mapped to body NED per AtomS3R mapping)
  Vector3f w;     // rad/s  (mapped to body NED)
  Vector3f m;     // uT     (mapped to body)
  float tempC;    // deg C
  uint32_t mask;  // M5.Imu.update() mask
};

// Reads M5.Imu, applies AtomS3R axis mapping and unit conversion, but does NOT calibrate.
static inline bool readImuMapped(decltype(M5.Imu)& imu, ImuSample& out) {
  out.mask = imu.update();
  if (!out.mask) return false;

  const auto data = imu.getImuData();
  out.tempC = NAN;
  (void)imu.getTemp(&out.tempC);

  out.a = map_acc_to_body_ned_(data.accel);
  out.w = map_gyr_to_body_ned_(data.gyro);

  // IMPORTANT: do not rely on a "mag valid" bit. Many builds never set it.
  out.m = map_mag_to_body_uT_(data.mag);

  return true;
}

// "No collisions" helper
// Clears *M5Unified's* internal calibration/offset data, so we can safely apply our own calibration
// (stored in ImuCalStoreNvs) without them stacking.
static inline void clearM5UnifiedImuCalibration() {
  // Clears runtime offsets and any stored "offset data" M5Unified may apply.
  M5.Imu.setCalibration(0, 0, 0);
  M5.Imu.clearOffsetData();
}

} // namespace atoms3r_ical
