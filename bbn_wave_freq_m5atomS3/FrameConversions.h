#pragma once

/*
   Copyright 2025, Mikhail Grushinskiy
 */

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif

#include <cmath>

#ifdef FRAMECONV_TEST
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <algorithm>   // for std::clamp
#endif

using Eigen::Vector3f;
using Eigen::Matrix3f;
using Eigen::Quaternionf;

// Coordinate conversions
//   Nautical Z-up (ENU: East, North, Up)  ← simulator, wave models
//   Aerospace NED (North, East, Down)     ← Q-MEKF filter
//
// Mapping: (x_a, y_a, z_a) = (y_n, x_n, -z_n)

// Nautical (ENU, Z-up) → Aerospace (NED, Z-down)
static inline Vector3f zu_to_ned(const Vector3f& v) {
    return Vector3f(v.y(), v.x(), -v.z());
}

// Aerospace (NED, Z-down) → Nautical (ENU, Z-up)
static inline Vector3f ned_to_zu(const Vector3f& v) {
    return Vector3f(v.y(), v.x(), -v.z());
}

// Euler angle conversions (roll, pitch, yaw in degrees)

// Aerospace → Nautical
static inline void aero_to_nautical(float &roll, float &pitch, float &yaw) {
    float r_a = roll;
    float p_a = pitch;
    roll  = -p_a;  // aerospace pitch → nautical roll
    pitch = -r_a;  // aerospace roll  → nautical pitch
    // yaw unchanged
}

// Nautical → Aerospace
static inline void nautical_to_aero(float &roll, float &pitch, float &yaw) {
    float r_n = roll;
    float p_n = pitch;
    roll  = -p_n;  // nautical pitch → aerospace roll
    pitch = -r_n;  // nautical roll  → aerospace pitch
    // yaw unchanged
}

// Quaternion helpers

// Build quaternion from Euler (deg, aerospace convention ZYX: yaw→pitch→roll)
static Quaternionf quat_from_euler(float roll_deg, float pitch_deg, float yaw_deg) {
    float cr = std::cos(roll_deg * M_PI/180.0f / 2.0f);
    float sr = std::sin(roll_deg * M_PI/180.0f / 2.0f);
    float cp = std::cos(pitch_deg * M_PI/180.0f / 2.0f);
    float sp = std::sin(pitch_deg * M_PI/180.0f / 2.0f);
    float cy = std::cos(yaw_deg * M_PI/180.0f / 2.0f);
    float sy = std::sin(yaw_deg * M_PI/180.0f / 2.0f);

    Quaternionf q;
    q.w() = cy*cp*cr + sy*sp*sr;
    q.x() = cy*cp*sr - sy*sp*cr;
    q.y() = sy*cp*sr + cy*sp*cr;
    q.z() = sy*cp*cr - cy*sp*sr;
    return q.normalized();
}

// Extract Euler (aerospace convention, deg) from quaternion
static inline void quat_to_euler_aero(const Quaternionf &q, float &roll, float &pitch, float &yaw) {
    Matrix3f R = q.toRotationMatrix();

    pitch = std::atan2(-R(2,0), std::sqrt(R(0,0)*R(0,0) + R(1,0)*R(1,0)));
    roll  = std::atan2(R(2,1), R(2,2));
    yaw   = std::atan2(R(1,0), R(0,0));

    roll  *= 180.0f / M_PI;   
    pitch *= 180.0f / M_PI;   
    yaw   *= 180.0f / M_PI;  
}

// Extract Euler (nautical convention, deg) from quaternion
static inline void quat_to_euler_nautical(const Quaternionf &q, float &roll, float &pitch, float &yaw) {
    quat_to_euler_aero(q, roll, pitch, yaw);
    aero_to_nautical(roll, pitch, yaw);
}

// Magnetometer simulation using World Magnetic Model (WMM)
// Defaults: Statue of Liberty, USA, Sept 2025, elevation ~0 m

struct MagSim_WMM {
    static constexpr float default_declination_deg = -3.44f; // [deg] east is positive
    static constexpr float default_inclination_deg =  67.5f; // [deg] positive = down
    static constexpr float default_total_field_uT  = 51.0f;  // [µT]

    // World magnetic field in ENU (East, North, Up)
    // East = X, North = Y, Up = Z
    // Units: microteslas [µT]
    static Eigen::Vector3f mag_world_nautical(
        float declination_deg = default_declination_deg,
        float inclination_deg = default_inclination_deg,
        float total_field_uT  = default_total_field_uT)
    {
        float dec_rad  = declination_deg * M_PI / 180.0f;
        float incl_rad = inclination_deg * M_PI / 180.0f;

        float h = std::cos(incl_rad);  // horizontal fraction
        float v = -std::sin(incl_rad); // vertical (downwards, Z-up frame)

        Eigen::Vector3f mag_world;
        mag_world.x() = h * std::sin(dec_rad); // East
        mag_world.y() = h * std::cos(dec_rad); // North
        mag_world.z() = v;                     // Up

        return mag_world * total_field_uT;     // [µT]
    }

    // World magnetic field vector in Aerospace NED frame (North, East, Down)
    static Eigen::Vector3f mag_world_aero(
        float declination_deg = default_declination_deg,
        float inclination_deg = default_inclination_deg,
        float total_field_uT  = default_total_field_uT)
    {
        return zu_to_ned(mag_world_nautical(declination_deg, inclination_deg, total_field_uT));
    }

// Simulate body-frame magnetometer [µT] from nautical Euler (deg)
// Input: nautical Euler (ENU, Z-up)
// Output: body-frame magnetometer in nautical frame (body ENU)
static Eigen::Vector3f simulate_mag_from_euler_nautical(
    float roll_deg, float pitch_deg, float yaw_deg,
    float declination_deg = default_declination_deg,
    float inclination_deg = default_inclination_deg,
    float total_field_uT  = default_total_field_uT)
{
    // 1) Convert nautical → aerospace (NED)
    float r_a = roll_deg, p_a = pitch_deg, y_a = yaw_deg;
    nautical_to_aero(r_a, p_a, y_a);

    // 2) Aerospace quaternion (ZYX, NED)
    Quaternionf q_a = quat_from_euler(r_a, p_a, y_a);

    // 3) World mag in aerospace (NED)
    Eigen::Vector3f mag_world_a = mag_world_aero(declination_deg, inclination_deg, total_field_uT);

    // 4) Rotate into body-frame (NED)
    Eigen::Vector3f mag_body_a = q_a.inverse() * mag_world_a;

    // 5) Convert body NED → body ENU for return
    return ned_to_zu(mag_body_a);
}

// Simulate body-frame magnetometer [µT] from aerospace Euler (deg)
// Input: aerospace Euler (NED)
// Output: body-frame magnetometer in nautical frame (body ENU)
static Eigen::Vector3f simulate_mag_from_euler_aero(
    float roll_deg, float pitch_deg, float yaw_deg,
    float declination_deg = default_declination_deg,
    float inclination_deg = default_inclination_deg,
    float total_field_uT  = default_total_field_uT)
{
    // 1) Aerospace quaternion (ZYX, NED)
    Quaternionf q_a = quat_from_euler(roll_deg, pitch_deg, yaw_deg);

    // 2) World mag in aerospace (NED)
    Eigen::Vector3f mag_world_a = mag_world_aero(declination_deg, inclination_deg, total_field_uT);

    // 3) Rotate into body-frame (NED)
    Eigen::Vector3f mag_body_a = q_a.inverse() * mag_world_a;

    // 4) Convert body NED → body ENU for return
    return ned_to_zu(mag_body_a);
}

};

#ifdef FRAMECONV_TEST

// Utility: floating-point assert with tolerance
inline void assert_close(float a, float b, float tol, const char* msg) {
    if (std::fabs(a - b) > tol) {
        std::cerr << "ASSERT FAIL: " << msg
                  << "  got=" << a << " expected=" << b
                  << " tol=" << tol << "\n";
        assert(false);
    }
}

// Test entry point
inline int test_frame_conversions() {
    const float tol_angle = 1e-3f;   // tolerance for Euler round-trips
    const float tol_vec   = 1e-6f;   // tolerance for vector round-trips

    float r_n, p_n, y_n;
    float r_a, p_a, y_a;

    // Flat
    r_n = 0; p_n = 0; y_n = 0;
    r_a = r_n; p_a = p_n; y_a = y_n;
    nautical_to_aero(r_a, p_a, y_a);
    aero_to_nautical(r_a, p_a, y_a);
    assert_close(r_a, r_n, tol_angle, "Flat roll");
    assert_close(p_a, p_n, tol_angle, "Flat pitch");
    assert_close(y_a, y_n, tol_angle, "Flat yaw");

    // 90° roll
    r_n = 90; p_n = 0; y_n = 0;
    r_a = r_n; p_a = p_n; y_a = y_n;
    nautical_to_aero(r_a, p_a, y_a);
    aero_to_nautical(r_a, p_a, y_a);
    assert_close(r_a, r_n, tol_angle, "90 roll");

    // 90° pitch
    r_n = 0; p_n = 90; y_n = 0;
    r_a = r_n; p_a = p_n; y_a = y_n;
    nautical_to_aero(r_a, p_a, y_a);
    aero_to_nautical(r_a, p_a, y_a);
    assert_close(p_a, p_n, tol_angle, "90 pitch");

    // 90° yaw
    r_n = 0; p_n = 0; y_n = 90;
    r_a = r_n; p_a = p_n; y_a = y_n;
    nautical_to_aero(r_a, p_a, y_a);
    aero_to_nautical(r_a, p_a, y_a);
    assert_close(y_a, y_n, tol_angle, "90 yaw");

    // General case
    r_n = 30; p_n = 20; y_n = 45;
    r_a = r_n; p_a = p_n; y_a = y_n;
    nautical_to_aero(r_a, p_a, y_a);
    aero_to_nautical(r_a, p_a, y_a);
    assert_close(r_a, r_n, tol_angle, "General roll");
    assert_close(p_a, p_n, tol_angle, "General pitch");
    assert_close(y_a, y_n, tol_angle, "General yaw");

    // Gravity vector round-trip
    Vector3f g_n(0, 0, -9.81f);
    Vector3f g_a = zu_to_ned(g_n);
    Vector3f g_back = ned_to_zu(g_a);
    assert_close(g_back.x(), g_n.x(), tol_vec, "Gravity x");
    assert_close(g_back.y(), g_n.y(), tol_vec, "Gravity y");
    assert_close(g_back.z(), g_n.z(), tol_vec, "Gravity z");

    // Magnetometer vector round-trip
    Vector3f m_n(1.0f, 0.0f, 0.0f);
    Vector3f m_a = zu_to_ned(m_n);
    Vector3f m_back = ned_to_zu(m_a);
    assert_close(m_back.x(), m_n.x(), tol_vec, "Mag x");
    assert_close(m_back.y(), m_n.y(), tol_vec, "Mag y");
    assert_close(m_back.z(), m_n.z(), tol_vec, "Mag z");

    // Quaternion consistency test
    float rq = 30, pq = 20, yq = 45;
    Quaternionf q_n = quat_from_euler(rq, pq, yq);

    float rqa = rq, pqa = pq, yqa = yq;
    nautical_to_aero(rqa, pqa, yqa);
    Quaternionf q_a = quat_from_euler(rqa, pqa, yqa);

    Quaternionf q_diff = q_n * q_a.inverse();
    float angle_diff = 2.0f * std::acos(std::clamp(q_diff.w(), -1.0f, 1.0f)) * 180.0f/M_PI;
    assert_close(angle_diff, 0.0f, 1e-2f, "Quaternion difference");

    // Randomized stress test
    for (int i = 0; i < 100; ++i) {
        float rn = (float(rand()) / RAND_MAX - 0.5f) * 180.0f;  // [-90,90]
        float pn = (float(rand()) / RAND_MAX - 0.5f) * 180.0f;
        float yn = (float(rand()) / RAND_MAX) * 360.0f;         // [0,360]

        float ra = rn, pa = pn, ya = yn;
        nautical_to_aero(ra, pa, ya);
        aero_to_nautical(ra, pa, ya);

        assert_close(ra, rn, tol_angle, "Random roll");
        assert_close(pa, pn, tol_angle, "Random pitch");
        assert_close(ya, yn, tol_angle, "Random yaw");
    }

    // Quaternion → Euler (nautical) direct test
    float rq2 = 30, pq2 = 20, yq2 = 45;
    Quaternionf q_n2 = quat_from_euler(rq2, pq2, yq2);
    float r_e, p_e, y_e;
    quat_to_euler_nautical(q_n2, r_e, p_e, y_e);
    assert_close(r_e, rq2, tol_angle, "quat_to_euler_nautical roll");
    assert_close(p_e, pq2, tol_angle, "quat_to_euler_nautical pitch");
    assert_close(y_e, yq2, tol_angle, "quat_to_euler_nautical yaw");

    // Magnetometer world-field tests
    Vector3f mag_enu = MagSim_WMM::mag_world_nautical();
    Vector3f mag_ned = MagSim_WMM::mag_world_aero();
    assert_close(mag_enu.z(), -std::sin(MagSim_WMM::default_inclination_deg * M_PI/180.0f) * MagSim_WMM::default_total_field_uT,
                 1e-3f, "Mag ENU vertical");
    assert_close(mag_ned.z(), std::sin(MagSim_WMM::default_inclination_deg * M_PI/180.0f) * MagSim_WMM::default_total_field_uT,
                 1e-3f, "Mag NED vertical");

    // Magnetometer body-frame test at zero Euler (should match world ENU)
    Vector3f mag_body0 = MagSim_WMM::simulate_mag_from_euler_nautical(0,0,0);
    assert_close(mag_body0.x(), mag_enu.x(), 1e-3f, "Mag body0 east");
    assert_close(mag_body0.y(), mag_enu.y(), 1e-3f, "Mag body0 north");
    assert_close(mag_body0.z(), mag_enu.z(), 1e-3f, "Mag body0 up");

    // Rotate yaw by +90° in nautical convention
    Vector3f mag_body_yaw90 = MagSim_WMM::simulate_mag_from_euler_nautical(0,0,90);
    // Expect: world east→body forward (x), north→body left (–y)
    float norm0 = mag_body0.head<2>().norm();
    float norm90 = mag_body_yaw90.head<2>().norm();
    assert_close(norm0, norm90, 1e-3f, "Mag yaw rotation preserves horizontal norm");

    // Rotate yaw by 180°: should flip horizontal vector
    Vector3f mag_body_yaw180 = MagSim_WMM::simulate_mag_from_euler_nautical(0,0,180);
    assert_close(mag_body_yaw180.x(), -mag_body0.x(), 1e-3f, "Mag yaw180 east");
    assert_close(mag_body_yaw180.y(), -mag_body0.y(), 1e-3f, "Mag yaw180 north");
    
    std::cout << "All frame conversion tests passed\n";
    return 0;
}
#endif // FRAMECONV_TEST
