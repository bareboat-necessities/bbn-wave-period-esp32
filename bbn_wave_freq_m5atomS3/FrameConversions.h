#pragma once

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
#endif

using Eigen::Vector3f;
using Eigen::Matrix3f;
using Eigen::Quaternionf;

// Coordinate conversions between:
//   Nautical Z-up (simulator, wave models)
//   Aerospace NED Z-down (Q-MEKF filter)
//
// Mapping: (x_a, y_a, z_a) = (y_n, x_n, -z_n)

// Vector conversions

static inline Vector3f zu_to_ned(const Vector3f& v) {
    return Vector3f(v.y(), v.x(), -v.z());
}

static inline Vector3f ned_to_zu(const Vector3f& v) {
    return Vector3f(v.y(), v.x(), -v.z());
}

// Euler angle conversions

static inline void aero_to_nautical(float &roll, float &pitch, float &yaw) {
    float r_a = roll;
    float p_a = pitch;
    roll  = -p_a;  // aerospace pitch → nautical roll
    pitch = -r_a;  // aerospace roll  → nautical pitch
    // yaw unchanged
}

static inline void nautical_to_aero(float &roll, float &pitch, float &yaw) {
    float r_n = roll;
    float p_n = pitch;
    roll  = -p_n;  // nautical pitch → aerospace roll
    pitch = -r_n;  // nautical roll  → aerospace pitch
    // yaw unchanged
}

// Quaternion helpers

// Build quaternion from Euler (deg, 3-2-1)
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

// Extract Euler (nautical convention, deg) directly from quaternion
static inline void quat_to_euler_nautical(const Quaternionf &q, float &roll, float &pitch, float &yaw) {
    quat_to_euler_aero(q, roll, pitch, yaw);
    aero_to_nautical(roll, pitch, yaw);
}

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

    // Quaternion consistency test
    float rq = 30, pq = 20, yq = 45;
    Quaternionf q_n = quat_from_euler(rq, pq, yq);

    // Aerospace → nautical via conversion
    float rqa = rq, pqa = pq, yqa = yq;
    nautical_to_aero(rqa, pqa, yqa);
    Quaternionf q_a = quat_from_euler(rqa, pqa, yqa);

    Quaternionf q_diff = q_n * q_a.inverse();
    float angle_diff = 2.0f * std::acos(std::clamp(q_diff.w(), -1.0f, 1.0f)) * 180.0f/M_PI;
    assert_close(angle_diff, 0.0f, 1e-2f, "Quaternion difference");

    // Quaternion → Euler (nautical) direct test
    float r_e, p_e, y_e;
    quat_to_euler_nautical(q_n, r_e, p_e, y_e);
    assert_close(r_e, rq, tol_angle, "quat_to_euler_nautical roll");
    assert_close(p_e, pq, tol_angle, "quat_to_euler_nautical pitch");
    assert_close(y_e, yq, tol_angle, "quat_to_euler_nautical yaw");

    std::cout << "All frame conversion tests passed\n";
    return 0;
}
#endif // FRAMECONV_TEST
