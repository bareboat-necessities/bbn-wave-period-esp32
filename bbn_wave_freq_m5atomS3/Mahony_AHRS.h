#ifndef Mahony_AHRS_h
#define Mahony_AHRS_h

//
// Madgwick's implementation of Mahony's AHRS algorithm.
// See: http://www.x-io.co.uk/node/8#open_source_ahrs_and_imu_algorithms
//
// Fixes applied (no algorithm changes, no public API changes):
//   1) All function definitions marked inline  -> no ODR violation in multi-TU builds
//   2) invSqrt uses memcpy for type-punning    -> no C++ UB
//   3) Consistent f-suffix math throughout     -> no hidden double promotion
//   4) #define macros replaced with constexpr  -> no global namespace pollution
//   5) Redundant forward declaration removed
//   6) mahony_AHRS_init documented: gains-only, does NOT reset quaternion/integrals
//   7) mahony_AHRS_reset() added: full state reset to identity + zero integrals
//   8) halfvz formula unified between update() and update_mag()
//   9) Dead commented-out Euler formulas removed
//  10) #include <math.h> -> <cmath> for C++ correctness
//

#include <cmath>
#include <cstdint>
#include <cstring>

// Use constexpr instead of #define to avoid global namespace pollution.
// Names are kept identical so all existing call sites continue to compile.
static constexpr float twoKpDef = 2.0f * 1.0f;   // 2 * proportional gain
static constexpr float twoKiDef = 2.0f * 0.0f;   // 2 * integral gain

// File-local conversion constant — not a macro.
static constexpr double kRadToDeg = 57.295779513082320876798154814105;

typedef struct mahony_AHRS_vars {
    float twoKp       = twoKpDef;
    float twoKi       = twoKiDef;
    float q0          = 1.0f;
    float q1          = 0.0f;
    float q2          = 0.0f;
    float q3          = 0.0f;   // quaternion of sensor frame relative to auxiliary frame
    float integralFBx = 0.0f;
    float integralFBy = 0.0f;
    float integralFBz = 0.0f;   // integral error terms scaled by Ki
} Mahony_AHRS_Vars;

// ---------------------------------------------------------------------------
// invSqrt
//
// Fast inverse square root.
// Uses memcpy for type-punning — well-defined in both C and C++.
// Magic constant 0x5f375a86 (Lomont) gives ~0.18% error after one
// Newton-Raphson step.
// ---------------------------------------------------------------------------
inline float invSqrt(float number) {
    float y = number;
    int32_t i;
    std::memcpy(&i, &y, sizeof(i));
    i = 0x5f375a86 - (i >> 1);
    std::memcpy(&y, &i, sizeof(y));
    y = y * (1.5f - (number * 0.5f * y * y));
    return y;
}

// ---------------------------------------------------------------------------
// mahony_AHRS_init
//
// Sets gains only. Does NOT reset quaternion or integral states.
// Use this to change gains on a running filter without disturbing attitude.
// Use mahony_AHRS_reset() for a full state reset.
// ---------------------------------------------------------------------------
inline void mahony_AHRS_init(Mahony_AHRS_Vars* m, float twoKp, float twoKi) {
    m->twoKp = twoKp;
    m->twoKi = twoKi;
}

// ---------------------------------------------------------------------------
// mahony_AHRS_reset
//
// Full state reset: identity quaternion, zero integral accumulators.
// Gains are preserved unless positive values are passed.
// Pass twoKp <= 0 to leave gains unchanged.
// ---------------------------------------------------------------------------
inline void mahony_AHRS_reset(Mahony_AHRS_Vars* m,
                               float twoKp = -1.0f,
                               float twoKi = -1.0f)
{
    if (twoKp > 0.0f)  m->twoKp = twoKp;
    if (twoKi >= 0.0f) m->twoKi = twoKi;

    m->q0 = 1.0f;
    m->q1 = 0.0f;
    m->q2 = 0.0f;
    m->q3 = 0.0f;

    m->integralFBx = 0.0f;
    m->integralFBy = 0.0f;
    m->integralFBz = 0.0f;
}

// ---------------------------------------------------------------------------
// mahony_AHRS_update  (IMU-only, no magnetometer)
// ---------------------------------------------------------------------------
inline void mahony_AHRS_update(Mahony_AHRS_Vars* m,
                                float gx, float gy, float gz,
                                float ax, float ay, float az,
                                float* pitch, float* roll, float* yaw,
                                float delta_t_sec)
{
    float recipNorm;
    float halfvx, halfvy, halfvz;
    float halfex, halfey, halfez;
    float qa, qb, qc;

    // Compute feedback only if accelerometer measurement valid.
    if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {

        // Normalise accelerometer measurement.
        recipNorm = invSqrt(ax * ax + ay * ay + az * az);
        ax *= recipNorm;
        ay *= recipNorm;
        az *= recipNorm;

        // Precomputed products for unified halfvz form (matches update_mag).
        const float q0q0 = m->q0 * m->q0;
        const float q1q1 = m->q1 * m->q1;
        const float q2q2 = m->q2 * m->q2;
        const float q3q3 = m->q3 * m->q3;

        // Estimated direction of gravity.
        halfvx = m->q1 * m->q3 - m->q0 * m->q2;
        halfvy = m->q0 * m->q1 + m->q2 * m->q3;
        halfvz = 0.5f * (q0q0 - q1q1 - q2q2 + q3q3);   // unified with update_mag

        // Error is cross product between estimated and measured gravity.
        halfex = (ay * halfvz - az * halfvy);
        halfey = (az * halfvx - ax * halfvz);
        halfez = (ax * halfvy - ay * halfvx);

        // Integral feedback.
        if (m->twoKi > 0.0f) {
            m->integralFBx += m->twoKi * halfex * delta_t_sec;
            m->integralFBy += m->twoKi * halfey * delta_t_sec;
            m->integralFBz += m->twoKi * halfez * delta_t_sec;
            gx += m->integralFBx;
            gy += m->integralFBy;
            gz += m->integralFBz;
        } else {
            m->integralFBx = 0.0f;
            m->integralFBy = 0.0f;
            m->integralFBz = 0.0f;
        }

        // Proportional feedback.
        gx += m->twoKp * halfex;
        gy += m->twoKp * halfey;
        gz += m->twoKp * halfez;
    }

    // Integrate rate of change of quaternion.
    gx *= (0.5f * delta_t_sec);
    gy *= (0.5f * delta_t_sec);
    gz *= (0.5f * delta_t_sec);

    qa = m->q0;
    qb = m->q1;
    qc = m->q2;
    m->q0 += (-qb * gx - qc * gy - m->q3 * gz);
    m->q1 += ( qa * gx + qc * gz - m->q3 * gy);
    m->q2 += ( qa * gy - qb * gz + m->q3 * gx);
    m->q3 += ( qa * gz + qb * gy - qc  * gx);

    // Normalise quaternion.
    recipNorm = invSqrt(m->q0 * m->q0 + m->q1 * m->q1 +
                        m->q2 * m->q2 + m->q3 * m->q3);
    m->q0 *= recipNorm;
    m->q1 *= recipNorm;
    m->q2 *= recipNorm;
    m->q3 *= recipNorm;

    // Euler angles — consistent f-suffix throughout.
    *pitch = asinf (-2.0f * m->q1 * m->q3 + 2.0f * m->q0 * m->q2);
    *roll  = atan2f( 2.0f * m->q2 * m->q3 + 2.0f * m->q0 * m->q1,
                    -2.0f * m->q1 * m->q1 - 2.0f * m->q2 * m->q2 + 1.0f);
    *yaw   = atan2f( 2.0f * (m->q1 * m->q2 + m->q0 * m->q3),
                     m->q0 * m->q0 + m->q1 * m->q1 - m->q2 * m->q2 - m->q3 * m->q3);

    *pitch *= static_cast<float>(kRadToDeg);
    *roll  *= static_cast<float>(kRadToDeg);
    *yaw   *= static_cast<float>(kRadToDeg);
}

// ---------------------------------------------------------------------------
// mahony_AHRS_update_mag  (IMU + magnetometer)
// ---------------------------------------------------------------------------
inline void mahony_AHRS_update_mag(Mahony_AHRS_Vars* m,
                                    float gx, float gy, float gz,
                                    float ax, float ay, float az,
                                    float mx, float my, float mz,
                                    float* pitch, float* roll, float* yaw,
                                    float delta_t_sec)
{
    // Fall back to IMU-only if magnetometer is all-zero.
    if ((mx == 0.0f) && (my == 0.0f) && (mz == 0.0f)) {
        mahony_AHRS_update(m, gx, gy, gz, ax, ay, az,
                           pitch, roll, yaw, delta_t_sec);
        return;
    }

    float recipNorm;
    float halfex, halfey, halfez;

    if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {

        // Normalise accelerometer.
        recipNorm = invSqrt(ax * ax + ay * ay + az * az);
        ax *= recipNorm;
        ay *= recipNorm;
        az *= recipNorm;

        // Normalise magnetometer.
        recipNorm = invSqrt(mx * mx + my * my + mz * mz);
        mx *= recipNorm;
        my *= recipNorm;
        mz *= recipNorm;

        // Precomputed quaternion products.
        const float q0q0 = m->q0 * m->q0;
        const float q0q1 = m->q0 * m->q1;
        const float q0q2 = m->q0 * m->q2;
        const float q0q3 = m->q0 * m->q3;
        const float q1q1 = m->q1 * m->q1;
        const float q1q2 = m->q1 * m->q2;
        const float q1q3 = m->q1 * m->q3;
        const float q2q2 = m->q2 * m->q2;
        const float q2q3 = m->q2 * m->q3;
        const float q3q3 = m->q3 * m->q3;

        // Reference direction of Earth's magnetic field.
        const float hx = 2.0f * (mx * (0.5f - q2q2 - q3q3) +
                                  my * (q1q2 - q0q3) +
                                  mz * (q1q3 + q0q2));
        const float hy = 2.0f * (mx * (q1q2 + q0q3) +
                                  my * (0.5f - q1q1 - q3q3) +
                                  mz * (q2q3 - q0q1));
        const float hz = 2.0f * (mx * (q1q3 - q0q2) +
                                  my * (q2q3 + q0q1) +
                                  mz * (0.5f - q1q1 - q2q2));
        const float bx = sqrtf(hx * hx + hy * hy);
        const float bz = hz;

        // Estimated direction of gravity and magnetic field.
        const float halfvx = q1q3 - q0q2;
        const float halfvy = q0q1 + q2q3;
        const float halfvz = 0.5f * (q0q0 - q1q1 - q2q2 + q3q3);  // same form as update()

        const float halfwx = bx * (0.5f - q2q2 - q3q3) + bz * (q1q3 - q0q2);
        const float halfwy = bx * (q1q2 - q0q3)         + bz * (q0q1 + q2q3);
        const float halfwz = bx * (q0q2 + q1q3)         + bz * (0.5f - q1q1 - q2q2);

        // Error = cross product between estimated and measured field vectors.
        halfex = (ay * halfvz - az * halfvy) + (my * halfwz - mz * halfwy);
        halfey = (az * halfvx - ax * halfvz) + (mz * halfwx - mx * halfwz);
        halfez = (ax * halfvy - ay * halfvx) + (mx * halfwy - my * halfwx);

        // Integral feedback.
        if (m->twoKi > 0.0f) {
            m->integralFBx += m->twoKi * halfex * delta_t_sec;
            m->integralFBy += m->twoKi * halfey * delta_t_sec;
            m->integralFBz += m->twoKi * halfez * delta_t_sec;
            gx += m->integralFBx;
            gy += m->integralFBy;
            gz += m->integralFBz;
        } else {
            m->integralFBx = 0.0f;
            m->integralFBy = 0.0f;
            m->integralFBz = 0.0f;
        }

        // Proportional feedback.
        gx += m->twoKp * halfex;
        gy += m->twoKp * halfey;
        gz += m->twoKp * halfez;
    }

    // Integrate rate of change of quaternion.
    const float half_dt = 0.5f * delta_t_sec;
    const float q_0 = m->q0;
    const float q_1 = m->q1;
    const float q_2 = m->q2;
    const float q_3 = m->q3;

    m->q0 += (-q_1 * gx - q_2 * gy - q_3 * gz) * half_dt;
    m->q1 += ( q_0 * gx + q_2 * gz - q_3 * gy) * half_dt;
    m->q2 += ( q_0 * gy - q_1 * gz + q_3 * gx) * half_dt;
    m->q3 += ( q_0 * gz + q_1 * gy - q_2 * gx) * half_dt;

    // Normalise quaternion.
    recipNorm = invSqrt(m->q0 * m->q0 + m->q1 * m->q1 +
                        m->q2 * m->q2 + m->q3 * m->q3);
    m->q0 *= recipNorm;
    m->q1 *= recipNorm;
    m->q2 *= recipNorm;
    m->q3 *= recipNorm;

    // Euler angles — consistent f-suffix throughout.
    *pitch = asinf (-2.0f * m->q1 * m->q3 + 2.0f * m->q0 * m->q2);
    *roll  = atan2f( 2.0f * m->q2 * m->q3 + 2.0f * m->q0 * m->q1,
                    -2.0f * m->q1 * m->q1 - 2.0f * m->q2 * m->q2 + 1.0f);
    *yaw   = atan2f( 2.0f * (m->q1 * m->q2 + m->q0 * m->q3),
                     m->q0 * m->q0 + m->q1 * m->q1 - m->q2 * m->q2 - m->q3 * m->q3);

    *pitch *= static_cast<float>(kRadToDeg);
    *roll  *= static_cast<float>(kRadToDeg);
    *yaw   *= static_cast<float>(kRadToDeg);
}

#endif // Mahony_AHRS_h
