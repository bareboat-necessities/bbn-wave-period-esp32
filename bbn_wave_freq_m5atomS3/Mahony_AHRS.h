#ifndef Mahony_AHRS_h
#define Mahony_AHRS_h

//
// Madgwick's implementation of Mahony's AHRS algorithm.
// See: http://www.x-io.co.uk/node/8#open_source_ahrs_and_imu_algorithms
//

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

template<typename T = float>
class Mahony_AHRS {
    static_assert(std::is_floating_point<T>::value,
                  "Mahony_AHRS<T>: T must be a floating-point type.");

public:
    static constexpr T twoKpDef = T(2) * T(1);   // 2 * proportional gain
    static constexpr T twoKiDef = T(2) * T(0);   // 2 * integral gain
    static constexpr T kRadToDeg =
        T(57.295779513082320876798154814105L);

    T twoKp       = twoKpDef;
    T twoKi       = twoKiDef;
    T q0          = T(1);
    T q1          = T(0);
    T q2          = T(0);
    T q3          = T(0);   // quaternion of sensor frame relative to auxiliary frame
    T integralFBx = T(0);
    T integralFBy = T(0);
    T integralFBz = T(0);   // integral error terms scaled by Ki

    // Fast inverse square root.
    // Uses memcpy for type-punning — well-defined in both C and C++.
    // Magic constant 0x5f375a86 (Lomont) gives ~0.18% error after one
    // Newton-Raphson step for float. Other types fall back to std::sqrt.
    static T invSqrt(T number) {
        if constexpr (std::is_same<T, float>::value) {
            float y = number;
            int32_t i;
            std::memcpy(&i, &y, sizeof(i));
            i = 0x5f375a86 - (i >> 1);
            std::memcpy(&y, &i, sizeof(y));
            y = y * (1.5f - (number * 0.5f * y * y));
            return y;
        } else {
            return T(1) / std::sqrt(number);
        }
    }

    // Sets gains only. Does NOT reset quaternion or integral states.
    // Use this to change gains on a running filter without disturbing attitude.
    // Use reset() for a full state reset.
    void init(T twoKp_in, T twoKi_in) {
        twoKp = twoKp_in;
        twoKi = twoKi_in;
    }

    // Full state reset: identity quaternion, zero integral accumulators.
    // Gains are preserved unless positive values are passed.
    // Pass twoKp_in <= 0 to leave gains unchanged.
    void reset(T twoKp_in = T(-1), T twoKi_in = T(-1)) {
        if (twoKp_in > T(0))  twoKp = twoKp_in;
        if (twoKi_in >= T(0)) twoKi = twoKi_in;

        q0 = T(1);
        q1 = T(0);
        q2 = T(0);
        q3 = T(0);

        integralFBx = T(0);
        integralFBy = T(0);
        integralFBz = T(0);
    }

    // update()  (IMU-only, no magnetometer)
    void update(T gx, T gy, T gz,
                T ax, T ay, T az,
                T* pitch, T* roll, T* yaw,
                T delta_t_sec)
    {
        T recipNorm;
        T halfvx, halfvy, halfvz;
        T halfex, halfey, halfez;
        T qa, qb, qc;

        // Compute feedback only if accelerometer measurement valid.
        if (!((ax == T(0)) && (ay == T(0)) && (az == T(0)))) {

            // Normalise accelerometer measurement.
            recipNorm = invSqrt(ax * ax + ay * ay + az * az);
            ax *= recipNorm;
            ay *= recipNorm;
            az *= recipNorm;

            // Precomputed products for unified halfvz form (matches updateMag).
            const T q0q0 = q0 * q0;
            const T q1q1 = q1 * q1;
            const T q2q2 = q2 * q2;
            const T q3q3 = q3 * q3;

            // Estimated direction of gravity.
            halfvx = q1 * q3 - q0 * q2;
            halfvy = q0 * q1 + q2 * q3;
            halfvz = T(0.5) * (q0q0 - q1q1 - q2q2 + q3q3);   // unified with updateMag

            // Error is cross product between estimated and measured gravity.
            halfex = (ay * halfvz - az * halfvy);
            halfey = (az * halfvx - ax * halfvz);
            halfez = (ax * halfvy - ay * halfvx);

            // Integral feedback.
            if (twoKi > T(0)) {
                integralFBx += twoKi * halfex * delta_t_sec;
                integralFBy += twoKi * halfey * delta_t_sec;
                integralFBz += twoKi * halfez * delta_t_sec;
                gx += integralFBx;
                gy += integralFBy;
                gz += integralFBz;
            } else {
                integralFBx = T(0);
                integralFBy = T(0);
                integralFBz = T(0);
            }

            // Proportional feedback.
            gx += twoKp * halfex;
            gy += twoKp * halfey;
            gz += twoKp * halfez;
        }

        // Integrate rate of change of quaternion.
        gx *= (T(0.5) * delta_t_sec);
        gy *= (T(0.5) * delta_t_sec);
        gz *= (T(0.5) * delta_t_sec);

        qa = q0;
        qb = q1;
        qc = q2;
        q0 += (-qb * gx - qc * gy - q3 * gz);
        q1 += ( qa * gx + qc * gz - q3 * gy);
        q2 += ( qa * gy - qb * gz + q3 * gx);
        q3 += ( qa * gz + qb * gy - qc * gx);

        // Normalise quaternion.
        recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        q0 *= recipNorm;
        q1 *= recipNorm;
        q2 *= recipNorm;
        q3 *= recipNorm;

        // Euler angles.
        *pitch = std::asin(-T(2) * q1 * q3 + T(2) * q0 * q2);
        *roll  = std::atan2(T(2) * q2 * q3 + T(2) * q0 * q1,
                            -T(2) * q1 * q1 - T(2) * q2 * q2 + T(1));
        *yaw   = std::atan2(T(2) * (q1 * q2 + q0 * q3),
                            q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3);

        *pitch *= kRadToDeg;
        *roll  *= kRadToDeg;
        *yaw   *= kRadToDeg;
    }

    // updateMag()  (IMU + magnetometer)
    void updateMag(T gx, T gy, T gz,
                   T ax, T ay, T az,
                   T mx, T my, T mz,
                   T* pitch, T* roll, T* yaw,
                   T delta_t_sec)
    {
        // Fall back to IMU-only if magnetometer is all-zero.
        if ((mx == T(0)) && (my == T(0)) && (mz == T(0))) {
            update(gx, gy, gz, ax, ay, az, pitch, roll, yaw, delta_t_sec);
            return;
        }
    
        T recipNorm;
        T halfex, halfey, halfez;
    
        if (!((ax == T(0)) && (ay == T(0)) && (az == T(0)))) {
    
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
            const T q0q0 = q0 * q0;
            const T q0q1 = q0 * q1;
            const T q0q2 = q0 * q2;
            const T q0q3 = q0 * q3;
            const T q1q1 = q1 * q1;
            const T q1q2 = q1 * q2;
            const T q1q3 = q1 * q3;
            const T q2q2 = q2 * q2;
            const T q2q3 = q2 * q3;
            const T q3q3 = q3 * q3;
    
            // Rotate measured magnetic field into the auxiliary/world frame.
            const T hx = T(2) * (mx * (T(0.5) - q2q2 - q3q3) +
                                 my * (q1q2 - q0q3) +
                                 mz * (q1q3 + q0q2));
            const T hy = T(2) * (mx * (q1q2 + q0q3) +
                                 my * (T(0.5) - q1q1 - q3q3) +
                                 mz * (q2q3 - q0q1));
            const T hz = T(2) * (mx * (q1q3 - q0q2) +
                                 my * (q2q3 + q0q1) +
                                 mz * (T(0.5) - q1q1 - q2q2));
    
            // ENU / Z-up auxiliary frame:
            // horizontal magnetic reference is along +Y (North), not +X.
            const T by = std::sqrt(hx * hx + hy * hy);
            const T bz = hz;
    
            // Estimated direction of gravity in body frame.
            const T halfvx = q1q3 - q0q2;
            const T halfvy = q0q1 + q2q3;
            const T halfvz = T(0.5) * (q0q0 - q1q1 - q2q2 + q3q3);
    
            // Estimated magnetic field in body frame for auxiliary/world ref [0, by, bz].
            const T halfwx = by * (q1q2 + q0q3)           + bz * (q1q3 - q0q2);
            const T halfwy = by * (T(0.5) - q1q1 - q3q3) + bz * (q0q1 + q2q3);
            const T halfwz = by * (q2q3 - q0q1)           + bz * (T(0.5) - q1q1 - q2q2);
    
            // Error = cross product between estimated and measured field vectors.
            halfex = (ay * halfvz - az * halfvy) + (my * halfwz - mz * halfwy);
            halfey = (az * halfvx - ax * halfvz) + (mz * halfwx - mx * halfwz);
            halfez = (ax * halfvy - ay * halfvx) + (mx * halfwy - my * halfwx);
    
            // Integral feedback.
            if (twoKi > T(0)) {
                integralFBx += twoKi * halfex * delta_t_sec;
                integralFBy += twoKi * halfey * delta_t_sec;
                integralFBz += twoKi * halfez * delta_t_sec;
                gx += integralFBx;
                gy += integralFBy;
                gz += integralFBz;
            } else {
                integralFBx = T(0);
                integralFBy = T(0);
                integralFBz = T(0);
            }
    
            // Proportional feedback.
            gx += twoKp * halfex;
            gy += twoKp * halfey;
            gz += twoKp * halfez;
        }
    
        // Integrate rate of change of quaternion.
        const T half_dt = T(0.5) * delta_t_sec;
        const T q_0 = q0;
        const T q_1 = q1;
        const T q_2 = q2;
        const T q_3 = q3;
    
        q0 += (-q_1 * gx - q_2 * gy - q_3 * gz) * half_dt;
        q1 += ( q_0 * gx + q_2 * gz - q_3 * gy) * half_dt;
        q2 += ( q_0 * gy - q_1 * gz + q_3 * gx) * half_dt;
        q3 += ( q_0 * gz + q_1 * gy - q_2 * gx) * half_dt;
    
        // Normalise quaternion.
        recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        q0 *= recipNorm;
        q1 *= recipNorm;
        q2 *= recipNorm;
        q3 *= recipNorm;
    
        // Euler angles.
        *pitch = std::asin(-T(2) * q1 * q3 + T(2) * q0 * q2);
        *roll  = std::atan2(T(2) * q2 * q3 + T(2) * q0 * q1,
                            -T(2) * q1 * q1 - T(2) * q2 * q2 + T(1));
        *yaw   = std::atan2(T(2) * (q1 * q2 + q0 * q3),
                            q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3);
    
        *pitch *= kRadToDeg;
        *roll  *= kRadToDeg;
        *yaw   *= kRadToDeg;
    }
};

using Mahony_AHRS_Vars = Mahony_AHRS<float>;

static constexpr float twoKpDef = Mahony_AHRS<float>::twoKpDef;
static constexpr float twoKiDef = Mahony_AHRS<float>::twoKiDef;
static constexpr double kRadToDeg = Mahony_AHRS<double>::kRadToDeg;

// Backward-compatible free-function wrappers.
template<typename T>
inline void mahony_AHRS_init(Mahony_AHRS<T>* m, T twoKp, T twoKi) {
    m->init(twoKp, twoKi);
}

inline void mahony_AHRS_init(Mahony_AHRS_Vars* m, float twoKp, float twoKi) {
    m->init(twoKp, twoKi);
}

template<typename T>
inline void mahony_AHRS_reset(Mahony_AHRS<T>* m,
                              T twoKp = T(-1),
                              T twoKi = T(-1))
{
    m->reset(twoKp, twoKi);
}

inline void mahony_AHRS_reset(Mahony_AHRS_Vars* m,
                              float twoKp = -1.0f,
                              float twoKi = -1.0f)
{
    m->reset(twoKp, twoKi);
}

template<typename T>
inline void mahony_AHRS_update(Mahony_AHRS<T>* m,
                               T gx, T gy, T gz,
                               T ax, T ay, T az,
                               T* pitch, T* roll, T* yaw,
                               T delta_t_sec)
{
    m->update(gx, gy, gz, ax, ay, az, pitch, roll, yaw, delta_t_sec);
}

inline void mahony_AHRS_update(Mahony_AHRS_Vars* m,
                               float gx, float gy, float gz,
                               float ax, float ay, float az,
                               float* pitch, float* roll, float* yaw,
                               float delta_t_sec)
{
    m->update(gx, gy, gz, ax, ay, az, pitch, roll, yaw, delta_t_sec);
}

template<typename T>
inline void mahony_AHRS_update_mag(Mahony_AHRS<T>* m,
                                   T gx, T gy, T gz,
                                   T ax, T ay, T az,
                                   T mx, T my, T mz,
                                   T* pitch, T* roll, T* yaw,
                                   T delta_t_sec)
{
    m->updateMag(gx, gy, gz, ax, ay, az, mx, my, mz, pitch, roll, yaw, delta_t_sec);
}

inline void mahony_AHRS_update_mag(Mahony_AHRS_Vars* m,
                                   float gx, float gy, float gz,
                                   float ax, float ay, float az,
                                   float mx, float my, float mz,
                                   float* pitch, float* roll, float* yaw,
                                   float delta_t_sec)
{
    m->updateMag(gx, gy, gz, ax, ay, az, mx, my, mz, pitch, roll, yaw, delta_t_sec);
}

#endif // Mahony_AHRS_h
