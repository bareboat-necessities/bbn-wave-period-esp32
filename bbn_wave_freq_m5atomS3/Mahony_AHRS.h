#ifndef Mahony_AHRS_h
#define Mahony_AHRS_h

//
// Madgwick's implementation of Mahony's AHRS algorithm.
// See: http://www.x-io.co.uk/node/8#open_source_ahrs_and_imu_algorithms
//

#include <math.h>

#define RAD_TO_DEG 57.295779513082320876798154814105

#define twoKpDef   (2.0f * 1.0f)  // 2 * proportional gain
#define twoKiDef   (2.0f * 0.0f)  // 2 * integral gain

typedef struct mahony_AHRS_vars {
  float twoKp = twoKpDef;  // 2 * proportional gain (Kp)
  float twoKi = twoKiDef;  // 2 * integral gain (Ki)
  float q0 = 1.0;
  float q1 = 0.0;
  float q2 = 0.0;
  float q3 = 0.0;  // quaternion of sensor frame relative to auxiliary frame
  float integralFBx = 0.0f;
  float integralFBy = 0.0f;
  float integralFBz = 0.0f;  // integral error terms scaled by Ki
} Mahony_AHRS_Vars;

void mahony_AHRS_init(Mahony_AHRS_Vars* m, float twoKp, float twoKi);

void mahony_AHRS_update(Mahony_AHRS_Vars* m,
                        float gx, float gy, float gz, float ax, float ay, float az, 
                        float *pitch, float *roll, float *yaw, float delta_t_sec);
float invSqrt(float x);

/*
  The gain is the Kp term in a PID controller, tune it as you would any PID controller (missing the I and D terms).
  If Kp is too low, the filter will respond slowly to changes in sensor orientation. 
  If too high, the filter output will oscillate.
*/
void mahony_AHRS_init(Mahony_AHRS_Vars* m, float twoKp, float twoKi) {
  m->twoKp = twoKp;
  m->twoKi = twoKi;
}

// IMU algorithm update (without magnetometer)
void mahony_AHRS_update(Mahony_AHRS_Vars* m,
                        float gx, float gy, float gz, float ax, float ay, float az,
                        float *pitch, float *roll, float *yaw, float delta_t_sec) {
  float recipNorm;
  float halfvx, halfvy, halfvz;
  float halfex, halfey, halfez;
  float qa, qb, qc;

  // Compute feedback only if accelerometer measurement valid (avoids NaN in
  // accelerometer normalisation)
  if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {
    // Normalise accelerometer measurement
    recipNorm = invSqrt(ax * ax + ay * ay + az * az);
    ax *= recipNorm;
    ay *= recipNorm;
    az *= recipNorm;

    // Estimated direction of gravity and vector perpendicular to magnetic flux
    halfvx = m->q1 * m->q3 - m->q0 * m->q2;
    halfvy = m->q0 * m->q1 + m->q2 * m->q3;
    halfvz = m->q0 * m->q0 - 0.5f + m->q3 * m->q3;

    // Error is sum of cross product between estimated and measured direction of gravity
    halfex = (ay * halfvz - az * halfvy);
    halfey = (az * halfvx - ax * halfvz);
    halfez = (ax * halfvy - ay * halfvx);

    // Compute and apply integral feedback if enabled
    if (m->twoKi > 0.0f) {
      m->integralFBx += m->twoKi * halfex * delta_t_sec;  // integral error scaled by Ki
      m->integralFBy += m->twoKi * halfey * delta_t_sec;
      m->integralFBz += m->twoKi * halfez * delta_t_sec;
      gx += m->integralFBx;  // apply integral feedback
      gy += m->integralFBy;
      gz += m->integralFBz;
    } else {
      m->integralFBx = 0.0f;  // prevent integral windup
      m->integralFBy = 0.0f;
      m->integralFBz = 0.0f;
    }

    // Apply proportional feedback
    gx += m->twoKp * halfex;
    gy += m->twoKp * halfey;
    gz += m->twoKp * halfez;
  }

  // Integrate rate of change of quaternion
  gx *= (0.5f * delta_t_sec);  // pre-multiply common factors
  gy *= (0.5f * delta_t_sec);
  gz *= (0.5f * delta_t_sec);
  qa = m->q0;
  qb = m->q1;
  qc = m->q2;
  m->q0 += (-qb * gx - qc * gy - m->q3 * gz);
  m->q1 += (qa * gx + qc * gz - m->q3 * gy);
  m->q2 += (qa * gy - qb * gz + m->q3 * gx);
  m->q3 += (qa * gz + qb * gy - qc * gx);

  // Normalise quaternion
  recipNorm = invSqrt(m->q0 * m->q0 + m->q1 * m->q1 + m->q2 * m->q2 + m->q3 * m->q3);
  m->q0 *= recipNorm;
  m->q1 *= recipNorm;
  m->q2 *= recipNorm;
  m->q3 *= recipNorm;

  *pitch = asin(-2 * m->q1 * m->q3 + 2 * m->q0 * m->q2);  // pitch
  *roll  = atan2(2 * m->q2 * m->q3 + 2 * m->q0 * m->q1,
                 -2 * m->q1 * m->q1 - 2 * m->q2 * m->q2 + 1);  // roll
  *yaw   = atan2(2 * (m->q1 * m->q2 + m->q0 * m->q3),
                 m->q0 * m->q0 + m->q1 * m->q1 - m->q2 * m->q2 - m->q3 * m->q3);  // yaw

  *pitch *= RAD_TO_DEG;
  *yaw *= RAD_TO_DEG;
  *roll *= RAD_TO_DEG;
}

/**
 * Fast inverse square root implementation. Compatible both for 32 and 8 bit microcontrollers.
 * @see http://en.wikipedia.org/wiki/Fast_inverse_square_root
*/
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

float invSqrt(float number) {
  union {
    float f;
    int32_t i;
  } y;

  y.f = number;
  y.i = 0x5f375a86 - (y.i >> 1);
  y.f = y.f * ( 1.5f - ( number * 0.5f * y.f * y.f ) );
  return y.f;
}

/* Old 8bit version. Kept it here only for testing/debugging of the new code above.
float invSqrt(float number) {
  volatile long i;
  volatile float x, y;
  volatile const float f = 1.5F;

  x = number * 0.5F;
  y = number;
  i = * ( long * ) &y;
  i = 0x5f375a86 - ( i >> 1 );
  y = * ( float * ) &i;
  y = y * ( f - ( x * y * y ) );
  return y;
}
*/

#pragma GCC diagnostic pop

#endif

