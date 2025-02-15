#ifndef CentripetalAcceleration_h
#define CentripetalAcceleration_h

// Structure to represent a 3D vector
typedef struct {
    float x;
    float y;
    float z;
} Vector3D;

// Function to compute the cross product of two 3D vectors
Vector3D cross_product(Vector3D a, Vector3D b) {
    Vector3D result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

// Function to subtract two 3D vectors
Vector3D subtract_vectors(Vector3D a, Vector3D b) {
    Vector3D result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
}

// Function to estimate centripetal acceleration
// Note: the vectors are in the same frame so if
// gyro is in IMU frame of coordinates then
// velocity must br in same device body frame (not Earth xyz frame)
Vector3D estimate_centripetal_acceleration(Vector3D velocity, Vector3D gyro_data) {
    // Centripetal acceleration: a_c = ω × v
    return cross_product(gyro_data, velocity);
}

// Function to compensate for centripetal acceleration
Vector3D compensate_centripetal_acceleration(Vector3D accel_data, Vector3D velocity, Vector3D gyro_data) {
    // Estimate centripetal acceleration
    Vector3D a_c = estimate_centripetal_acceleration(velocity, gyro_data);

    // Subtract centripetal acceleration from accelerometer data
    return subtract_vectors(accel_data, a_c);
}

#endif
