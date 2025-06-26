#include "FentonWave.h"

#include <iostream>

template class FentonWave<3>;
template class WaveSurfaceTracker<3>;

void FentonWave_test() {
    // Wave parameters
    const float height = 2.0f;   // Wave height (m)
    const float depth = 10.0f;   // Water depth (m)
    const float length = 50.0f;  // Wavelength (m)
    
    // Simulation parameters
    const float duration = 20.0f; // Simulation duration (s)
    const float dt = 0.1f;       // Time step (s)

    // Create a 3rd-order Fenton wave and a surface tracker
    WaveSurfaceTracker<3> tracker(height, depth, length);

    // Output file
    std::ofstream out("wave_data.csv");
    out << "Time(s),Displacement(m),Velocity(m/s),Acceleration(m/s²),X_Position(m)\n";

    // Define the kinematics callback (writes data to file)
    auto kinematics_callback = [&out](
        float time, float elevation, float vertical_velocity, float vertical_acceleration, float horizontal_position) {
        out << time << "," << elevation << "," << vertical_velocity << "," << vertical_acceleration << "," << horizontal_position << "\n";
    };

    // Track Lagrangian kinematics (using callback)
    tracker.track_lagrangian_kinematics(duration, dt, kinematics_callback);

    std::cout << "Wave data saved to wave_data.csv\n";
}
