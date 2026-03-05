#pragma once

#include "AtomS3R_ImuCal.h"
#include "AtomS3R_M5Ui.h"

// Bosch availability detection
// We detect the *vendor headers* directly
// as a guard if its dependencies may not be installed on the build machine).
#ifndef ATOMS3R_HAVE_BOSCH_API
  #if !defined(NO_BOSCH_API) && \
      __has_include(<utilities/BMI270-Sensor-API/bmi2.h>) && \
      __has_include(<utilities/BMI270-Sensor-API/bmi270.h>) && \
      __has_include(<utilities/BMM150-Sensor-API/bmm150.h>)
    #define ATOMS3R_HAVE_BOSCH_API 1
  #else
    #define ATOMS3R_HAVE_BOSCH_API 0
  #endif
#endif

namespace atoms3r_ical {

// Forward declare only — NO Bosch includes here.
#if ATOMS3R_HAVE_BOSCH_API
class BoschBmi270_ImuCal;
#endif

// Bosch path: only declared when the Bosch vendor headers are actually available.
#if ATOMS3R_HAVE_BOSCH_API
bool runImuCalWizard(M5Ui& ui,
                     ImuCalStoreNvs& store,
                     BoschBmi270_ImuCal& imu,
                     ImuCalBlobV1& out_saved);
#endif

// Legacy path (always available): uses M5.Imu internally.
bool runImuCalWizard(M5Ui& ui,
                     ImuCalStoreNvs& store,
                     ImuCalBlobV1& out_saved);

} // namespace atoms3r_ical
