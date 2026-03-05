#pragma once

#include "AtomS3R_ImuCal.h"
#include "AtomS3R_M5Ui.h"

#if !defined(NO_BOSCH_API)
  #include "BoschBmi270_ImuCal.h"
#endif

namespace atoms3r_ical {

// Bosch path (default build): caller must provide an initialized BoschBmi270_ImuCal source.
#if !defined(NO_BOSCH_API)
bool runImuCalWizard(M5Ui& ui, ImuCalStoreNvs& store, BoschBmi270_ImuCal& imu, ImuCalBlobV1& out_saved);
#endif

// Legacy path (only when NO_BOSCH_API is defined)
bool runImuCalWizard(M5Ui& ui, ImuCalStoreNvs& store, ImuCalBlobV1& out_saved);

}  // namespace atoms3r_ical
