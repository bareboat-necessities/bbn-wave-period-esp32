#include "ImuCalWizardRunner.h"
#include "AtomS3R_ImuCalWizard.h"

namespace atoms3r_ical {

#if !defined(NO_BOSCH_API)

bool runImuCalWizard(M5Ui& ui, ImuCalStoreNvs& store, BoschBmi270_ImuCal& imu, ImuCalBlobV1& out_saved) {
  ImuCalWizard wizard(ui, store, imu);
  return wizard.runAndSave(out_saved);
}

bool runImuCalWizard(M5Ui& /*ui*/, ImuCalStoreNvs& /*store*/, ImuCalBlobV1& /*out_saved*/) {
  // Bosch build requires passing the IMU source explicitly.
  Serial.println("[WIZ] ERROR: Bosch build requires runImuCalWizard(ui, store, imu, out_saved)");
  return false;
}

#else  // NO_BOSCH_API

bool runImuCalWizard(M5Ui& ui, ImuCalStoreNvs& store, ImuCalBlobV1& out_saved) {
  ImuCalWizard wizard(ui, store);
  return wizard.runAndSave(out_saved);
}

#endif

}  // namespace atoms3r_ical
