#include "ImuCalWizardRunner.h"
#include "AtomS3R_ImuCalWizard.h"

#if ATOMS3R_HAVE_BOSCH_API
  #include "BoschBmi270_ImuCal.h"
#endif

namespace atoms3r_ical {

bool runImuCalWizard(M5Ui& ui, ImuCalStoreNvs& store, ImuCalBlobV1& out_saved) {
  ImuCalWizard wizard(ui, store);
  return wizard.runAndSave(out_saved);
}

#if ATOMS3R_HAVE_BOSCH_API
bool runImuCalWizard(M5Ui& ui,
                     ImuCalStoreNvs& store,
                     BoschBmi270_ImuCal& imu,
                     ImuCalBlobV1& out_saved)
{
  ImuCalWizard wizard(ui, store, imu);
  return wizard.runAndSave(out_saved);
}
#endif

} // namespace atoms3r_ical
