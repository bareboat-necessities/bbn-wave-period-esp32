#include "ImuCalWizardRunner.h"

#include "AtomS3R_ImuCalWizard.h"

namespace atoms3r_ical {

bool runImuCalWizard(M5Ui& ui, ImuCalStoreNvs& store, ImuCalBlobV1& out_saved) {
  ImuCalWizard wizard(ui, store);
  return wizard.runAndSave(out_saved);
}

}  // namespace atoms3r_ical
