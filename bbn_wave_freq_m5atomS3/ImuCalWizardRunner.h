#pragma once

#include "AtomS3R_ImuCal.h"
#include "AtomS3R_M5Ui.h"

namespace atoms3r_ical {

bool runImuCalWizard(M5Ui& ui, ImuCalStoreNvs& store, ImuCalBlobV1& out_saved);

}  // namespace atoms3r_ical
