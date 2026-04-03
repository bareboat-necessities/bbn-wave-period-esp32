#include "ImuCalWizardRunner.h"

#include <Wire.h>

#include "BoschBmi270_ImuCal.h"
#include "AtomS3R_ImuCalWizard.h"

namespace atoms3r_ical {

static constexpr uint8_t RUNNER_BMI270_ADDR = 0x68;
static constexpr float   RUNNER_AG_HZ       = 200.0f;

bool runImuCalWizard(M5Ui& ui, ImuCalStoreNvs& store, ImuCalBlobV1& out_saved) {
  BoschBmi270_ImuCal imu;

  BoschBmi270_ImuCal::Config cfg;
  cfg.bmi270_addr                = RUNNER_BMI270_ADDR;
  cfg.ag_hz                      = RUNNER_AG_HZ;
  cfg.enable_mag_aux             = true;
  cfg.mag_bmm150_addr            = 0x10;
  cfg.mag_aux_odr_hz             = 25.0f;
  cfg.mag_startup_settle_ms      = 3;
  cfg.mag_verify_first_read      = true;
  cfg.mag_stale_min_us           = 75000u;
  cfg.mag_stale_factor           = 3u;
  cfg.enable_mag_recovery        = true;
  cfg.mag_recover_after_failures = 6u;
  cfg.mag_recover_cooldown_us    = 1000000u;
  cfg.tempC_default              = 25.0f;
  cfg.i2c_hz                     = 400000u;

  if (!imu.begin(M5.In_I2C, cfg)) {
    Serial.printf("[WIZ] IMU init failed: %s\n", imu.lastErrorString());
    Serial.printf("[WIZ] FIFO detail: %s\n", imu.fifo().lastErrorString());
    Serial.printf("[WIZ] FIFO init path: %s\n", imu.fifo().initPathString());
    Serial.printf("[WIZ] FIFO Bosch init rslt: %d\n", (int)imu.fifo().lastBoschInitResult());
    Serial.printf("[WIZ] BMI addr tried: 0x%02X\n", cfg.bmi270_addr);
    return false;
  }

  ImuCalWizard wizard(ui, store, imu);
  const bool ok = wizard.runAndSave(out_saved);

  (void)imu.end();
  return ok;
}

} // namespace atoms3r_ical
