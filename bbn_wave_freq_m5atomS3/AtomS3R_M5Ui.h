#pragma once

#include <Arduino.h>
#include <M5Unified.h>
#include <math.h>

namespace atoms3r_ical {

struct M5UiCfg {
  static constexpr uint8_t  ROT_READ = 0;
  static constexpr uint8_t  LCD_BRIGHTNESS = 200;
};

static inline float clamp01_(float x){ return x<0?0:(x>1?1:x); }

// Input: BtnA ONLY + keep-awake
class Input {
public:
  static void update() {
    M5.update();

    uint32_t now = millis();
    if ((uint32_t)(now - last_keep_awake_ms_) > ImuCalWizardCfg::KEEP_AWAKE_EVERY_MS) {
      last_keep_awake_ms_ = now;
      M5.Display.setBrightness(ImuCalWizardCfg::LCD_BRIGHTNESS);
      M5.Display.wakeup();
    }
    tap_edge_ = M5.BtnA.wasPressed();
  }

  static bool tapPressed() { return tap_edge_; }

private:
  static inline bool tap_edge_ = false;
  static inline uint32_t last_keep_awake_ms_ = 0;
};

// UI helpers
class M5Ui {
public:
  void begin() {
    rot_ = ImuCalWizardCfg::ROT_READ;
    M5.Display.setRotation(rot_);
    M5.Display.setBrightness(ImuCalWizardCfg::LCD_BRIGHTNESS);
    M5.Display.setTextSize(1);
  }

  void setRotation(uint8_t r) {
    rot_ = (uint8_t)(r & 3);
    M5.Display.setRotation(rot_);
  }

  void setReadRotation() { setRotation(ImuCalWizardCfg::ROT_READ); }

  void clear() {
    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setCursor(0,0);
  }

  void title(const char* t) {
    clear();
    M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Display.setTextSize(2);
    M5.Display.println(t);
    M5.Display.setTextSize(1);
    M5.Display.println();
  }

  void line(const char* s) {
    M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Display.println(s);
  }

  void bar01(float t01) {
    t01 = clamp01_(t01);
    int x = 6;
    int w = M5.Display.width() - 12;
    int h = 10;
    int y = M5.Display.height() - 18;
    M5.Display.drawRect(x, y, w, h, TFT_DARKGREY);
    int fillw = (int)((w-2) * t01);
    M5.Display.fillRect(x+1, y+1, fillw, h-2, TFT_GREEN);
  }

  void waitTap(const char* t, const char* l1=nullptr, const char* l2=nullptr) {
    setReadRotation();
    title(t);
    if (l1) line(l1);
    if (l2) line(l2);
    line("");
    line("Tap BtnA");
    while (true) {
      Input::update();
      if (Input::tapPressed()) break;
      delay(10);
    }
  }

  void showOkAuto(const char* l1=nullptr, const char* l2=nullptr) {
    title("OK");
    if (l1) line(l1);
    if (l2) line(l2);
    uint32_t t0 = millis();
    while (millis() - t0 < ImuCalWizardCfg::OK_PAUSE_MS) {
      Input::update();
      if (Input::tapPressed()) break;
      delay(10);
    }
  }

  void fail(const char* where, const char* why) {
    setReadRotation();
    title("FAILED");
    line(where);
    line(why);
    line("");
    line("Tap BtnA");
    while (true) {
      Input::update();
      if (Input::tapPressed()) break;
      delay(10);
    }
  }

  bool eraseConfirm() {
    setReadRotation();
    title("ERASE?");
    line("Delete saved cal");
    line("Tap=YES  Wait=NO");
    uint32_t t0 = millis();
    while (millis() - t0 < 4500) {
      Input::update();
      if (Input::tapPressed()) return true;
      delay(10);
    }
    return false;
  }

  enum class MagFailAction : uint8_t { RETRY_MAG=0, REDO_ALL=1, ABORT=2 };

  // NO TIMEOUT 
  MagFailAction magFailMenu(const char* why1, const char* why2=nullptr) {
    setReadRotation();
    title("MAG FAIL");
    if (why1) line(why1);
    if (why2) line(why2);
    line("");
    line("Tap: retry MAG");
    line("Tap x2: redo ALL");
    line("Tap x3: abort");

    uint8_t taps = waitTapGroupNoTimeout_(ImuCalWizardCfg::MENU_TAP_WINDOW_MS);
    if (taps >= 3) return MagFailAction::ABORT;
    if (taps == 2) return MagFailAction::REDO_ALL;
    return MagFailAction::RETRY_MAG;
  }

  void notSavedNotice() {
    setReadRotation();
    title("NOT SAVED");
    line("Calibration not saved");
    line("See Serial log");
    line("");
    line("Tap BtnA");
    while (true) {
      Input::update();
      if (Input::tapPressed()) break;
      delay(10);
    }
  }

private:
  static uint8_t waitTapGroupNoTimeout_(uint32_t window_ms) {
    uint8_t count = 0;
    uint32_t deadline = 0;

    while (true) {
      Input::update();
      if (Input::tapPressed()) {
        count = 1;
        deadline = millis() + window_ms;
        break;
      }
      delay(10);
    }

    while (true) {
      Input::update();
      if (Input::tapPressed()) {
        count++;
        deadline = millis() + window_ms;
      }
      if ((int32_t)(millis() - deadline) > 0) break;
      delay(10);
    }
    return count;
  }

  uint8_t rot_ = ImuCalWizardCfg::ROT_READ;
};

} // namespace atoms3r_ical
