#pragma once

#include <Arduino.h>
#include <M5Unified.h>
#include <math.h>

namespace atoms3r_ical {

// Standalone UI config 
struct M5UiCfg {
  // If text rotated wrong initially, change this to 1 or 3.
  static constexpr uint8_t  ROT_READ = 0;
  static constexpr uint8_t  LCD_BRIGHTNESS = 200;

  // Keep-awake / pacing defaults (wizard can live with these)
  static constexpr uint32_t KEEP_AWAKE_EVERY_MS = 200;
  static constexpr uint32_t OK_PAUSE_MS         = 900;
  static constexpr uint32_t MENU_TAP_WINDOW_MS  = 650;
};

static inline float clamp01_(float x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

// Input: BtnA ONLY + keep-awake (self-contained)
class Input {
public:
  static void update() {
    M5.update();

    const uint32_t now = millis();
    if ((uint32_t)(now - last_keep_awake_ms_) >= M5UiCfg::KEEP_AWAKE_EVERY_MS) {
      last_keep_awake_ms_ = now;
      M5.Display.setBrightness(M5UiCfg::LCD_BRIGHTNESS);
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
    rot_ = M5UiCfg::ROT_READ;
    M5.Display.setRotation(rot_);
    M5.Display.setBrightness(M5UiCfg::LCD_BRIGHTNESS);
    M5.Display.setTextSize(1);
  }

  void setRotation(uint8_t r) {
    rot_ = (uint8_t)(r & 3);
    M5.Display.setRotation(rot_);
  }

  void setReadRotation() { setRotation(M5UiCfg::ROT_READ); }

  void clear() {
    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setCursor(0, 0);
  }

  void title(const char* t) {
    clear();
    M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Display.setTextSize(2);
    M5.Display.println(t);
    M5.Display.setTextSize(1);
    M5.Display.println();
  }

  void line(const char* s = "") {
    M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Display.println(s);
  }

  void bar01(float t01) {
    t01 = clamp01_(t01);
    const int x = 6;
    const int w = M5.Display.width() - 12;
    const int h = 10;
    const int y = M5.Display.height() - 18;
    M5.Display.drawRect(x, y, w, h, TFT_DARKGREY);
    const int fillw = (int)((w - 2) * t01);
    M5.Display.fillRect(x + 1, y + 1, fillw, h - 2, TFT_GREEN);
  }

  void waitTap(const char* t, const char* l1 = nullptr, const char* l2 = nullptr) {
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

  void showOkAuto(const char* l1 = nullptr, const char* l2 = nullptr) {
    title("OK");
    if (l1) line(l1);
    if (l2) line(l2);
    const uint32_t t0 = millis();
    while (millis() - t0 < M5UiCfg::OK_PAUSE_MS) {
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
    const uint32_t t0 = millis();
    while (millis() - t0 < 4500) {
      Input::update();
      if (Input::tapPressed()) return true;
      delay(10);
    }
    return false;
  }

  enum class MagFailAction : uint8_t { RETRY_MAG = 0, REDO_ALL = 1, ABORT = 2 };

  // No timeout (tap-group UI)
  MagFailAction magFailMenu(const char* why1, const char* why2 = nullptr) {
    setReadRotation();
    title("MAG FAIL");
    if (why1) line(why1);
    if (why2) line(why2);
    line("");
    line("Tap: retry MAG");
    line("Tap x2: redo ALL");
    line("Tap x3: abort");

    const uint8_t taps = waitTapGroupNoTimeout_(M5UiCfg::MENU_TAP_WINDOW_MS);
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

    // wait for first tap
    while (true) {
      Input::update();
      if (Input::tapPressed()) {
        count = 1;
        deadline = millis() + window_ms;
        break;
      }
      delay(10);
    }

    // collect taps until window expires
    while (true) {
      Input::update();
      if (Input::tapPressed()) {
        ++count;
        deadline = millis() + window_ms;
      }
      if ((int32_t)(millis() - deadline) > 0) break;
      delay(10);
    }
    return count;
  }

  uint8_t rot_ = M5UiCfg::ROT_READ;
};

} // namespace atoms3r_ical
