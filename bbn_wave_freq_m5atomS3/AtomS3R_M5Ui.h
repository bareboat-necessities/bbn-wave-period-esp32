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

  void line(const char* s="") {
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

  void fail(const char* where, const char* why) {
    setReadRotation();
    title("FAILED");
    line(where);
    line(why);
  }

  // Optional: used only if wizard enabled (but harmless here)
  bool eraseConfirm() {
    setReadRotation();
    title("ERASE?");
    line("Delete saved cal");
    line("Tap=YES  Wait=NO");
    uint32_t t0 = millis();
    while (millis() - t0 < 4500) {
      M5.update();
      if (M5.BtnA.wasPressed()) return true;
      delay(10);
    }
    return false;
  }

  void notSavedNotice() {
    setReadRotation();
    title("NOT SAVED");
    line("Calibration not saved");
    line("See Serial log");
  }

private:
  uint8_t rot_ = M5UiCfg::ROT_READ;
};

} // namespace atoms3r_ical
