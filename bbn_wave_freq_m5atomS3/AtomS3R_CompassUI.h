#pragma once

/*

  Copyright 2026, Mikhail Grushinskiy

*/

#include <M5Unified.h>
#include <cmath>
#include <cstring>
#include <algorithm>

static inline float wrap360(float deg) {
  while (deg < 0) deg += 360.0f;
  while (deg >= 360.0f) deg -= 360.0f;
  return deg;
}

static inline const char* cardinal8(float deg) {
  // 8-wind labels
  static const char* k[] = {"N","NE","E","SE","S","SW","W","NW"};
  int idx = (int)lroundf(wrap360(deg) / 45.0f) & 7;
  return k[idx];
}

class CompassUI {
 public:
  bool ok() const { return _ok; }

  void begin() {
    _ok = false;

    _w = M5.Display.width();
    _h = M5.Display.height();
    _cx = _w / 2;
    _cy = _h / 2;

    _rOuter = (std::min(_w, _h) / 2) - 3;
    _rInner = _rOuter - 10;

    // Allocate background sprite
    _bg.setColorDepth(16);
    if (!_bg.createSprite(_w, _h)) return;

    // Allocate frame sprite (DO NOT assign a temporary canvas)
    _frame.setColorDepth(16);
    if (!_frame.createSprite(_w, _h)) {
      _bg.deleteSprite();
      return;
    }

    drawBackground();
    _ok = true;
  }

  void draw(float headingDeg,
            bool magOk = true,
            float magStrength_uT = NAN,
            bool tiltWarn = false) {
    if (!_ok) return;

    headingDeg = wrap360(headingDeg);

    auto* fb = (uint8_t*)_frame.getBuffer();
    auto* bb = (uint8_t*)_bg.getBuffer();
    if (!fb || !bb) return;               // avoid null memcpy crash

    // 16-bit RGB565 => 2 bytes/pixel
    std::memcpy(fb, bb, (size_t)_w * (size_t)_h * 2u);

    drawNeedle(headingDeg);
    drawCenterText(headingDeg, magOk, magStrength_uT, tiltWarn);

    // Push explicitly to display (no parent pointer needed)
    _frame.pushSprite(&M5.Display, 0, 0);
  }

 private:
  void drawBackground() {
    _bg.fillSprite(TFT_BLACK);

    // Ring
    _bg.drawCircle(_cx, _cy, _rOuter, TFT_DARKGREY);
    _bg.drawCircle(_cx, _cy, _rInner, TFT_DARKGREY);

    // Ticks + labels
    for (int deg = 0; deg < 360; deg += 5) {
      float a = deg * (float)M_PI / 180.0f;
      float sn = sinf(a);
      float cs = cosf(a);

      int len = (deg % 30 == 0) ? 8 : (deg % 10 == 0 ? 5 : 3);
      int r1 = _rOuter - 1;
      int r0 = r1 - len;

      int x0 = _cx + (int)lroundf(sn * r0);
      int y0 = _cy - (int)lroundf(cs * r0);
      int x1 = _cx + (int)lroundf(sn * r1);
      int y1 = _cy - (int)lroundf(cs * r1);

      uint16_t col = (deg % 30 == 0) ? TFT_LIGHTGREY : TFT_DARKGREY;
      _bg.drawLine(x0, y0, x1, y1, col);
    }

    // Cardinal letters (keep it big + readable)
    _bg.setTextColor(TFT_WHITE, TFT_BLACK);
    _bg.setTextSize(1);

    // If your build supports datum:
    // _bg.setTextDatum(middle_center);

    drawLabel("N", _cx, _cy - (_rInner - 10));
    drawLabel("S", _cx, _cy + (_rInner - 10));
    drawLabel("E", _cx + (_rInner - 10), _cy);
    drawLabel("W", _cx - (_rInner - 10), _cy);
  }

  void drawNeedle(float headingDeg) {
    // Screen coordinates: +x right, +y down
    // We want 0° (north) to point up => x = sin, y = -cos
    float a = headingDeg * (float)M_PI / 180.0f;
    float sn = sinf(a), cs = cosf(a);

    int tipR  = _rInner - 2;
    int baseR = 18;
    int halfW = 6;

    int xt = _cx + (int)lroundf(sn * tipR);
    int yt = _cy - (int)lroundf(cs * tipR);

    int xb = _cx + (int)lroundf(sn * baseR);
    int yb = _cy - (int)lroundf(cs * baseR);

    // Perp direction for needle width
    int px = (int)lroundf(cs * halfW);
    int py = (int)lroundf(sn * halfW);

    int xL = xb - px, yL = yb - py;
    int xR = xb + px, yR = yb + py;

    // North triangle
    _frame.fillTriangle(xt, yt, xL, yL, xR, yR, TFT_RED);

    // South tail (small)
    int tailR = 10;
    int xs = _cx - (int)lroundf(sn * tailR);
    int ys = _cy + (int)lroundf(cs * tailR);
    _frame.fillCircle(xs, ys, 3, TFT_LIGHTGREY);

    // Center hub
    _frame.fillCircle(_cx, _cy, 4, TFT_BLACK);
    _frame.drawCircle(_cx, _cy, 4, TFT_LIGHTGREY);
  }

  void drawCenterText(float headingDeg, bool magOk, float mag_uT, bool tiltWarn) {
    // Big numeric
    char buf[16];
    int hdg = (int)lroundf(headingDeg);
    snprintf(buf, sizeof(buf), "%d", hdg);
    
    _frame.setTextColor(TFT_WHITE, TFT_BLACK);
    _frame.setTextSize(2);

    int yText = _cy - 10;
    centerPrint(buf, yText);

    // draw degree circle to the right of the number
    int w = _frame.textWidth(buf);
    int x = (_w - w) / 2 + w + 4;
    int y = yText + 2;
    _frame.drawCircle(x, y, 3, TFT_WHITE);

    // Small cardinal
    _frame.setTextSize(1);
    centerPrint(cardinal8(headingDeg), yText + 22);

    // Bottom status strip
    if (false) {
      _frame.fillRect(0, _h - 16, _w, 16, TFT_BLACK);
      _frame.setCursor(2, _h - 14);
      _frame.setTextColor(magOk ? TFT_GREEN : TFT_ORANGE, TFT_BLACK);
      _frame.print(magOk ? "MAG OK" : "MAG ?");
  
      _frame.setTextColor(tiltWarn ? TFT_ORANGE : TFT_DARKGREY, TFT_BLACK);
      _frame.setCursor(_w - 40, _h - 14);
      _frame.print(tiltWarn ? "TILT" : "    ");
  
      if (std::isfinite(mag_uT)) {
        _frame.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
        _frame.setCursor(_w/2 - 18, _h - 14);
        char mbuf[16];
        snprintf(mbuf, sizeof(mbuf), "%4.1fuT", mag_uT);
        _frame.print(mbuf);
      }
    }
  }

  void centerPrint(const char* s, int y) {
    // Works on M5Canvas (no getTextBounds on some builds)
    int ww = _frame.textWidth(s);
    int x = (_w - ww) / 2;
    _frame.setCursor(x, y);
    _frame.print(s);
  }

  void drawLabel(const char* s, int x, int y) {
    int ww = _bg.textWidth(s);
    int hh = _bg.fontHeight();   // current font height with current text size
    _bg.setCursor(x - ww / 2, y - hh / 2);
    _bg.print(s);
  }

 private:
  bool _ok = false;
  int _w = 128, _h = 128, _cx = 64, _cy = 64;
  int _rOuter = 60, _rInner = 50;

  M5Canvas _bg;
  M5Canvas _frame;
};

#ifdef COMPASS_UI_STANDALONE

CompassUI ui;

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);

  ui.begin();

  // IMU (BMI270 + BMM150 on AtomS3R)  [oai_citation:4‡M5Stack Docs](https://docs.m5stack.com/en/core/AtomS3R)
  M5.Imu.begin();
}

void loop() {
  M5.update();

  if (M5.Imu.update()) {
    // getImuData includes accel/gyro/mag  [oai_citation:5‡m5stack.oss-cn-shenzhen.aliyuncs.com](https://m5stack.oss-cn-shenzhen.aliyuncs.com/resource/docs/static/pdf/static/en/arduino/m5unified/imu_class.pdf)
    auto d = M5.Imu.getImuData();

    // Very basic “flat” heading (no tilt compensation):
    float heading = atan2f(d.mag.y, d.mag.x) * 180.0f / (float)M_PI;
    heading = wrap360(heading);

    float mag_uT = sqrtf(d.mag.x*d.mag.x + d.mag.y*d.mag.y + d.mag.z*d.mag.z);

    ui.draw(heading, true, mag_uT, false);
  }

  delay(16); // ~60 fps cap
}

#endif
