#ifndef M5_Calibr_h
#define M5_Calibr_h

#include <M5Unified.h>

// Strength of the calibration operation;
// 0: disables calibration.
// 1 is weakest and 255 is strongest.
static constexpr const uint8_t calib_value = 64;

// This code performs calibration by clicking on a button or screen.
// After 120 seconds of calibration, the results are stored in NVS.
// The saved calibration values are loaded at the next startup.
//
// === How to calibration ===
// ※ Calibration method for Accelerometer
//    Change the direction of the main unit by 90 degrees
//     and hold it still for 2 seconds. Repeat multiple times.
//     It is recommended that as many surfaces as possible be on the bottom.
//
// ※ Calibration method for Gyro
//    Simply place the unit on a quiet desk and hold it still.
//    It is recommended that this be done after the accelerometer calibration.
//
// ※ Calibration method for geomagnetic sensors
//    Rotate the main unit slowly in multiple directions.
//    It is recommended that as many surfaces as possible be oriented to the north.
//
// Values for extremely large attitude changes are ignored.
// During calibration, it is desirable to move the device as gently as possible.

struct rect_t {
  int32_t x;
  int32_t y;
  int32_t w;
  int32_t h;
};

static constexpr const uint32_t color_tbl[18] = {
  0xFF0000u, 0xCCCC00u, 0xCC00FFu,
  0xFFCC00u, 0x00FF00u, 0x0088FFu,
  0xFF00CCu, 0x00FFCCu, 0x0000FFu,
  0xFF0000u, 0xCCCC00u, 0xCC00FFu,
  0xFFCC00u, 0x00FF00u, 0x0088FFu,
  0xFF00CCu, 0x00FFCCu, 0x0000FFu,
};
static constexpr const float coefficient_tbl[3] = { 0.5f, (1.0f / 256.0f), (1.0f / 1024.0f) };

static auto &disp = (M5.Display);
static rect_t rect_graph_area;
static rect_t rect_text_area;

static uint8_t calib_countdown = 0;
static uint32_t frame_count = 0;
static uint32_t prev_sec = 0;

static int prev_xpos[18];

void drawBar(int32_t ox, int32_t oy, int32_t nx, int32_t px, int32_t h, uint32_t color) {
  uint32_t bgcolor = (color >> 3) & 0x1F1F1Fu;
  if (px && ((nx < 0) != (px < 0))) {
    disp.fillRect(ox, oy, px, h, bgcolor);
    px = 0;
  }
  if (px != nx) {
    if ((nx > px) != (nx < 0)) {
      bgcolor = color;
    }
    disp.setColor(bgcolor);
    disp.fillRect(nx + ox, oy, px - nx, h);
  }
}

void drawGraph(const rect_t& r, const m5::imu_data_t& data) {
  float aw = (128 * r.w) >> 1;
  float gw = (128 * r.w) / 256.0f;
  float mw = (128 * r.w) / 1024.0f;
  int ox = (r.x + r.w) >> 1;
  int oy = r.y;
  int h = (r.h / 18) * (calib_countdown ? 1 : 2);
  int bar_count = 9 * (calib_countdown ? 2 : 1);

  disp.startWrite();
  for (int index = 0; index < bar_count; ++index) {
    float xval;
    if (index < 9) {
      auto coe = coefficient_tbl[index / 3] * r.w;
      xval = data.value[index] * coe;
    }
    else {
      xval = M5.Imu.getOffsetData(index - 9) * (1.0f / (1 << 19));
    }

    // for Linear scale graph.
    float tmp = xval;

    // The smaller the value, the larger the amount of change in the graph.
    //  float tmp = sqrtf(fabsf(xval * 128)) * (signbit(xval) ? -1 : 1);

    int nx = tmp;
    int px = prev_xpos[index];
    if (nx != px)
      prev_xpos[index] = nx;
    drawBar(ox, oy + h * index, nx, px, h - 1, color_tbl[index]);
  }
  disp.endWrite();
}

void drawCalibrGraph(const rect_t& r, const m5::imu_data_t& data) {
    drawGraph(rect_graph_area, data);
    /*
        // The data obtained by getImuData can be used as follows.
        data.accel.x;      // accel x-axis value.
        data.accel.y;      // accel y-axis value.
        data.accel.z;      // accel z-axis value.
        data.accel.value;  // accel 3values array [0]=x / [1]=y / [2]=z.

        data.gyro.x;      // gyro x-axis value.
        data.gyro.y;      // gyro y-axis value.
        data.gyro.z;      // gyro z-axis value.
        data.gyro.value;  // gyro 3values array [0]=x / [1]=y / [2]=z.

        data.mag.x;       // mag x-axis value.
        data.mag.y;       // mag y-axis value.
        data.mag.z;       // mag z-axis value.
        data.mag.value;   // mag 3values array [0]=x / [1]=y / [2]=z.

        data.value;       // all sensor 9values array [0~2]=accel / [3~5]=gyro / [6~8]=mag

        M5_LOGV("ax:%f  ay:%f  az:%f", data.accel.x, data.accel.y, data.accel.z);
        M5_LOGV("gx:%f  gy:%f  gz:%f", data.gyro.x , data.gyro.y , data.gyro.z );
        M5_LOGV("mx:%f  my:%f  mz:%f", data.mag.x  , data.mag.y  , data.mag.z  );
      //*/
    ++frame_count;
}

void updateCalibration(uint32_t c, bool clear = false) {
  calib_countdown = c;

  if (c == 0) {
    clear = true;
  }

  if (clear) {
    memset(prev_xpos, 0, sizeof(prev_xpos));
    disp.fillScreen(TFT_BLACK);

    if (c) { // Start calibration.
      M5.Imu.setCalibration(calib_value, calib_value, 0);
      // ※ The actual calibration operation is performed each time during M5.Imu.update.
      //
      // There are three arguments, which can be specified in the order of Accelerometer, gyro, and geomagnetic.
      // If you want to calibrate only the Accelerometer, do the following.
      // M5.Imu.setCalibration(100, 0, 0);
      //
      // If you want to calibrate only the gyro, do the following.
      // M5.Imu.setCalibration(0, 100, 0);
      //
      // If you want to calibrate only the geomagnetism, do the following.
      // M5.Imu.setCalibration(0, 0, 100);
    }
    else { 
      // Stop calibration. (Continue calibration only for the geomagnetic sensor)
      //M5.Imu.setCalibration(0, 0, calib_value);

      // If you want to stop all calibration, write this.
      M5.Imu.setCalibration(0, 0, 0);

      // save calibration values.
      M5.Imu.saveOffsetToNVS();
    }
  }

  auto backcolor = (c == 0) ? TFT_BLACK : TFT_BLUE;
  disp.fillRect(rect_text_area.x, rect_text_area.y, rect_text_area.w, rect_text_area.h, backcolor);

  if (c) {
    disp.setCursor(rect_text_area.x + 2, rect_text_area.y + 1);
    disp.setTextColor(TFT_WHITE, TFT_BLUE);
    disp.printf("Count:%d ", c);
  }
}

void startCalibration(void) {
  updateCalibration(120, true);
}

void initCalibrDisplay(void) {
  int32_t w = disp.width();
  int32_t h = disp.height();
  if (w < h) {
    disp.setRotation(disp.getRotation() ^ 1);
    w = disp.width();
    h = disp.height();
  }
  int32_t graph_area_h = ((h - 8) / 18) * 18;
  int32_t text_area_h = h - graph_area_h;
  float fontsize = text_area_h / 8;
  disp.setTextSize(fontsize);

  rect_graph_area = { 0, 0, w, graph_area_h };
  rect_text_area = {0, graph_area_h, w, text_area_h };
}

void makeCalibrStep(void) {
  int32_t sec = millis() / 1000;
  if (prev_sec != sec) {
    prev_sec = sec;
    //M5_LOGI("sec:%d  frame:%d", sec, frame_count);
    frame_count = 0;
    if (calib_countdown) {
      updateCalibration(calib_countdown - 1);
    }
    if ((sec & 7) == 0) { // prevent WDT.
      vTaskDelay(1);
    }
  }
}

#endif
