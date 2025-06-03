#pragma once

/*
  Copyright 2025, Mikhail Grushinskiy

  Extended Kalman filter for wave direction estimation.

  Wave direction is given by the plane in which we observe oscillations of horizontal acceleration.
  In case of trochoidal wave model those oscillations are harmonic.

  This model assumes x, y axis acceleration measurements have constant biases and Gaussian noise.
  True x, y accelerations without bias and noise are harmonic and have same phase. Phase is unknown and estimated by the filter.
  Frequency is considered known and is a parameter on each step.

  See details in:  https://github.com/bareboat-necessities/bbn-wave-period-esp32/issues/30#issuecomment-2931856187

*/
