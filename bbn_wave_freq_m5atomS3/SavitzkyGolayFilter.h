/*
   Adopted from https://github.com/uutzinger/SavitzkyGolayFilter
  
   Arduino Savitzky Golay Library - Version 1.2.1
   by James Deromedi <jmderomedi@gmail.com>
   
   Version 1.3.0
   by Urs Utzinger <uutzinger@gmail.com>
  
   This Library is licensed under the MIT License
 */

#ifndef SavitzkyGolayFilter_h
#define SavitzkyGolayFilter_h

#include <vector>
#include <cstdint>

#define MAX_WINDOW_SIZE 25

class SavLayFilter {
  public:
    SavLayFilter(int windowSize, int order, int derivative);
    float   update(float newValue);

  private:
    // Ring buffer
    std::vector<float>   _buffer_float;
    int _head;
    bool _isBufferFull;

    // Convolution Table
    std::vector<std::vector<int32_t>> _convolutionTable;
    int _kernelPointer;
    std::vector<int32_t> _mirroredKernel;

    // Variables
    float   _sum_float;
    int _windowSize;
    int _bufferSize;
    int _halfWindowSize;
    int _derivative;
    int _order;
    int _norm;

    void initializeConvolutionTable(int order, int derivative);

    std::vector<std::vector<int32_t>>  _linearSmooth = {
      {    3,    1,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 3
      {    5,    1,    1,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 5
      {    7,    1,    1,    1,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 7
      {    9,    1,    1,    1,    1,    1,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 9
      {   11,    1,    1,    1,    1,    1,    1,    0,    0,    0,    0,    0,    0,    0 }, // Window size 11
      {   13,    1,    1,    1,    1,    1,    1,    1,    0,    0,    0,    0,    0,    0 }, // Window size 13
      {   15,    1,    1,    1,    1,    1,    1,    1,    1,    0,    0,    0,    0,    0 }, // Window size 15
      {   17,    1,    1,    1,    1,    1,    1,    1,    1,    1,    0,    0,    0,    0 }, // Window size 17
      {   19,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    0,    0,    0 }, // Window size 19
      {   21,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    0,    0 }, // Window size 21
      {   23,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    0 }, // Window size 23
      {   25,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1 }, // Window size 25
    };

    std::vector<std::vector<int32_t>> _quadraticSmooth = {
      {    1,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 3
      {   35,   -3,   12,   17,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 5
      {   21,   -2,    3,    6,    7,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 7
      {  231,  -21,   14,   39,   54,   59,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 9
      {  429,  -36,    9,   44,   69,   84,   89,    0,    0,    0,    0,    0,    0,    0 }, // Window size 11
      {   28,   -2,    0,    2,    3,    4,    5,    5,    0,    0,    0,    0,    0,    0 }, // Window size 13
      { 1105,  -78,  -13,   42,   87,  122,  147,  162,  167,    0,    0,    0,    0,    0 }, // Window size 15
      {  323,  -21,   -6,    7,   18,   27,   34,   39,   42,   43,    0,    0,    0,    0 }, // Window size 17
      { 2261, -136,  -51,   24,   89,  144,  189,  224,  249,  264,  269,    0,    0,    0 }, // Window size 19
      { 3059, -171,  -76,    9,   84,  149,  204,  249,  284,  309,  324,  329,    0,    0 }, // Window size 21
      {  805,  -42,  -21,   -2,   15,   30,   43,   54,   63,   70,   75,   78,   79,    0 }, // Window size 23
      { 5175, -253, -138,  -33,   62,  147,  222,  287,  342,  387,  422,  447,  462,  467 }, // Window size 25
    };

    std::vector<std::vector<int32_t>>  _cubicSmooth = {
      {    1,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 3
      {   35,   -3,   12,   17,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 5
      {   21,   -2,    3,    6,    7,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 7
      {  231,  -21,   14,   39,   54,   59,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 9
      {  429,  -36,    9,   44,   69,   84,   89,    0,    0,    0,    0,    0,    0,    0 }, // Window size 11
      {  143,  -11,    0,    9,   16,   21,   24,   25,    0,    0,    0,    0,    0,    0 }, // Window size 13 * hand fix
      { 1105,  -78,  -13,   42,   87,  122,  147,  162,  167,    0,    0,    0,    0,    0 }, // Window size 15
      {  323,  -21,   -6,    7,   18,   27,   34,   39,   42,   43,    0,    0,    0,    0 }, // Window size 17
      { 2261, -136,  -51,   24,   89,  144,  189,  224,  249,  264,  269,    0,    0,    0 }, // Window size 19
      { 3059, -171,  -76,    9,   84,  149,  204,  249,  284,  309,  324,  329,    0,    0 }, // Window size 21
      {  805,  -42,  -21,   -2,   15,   30,   43,   54,   63,   70,   75,   78,   79,    0 }, // Window size 23
      { 5175, -253, -138,  -33,   62,  147,  222,  287,  342,  387,  422,  447,  462,  467 }, // Window size 25
    };

    std::vector<std::vector<int32_t>>  _quarticSmooth = {
      {    1,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 3
      {    1,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 5
      {  231,    5,  -30,   75,  131,    0,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 7
      {  429,   15,  -55,   30,  135,  179,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 9
      {  429,   18,  -45,  -10,   60,  120,  143,    0,    0,    0,    0,    0,    0,    0 },               // Window size 11
      { 2431,  110, -198, -135,  110,  390,  600,  677,    0,    0,    0,    0,    0,    0 },               // Window size 13
      { 46189, 2145, -2860, -2937, -165, 3755, 7500, 10125, 11063,    0,    0,    0,    0,    0 },          // Window size 15
      { 4199,  195, -195, -260, -117,  135,  415,  660,  825,  883,    0,    0,    0,    0 },               // Window size 17
      { 7429,  340, -255, -420, -290,   18,  405,  790, 1110, 1320, 1393,    0,    0,    0 },               // Window size 19
      { 260015, 11628, -6460, -13005, -11220, -3940, 6378, 17655, 28190, 36660, 42120, 44003,    0,    0 }, // Window size 21
      { 2185,   95,  -38,  -95,  -95,  -55,   10,   87,  165,  235,  290,  325,  337,    0 },               // Window size 23
      { 30015, 1265, -345, -1122, -1255, -915, -255,  590, 1503, 2385, 3155, 3750, 4125, 4253 },            // Window size 25
    };
          
    std::vector<std::vector<int32_t>>  _quinticSmooth = {
      {    1,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 3
      {    1,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 5
      {  231,    5,  -30,   75,  131,    0,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 7
      {  429,   15,  -55,   30,  135,  179,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 9
      {  429,   18,  -45,  -10,   60,  120,  143,    0,    0,    0,    0,    0,    0,    0 },               // Window size 11
      { 2431,  110, -198, -135,  110,  390,  600,  677,    0,    0,    0,    0,    0,    0 },               // Window size 13
      { 46189, 2145, -2860, -2937, -165, 3755, 7500, 10125, 11063,    0,    0,    0,    0,    0 },          // Window size 15
      { 4199,  195, -195, -260, -117,  135,  415,  660,  825,  883,    0,    0,    0,    0 },               // Window size 17
      { 7429,  340, -255, -420, -290,   18,  405,  790, 1110, 1320, 1393,    0,    0,    0 },               // Window size 19
      { 260015, 11628, -6460, -13005, -11220, -3940, 6378, 17655, 28190, 36660, 42120, 44003,    0,    0 }, // Window size 21
      { 2185,   95,  -38,  -95,  -95,  -55,   10,   87,  165,  235,  290,  325,  337,    0 },               // Window size 23
      { 30015, 1265, -345, -1122, -1255, -915, -255,  590, 1503, 2385, 3155, 3750, 4125, 4253 },            // Window size 25
    };

    std::vector<std::vector<int32_t>> _sexicSmooth = {
      {    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },                       // Window size 3
      {    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },                       // Window size 5
      {    1,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0 },                       // Window size 7
      { 1287,   -7,   56, -196,  392,  797,    0,    0,    0,    0,    0,    0,    0,    0 },                       // Window size 9
      { 2431,  -28,  161, -308,   28,  784, 1157,    0,    0,    0,    0,    0,    0,    0 },                       // Window size 11
      { 46189, -770, 3388, -3605, -3500, 4550, 14000, 18063,    0,    0,    0,    0,    0,    0 },                  // Window size 13
      { 12597, -260,  910, -476, -1085, -140, 1750, 3500, 4199,    0,    0,    0,    0,    0 },                     // Window size 15
      { 96577, -2275, 6500, -910, -6916, -5215, 3500, 15050, 24500, 28109,    0,    0,    0,    0 },                // Window size 17
      { 37145, -952, 2261,  344, -1918, -2380, -679, 2492, 5978, 8624, 9605,    0,    0,    0 },                    // Window size 19
      { 334305, -9044, 18088, 7021, -11016, -19908, -14364, 3801, 28812, 53508, 71344, 77821,    0,    0 },         // Window size 21
      { 807904, -22610, 38437, 22610, -14131, -39950, -41055, -16338, 27274, 78750, 126175, 159250, 171081,    0 }, // Window size 23
      { 790152, -22559, 32813, 25109, -4280, -30049, -39335, -28458,  -45, 39463, 81920, 119201, 144543, 153505 },  // Window size 25
    };

    std::vector<std::vector<int32_t>>  _linearFirstDerivative = {
      {    2,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 3
      {   10,    2,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 5
      {   28,    3,    2,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 7
      {   60,    4,    3,    2,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 9
      {  111,    5,    4,    3,    2,    1,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 11
      {  183,    6,    5,    4,    3,    2,    1,    0,    0,    0,    0,    0,    0,    0 }, // Window size 13
      {  280,    7,    6,    5,    4,    3,    2,    1,    0,    0,    0,    0,    0,    0 }, // Window size 15
      {  408,    8,    7,    6,    5,    4,    3,    2,    1,    0,    0,    0,    0,    0 }, // Window size 17
      {  571,    9,    8,    7,    6,    5,    4,    3,    2,    1,    0,    0,    0,    0 }, // Window size 19
      {  770,   10,    9,    8,    7,    6,    5,    4,    3,    2,    1,    0,    0,    0 }, // Window size 21
      { 1013,   11,   10,    9,    8,    7,    6,    5,    4,    3,    2,    1,    0,    0 }, // Window size 23
      { 1300,   12,   11,   10,    9,    8,    7,    6,    5,    4,    3,    2,    1,    0 }, // Window size 25
    };

    std::vector<std::vector<int32_t>>  _quadraticFirstDerivative = {
      {    2,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 3
      {   10,    2,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 5
      {   28,    3,    2,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 7
      {   60,    4,    3,    2,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 9
      {  111,    5,    4,    3,    2,    1,    0,    0,    0,    0,    0,    0,    0,    0 }, // Window size 11
      {  183,    6,    5,    4,    3,    2,    1,    0,    0,    0,    0,    0,    0,    0 }, // Window size 13
      {  281,    7,    6,    5,    4,    3,    2,    1,    0,    0,    0,    0,    0,    0 }, // Window size 15
      {  408,    8,    7,    6,    5,    4,    3,    2,    1,    0,    0,    0,    0,    0 }, // Window size 17
      {  571,    9,    8,    7,    6,    5,    4,    3,    2,    1,    0,    0,    0,    0 }, // Window size 19
      {  770,   10,    9,    8,    7,    6,    5,    4,    3,    2,    1,    0,    0,    0 }, // Window size 21
      { 1012,   11,   10,    9,    8,    7,    6,    5,    4,    3,    2,    1,    0,    0 }, // Window size 23
      { 1300,   12,   11,   10,    9,    8,    7,    6,    5,    4,    3,    2,    1,    0 }, // Window size 25
    };

    std::vector<std::vector<int32_t>> _cubicFirstDerivative = {
      {    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },            // Window size 3
      {   12,   -1,    8,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },            // Window size 5
      {  252,  -22,   67,   58,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },            // Window size 7
      { 1188,  -86,  142,  193,  126,    0,    0,    0,    0,    0,    0,    0,    0,    0 },            // Window size 9
      { 5148, -300,  294,  532,  503,  296,    0,    0,    0,    0,    0,    0,    0,    0 },            // Window size 11
      { 24024, -1133,  660, 1578, 1796, 1489,  832,    0,    0,    0,    0,    0,    0,    0 },          // Window size 13
      { 334152, -12922, 4121, 14150, 18334, 17842, 13843, 7506,    0,    0,    0,    0,    0,    0 },    // Window size 15
      { 23256, -748,   98,  643,  930, 1002,  902,  673,  358,    0,    0,    0,    0,    0 },           // Window size 17
      { 255816, -6936,  -68, 4648, 7481, 8700, 8574, 7372, 5363, 2816,    0,    0,    0,    0 },         // Window size 19
      { 230385, -5330, -636, 2744, 4956, 6146, 6460, 6044, 5044, 3606, 1876,    0,    0,    0 },         // Window size 21
      { 197340, -3938, -815, 1518, 3140, 4130, 4567, 4530, 4098, 3350, 2365, 1222,    0,    0 },         // Window size 23
      { 806262, -14012, -3905, 3870, 9525, 13272, 15323, 15890, 15185, 13420, 10807, 7558, 3885,    0 }, // Window size 25
    };

    std::vector<std::vector<int32_t>> _quarticFirstDerivative = {
      {    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },            // Window size 3
      {   12,   -1,    8,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },            // Window size 5
      {  252,  -22,   67,   58,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },            // Window size 7
      { 1188,  -86,  142,  193,  126,    0,    0,    0,    0,    0,    0,    0,    0,    0 },            // Window size 9
      { 5148, -300,  294,  532,  503,  296,    0,    0,    0,    0,    0,    0,    0,    0 },            // Window size 11
      { 24024, -1133,  660, 1578, 1796, 1489,  832,    0,    0,    0,    0,    0,    0,    0 },          // Window size 13
      { 334152, -12922, 4121, 14150, 18334, 17842, 13843, 7506,    0,    0,    0,    0,    0,    0 },    // Window size 15
      { 23256, -748,   98,  643,  930, 1002,  902,  673,  358,    0,    0,    0,    0,    0 },           // Window size 17
      { 255816, -6936,  -68, 4648, 7481, 8700, 8574, 7372, 5363, 2816,    0,    0,    0,    0 },         // Window size 19
      { 230385, -5330, -636, 2744, 4956, 6146, 6460, 6044, 5044, 3606, 1876,    0,    0,    0 },         // Window size 21
      { 197340, -3938, -815, 1518, 3140, 4130, 4567, 4530, 4098, 3350, 2365, 1222,    0,    0 },         // Window size 23
      { 806262, -14012, -3905, 3870, 9525, 13272, 15323, 15890, 15185, 13420, 10807, 7558, 3885,    0 }, // Window size 25
    };

    std::vector<std::vector<int32_t>> _quinticFirstDerivative = {
      {    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },                  // Window size 3
      {    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },                  // Window size 5
      {   60,    1,   -9,   45,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },                  // Window size 7
      { 8580,  254, -1381, 2269, 2879,    0,    0,    0,    0,    0,    0,    0,    0,    0 },                 // Window size 9
      { 17160,  573, -2166, 1249, 3774, 3084,    0,    0,    0,    0,    0,    0,    0,    0 },                // Window size 11
      { 291720, 9647, -27093,   12, 33511, 45741, 31380,    0,    0,    0,    0,    0,    0,    0 },           // Window size 13
      { 784917, 24410, -52907, -20322, 40659, 82997, 87226, 54560,    0,    0,    0,    0,    0,    0 },       // Window size 15
      { 503880, 14404, -24661, -16679, 8671, 32306, 43973, 40483, 23945,    0,    0,    0,    0,    0 },       // Window size 17
      { 576615, 15000, -20576, -18956, -557, 20511, 35544, 40343, 34313, 19562,    0,    0,    0,    0 },      // Window size 19
      { 705458, 16638, -18427, -21087, -7086, 12119, 28820, 38524, 39416, 31824, 17683,    0,    0,    0 },    // Window size 21
      { 744552, 15912, -14264, -19448, -10531, 4246, 18998, 29900, 34895, 33398, 26001, 14180,    0,    0 },   // Window size 23
      { 788010, 15279, -11060, -17633, -12249, -1000, 11569, 22287, 29185, 31325, 28628, 21702, 11670,    0 }, // Window size 25
    };

    std::vector<std::vector<int32_t>> _sexicFirstDerivative = {
      {    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },                  // Window size 3
      {    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },                  // Window size 5
      {   60,    1,   -9,   45,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },                  // Window size 7
      { 8580,  254, -1381, 2269, 2879,    0,    0,    0,    0,    0,    0,    0,    0,    0 },                 // Window size 9
      { 17160,  573, -2166, 1249, 3774, 3084,    0,    0,    0,    0,    0,    0,    0,    0 },                // Window size 11
      { 291720, 9647, -27093,   12, 33511, 45741, 31380,    0,    0,    0,    0,    0,    0,    0 },           // Window size 13
      { 784917, 24410, -52907, -20322, 40659, 82997, 87226, 54560,    0,    0,    0,    0,    0,    0 },       // Window size 15
      { 503880, 14404, -24661, -16679, 8671, 32306, 43973, 40483, 23945,    0,    0,    0,    0,    0 },       // Window size 17
      { 576615, 15000, -20576, -18956, -557, 20511, 35544, 40343, 34313, 19562,    0,    0,    0,    0 },      // Window size 19
      { 705458, 16638, -18427, -21087, -7086, 12119, 28820, 38524, 39416, 31824, 17683,    0,    0,    0 },    // Window size 21
      { 744552, 15912, -14264, -19448, -10531, 4246, 18998, 29900, 34895, 33398, 26001, 14180,    0,    0 },   // Window size 23
      { 788010, 15279, -11060, -17633, -12249, -1000, 11569, 22287, 29185, 31325, 28628, 21702, 11670,    0 }, // Window size 25
    };

    std::vector<std::vector<int32_t>> _quadraticSecondDerivative = {
      {    1,    1,   -2,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },  // Window size 3
      {    7,    2,   -1,   -2,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },  // Window size 5
      {   42,    5,    0,   -3,   -4,    0,    0,    0,    0,    0,    0,    0,    0,    0 },  // Window size 7
      {  462,   28,    7,   -8,  -17,  -20,    0,    0,    0,    0,    0,    0,    0,    0 },  // Window size 9
      {  429,   15,    6,   -1,   -6,   -9,  -10,    0,    0,    0,    0,    0,    0,    0 },  // Window size 11
      { 1001,   22,   11,    2,   -5,  -10,  -13,  -14,    0,    0,    0,    0,    0,    0 },  // Window size 13
      { 6188,   91,   52,   19,   -8,  -29,  -44,  -53,  -56,    0,    0,    0,    0,    0 },  // Window size 15
      { 3876,   40,   25,   12,    1,   -8,  -15,  -20,  -23,  -24,    0,    0,    0,    0 },  // Window size 17
      { 6783,   51,   34,   19,    6,   -5,  -14,  -21,  -26,  -29,  -30,    0,    0,    0 },  // Window size 19
      { 33649,  190,  133,   82,   37,   -2,  -35,  -62,  -83,  -98, -107, -110,    0,    0 }, // Window size 21
      { 17710,   77,   56,   37,   20,    5,   -8,  -19,  -28,  -35,  -40,  -43,  -44,    0 }, // Window size 23
      { 26910,   92,   69,   48,   29,   12,   -3,  -16,  -27,  -36,  -43,  -48,  -51,  -52 }, // Window size 25
    };

    std::vector<std::vector<int32_t>> _cubicSecondDerivative = {
      {    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },  // Window size 3
      {    7,    2,   -1,   -2,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },  // Window size 5
      {   42,    5,    0,   -3,   -4,    0,    0,    0,    0,    0,    0,    0,    0,    0 },  // Window size 7
      {  462,   28,    7,   -8,  -17,  -20,    0,    0,    0,    0,    0,    0,    0,    0 },  // Window size 9
      {  429,   15,    6,   -1,   -6,   -9,  -10,    0,    0,    0,    0,    0,    0,    0 },  // Window size 11
      { 1001,   22,   11,    2,   -5,  -10,  -13,  -14,    0,    0,    0,    0,    0,    0 },  // Window size 13
      { 6188,   91,   52,   19,   -8,  -29,  -44,  -53,  -56,    0,    0,    0,    0,    0 },  // Window size 15
      { 3876,   40,   25,   12,    1,   -8,  -15,  -20,  -23,  -24,    0,    0,    0,    0 },  // Window size 17
      { 6783,   51,   34,   19,    6,   -5,  -14,  -21,  -26,  -29,  -30,    0,    0,    0 },  // Window size 19
      { 33649,  190,  133,   82,   37,   -2,  -35,  -62,  -83,  -98, -107, -110,    0,    0 }, // Window size 21
      { 17710,   77,   56,   37,   20,    5,   -8,  -19,  -28,  -35,  -40,  -43,  -44,    0 }, // Window size 23
      { 26910,   92,   69,   48,   29,   12,   -3,  -16,  -27,  -36,  -43,  -48,  -51,  -52 }, // Window size 25
    };

    std::vector<std::vector<int32_t>> _quarticSecondDerivative = {
      {    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 3
      {   12,   -1,   16,  -30,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 5
      {  132,  -13,   67,  -19,  -70,    0,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 7
      { 1716, -126,  371,  151, -211, -370,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 9
      { 1716,  -90,  174,  146,    1, -136, -190,    0,    0,    0,    0,    0,    0,    0 },               // Window size 11
      { 58344, -2211, 2970, 3504, 1614, -971, -3016, -3780,    0,    0,    0,    0,    0,    0 },           // Window size 13
      { 732438, -20503, 19558, 29399, 21048, 4347, -13050, -25675, -30248,    0,    0,    0,    0,    0 },  // Window size 15
      { 100776, -2132, 1443, 2691, 2405, 1256, -207, -1557, -2489, -2820,    0,    0,    0,    0 },         // Window size 17
      { 788777, -12881, 6044, 14136, 14622, 10300, 3536, -3733, -10001, -14192, -15661,    0,    0,    0 }, // Window size 19
      { 708400, -9100, 2800, 8621, 9972, 8272, 4752,  452, -3775, -7268, -9553, -10346,    0,    0 },       // Window size 21
      { 755136, -7759, 1384, 6310, 8023, 7421, 5297, 2337, -877, -3870, -6272, -7819, -8352,    0 },        // Window size 23
      { 751130, -6265,  454, 4348, 6029, 6049, 4903, 3027,  800, -1459, -3487, -5081, -6096, -6444 },       // Window size 25
    };

    std::vector<std::vector<int32_t>> _quinticSecondDerivative = {
      {    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 3
      {    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 5
      {  132,  -13,   67,  -19,  -70,    0,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 7
      { 1716, -126,  371,  151, -211, -370,    0,    0,    0,    0,    0,    0,    0,    0 },               // Window size 9
      { 1716,  -90,  174,  146,    1, -136, -190,    0,    0,    0,    0,    0,    0,    0 },               // Window size 11
      { 58344, -2211, 2970, 3504, 1614, -971, -3016, -3780,    0,    0,    0,    0,    0,    0 },           // Window size 13
      { 732438, -20503, 19558, 29399, 21048, 4347, -13050, -25675, -30248,    0,    0,    0,    0,    0 },  // Window size 15
      { 100776, -2132, 1443, 2691, 2405, 1256, -207, -1557, -2489, -2820,    0,    0,    0,    0 },         // Window size 17
      { 788777, -12881, 6044, 14136, 14622, 10300, 3536, -3733, -10001, -14192, -15661,    0,    0,    0 }, // Window size 19
      { 708400, -9100, 2800, 8621, 9972, 8272, 4752,  452, -3775, -7268, -9553, -10346,    0,    0 },       // Window size 21
      { 755136, -7759, 1384, 6310, 8023, 7421, 5297, 2337, -877, -3870, -6272, -7819, -8352,    0 },        // Window size 23
      { 751130, -6265,  454, 4348, 6029, 6049, 4903, 3027,  800, -1459, -3487, -5081, -6096, -6444 },       // Window size 25
    };

    std::vector<std::vector<int32_t>> _sexicSecondDerivative = {
      {    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },                   // Window size 3
      {    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 },                   // Window size 5
      {  180,    2,  -27,  270, -490,    0,    0,    0,    0,    0,    0,    0,    0,    0 },                   // Window size 7
      { 386100, 11014, -83822, 250477, -37634, -280070,    0,    0,    0,    0,    0,    0,    0,    0 },       // Window size 9
      { 645056, 20268, -107711, 159475, 130215, -94403, -215688,    0,    0,    0,    0,    0,    0,    0 },    // Window size 11
      { 808684, 23489, -94171, 68253, 128180, 40676, -91017, -150820,    0,    0,    0,    0,    0,    0 },     // Window size 13
      { 748624, 18922, -59766, 14711, 70594, 59380,  605, -61101, -86690,    0,    0,    0,    0,    0 },       // Window size 15
      { 699673, 15045, -38483, -3702, 35352, 45455, 25253, -10236, -41675, -54018,    0,    0,    0,    0 },    // Window size 17
      { 810484, 14721, -31065, -11239, 19958, 36621, 32138, 11534, -14788, -35919, -43922,    0,    0,    0 },  // Window size 19
      { 632602, 9705, -17114, -9945, 6371, 18417, 21012, 14448, 2152, -11227, -21305, -25028,    0,    0 },     // Window size 21
      { 805880, 10477, -15575, -12023, 1741, 14292, 20192, 18253, 10024, -1485, -12849, -21041, -24012,    0 }, // Window size 23
      { 770783, 8533, -10756, -10175, -1516, 7804, 13769, 14889, 11431, 4750, -3274, -10700, -15885, -17740 },  // Window size 25
    };
};

SavLayFilter::SavLayFilter(int windowSize, int order, int derivative):
  _buffer_float(windowSize), _derivative(derivative), 
  _bufferSize(windowSize), _windowSize(windowSize),
  _order(order), _head(0), _isBufferFull(false) {

  if (_windowSize > MAX_WINDOW_SIZE) { 
    _windowSize = MAX_WINDOW_SIZE; 
    _buffer_float.resize(_windowSize); // Resize buffer to MAX_WINDOW_SIZE
  }

  _halfWindowSize = windowSize / 2;
  _kernelPointer = (windowSize - 3) / 2;

  if (_kernelPointer > 11) { _kernelPointer = 11; }

  initializeConvolutionTable(order, derivative);
  _norm = _convolutionTable[_kernelPointer][0];

  // Create the full mirrored kernel
  _mirroredKernel.resize(_windowSize);
  for (int i = 0; i < _halfWindowSize; i++) {
    _mirroredKernel[i] = _convolutionTable[_kernelPointer][i + 1];
    if (_derivative % 2 == 1) {
      _mirroredKernel[_windowSize - 1 - i] = -_convolutionTable[_kernelPointer][i + 1];
    } else {
      _mirroredKernel[_windowSize - 1 - i] = _convolutionTable[_kernelPointer][i + 1];
    }
  }
  _mirroredKernel[_halfWindowSize] = _convolutionTable[_kernelPointer][_halfWindowSize + 1];
}

void SavLayFilter::initializeConvolutionTable(int order, int derivative) {
  if (order == 1) {
    if (derivative == 0) {
      _convolutionTable = _linearSmooth;
    } else if (derivative == 1) {
      _convolutionTable = _linearFirstDerivative;
    }
  } else if (order == 2) {
    if (derivative == 0) {
      _convolutionTable = _quadraticSmooth;
    } else if (derivative == 1) {
      _convolutionTable = _quadraticFirstDerivative;
    } else if (derivative == 2) {
      _convolutionTable = _quadraticSecondDerivative;
    }
  } else if (order == 3) {
    if (derivative == 0) {
      _convolutionTable = _cubicSmooth;
    } else if (derivative == 1) {
      _convolutionTable = _cubicFirstDerivative;
    } else if (derivative == 2) {
      _convolutionTable = _cubicSecondDerivative;
    }
  } else if (order == 4) {
    if (derivative == 0) {
      _convolutionTable = _quarticSmooth;
    } else if (derivative == 1) {
      _convolutionTable = _quarticFirstDerivative;
    } else if (derivative == 2) {
      _convolutionTable = _quarticSecondDerivative;
    }
  } else if (order == 5) {
    if (derivative == 0) {
      _convolutionTable = _quinticSmooth;
    } else if (derivative == 1) {
      _convolutionTable = _quinticFirstDerivative;
    } else if (derivative == 2) {
      _convolutionTable = _quinticSecondDerivative;
    }
  }
}

float SavLayFilter::update(float newValue) {
  // Circular buffer logic
  _buffer_float[_head] = newValue;
  _head = (_head + 1) % _bufferSize;
  if (!_isBufferFull) {
    // Return new value until buffer is full
    if (_head == 0) {
      _isBufferFull = true;
    }
    return newValue;
  } else {    
    // Convolve
    _sum_float = 0;
    for (int i = 0; i < _windowSize; i++) {
      int _index = (_head + i) % _bufferSize;
      _sum_float += _buffer_float[_index] * (float)_mirroredKernel[i];
    }
    // Normalize
    _sum_float /= (float)_norm;
    return _sum_float;
  }
}

#endif 
