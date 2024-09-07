#ifndef WavesCategories_h
#define WavesCategories_h

/*
  WMO Sea State codes
  Height (in meters_):
  Calm (glassy)   0    -  0.05  No waves breaking on beach                                                  0
  Calm (rippled)  0.05 -  0.1   No waves breaking on beach                                                  1
  Smooth          0.1  -  0.5   Slight waves breaking on beach                                              2
  Slight          0.5  -  1.25  Waves rock buoys and small craft                                            3
  Moderate        1.25 -  2.5   Sea becoming furrowed                                                       4
  Rough           2.5  -  4     Sea deeply furrowed                                                         5
  Very rough      4    -  6     Sea much disturbed with rollers having steep fronts                         6
  High            6    -  9     Sea much disturbed with rollers having steep fronts (damage to foreshore)   7
  Very high       9    - 14     Towering seas                                                               8
  Phenomenal     14    -  more  Precipitous seas (experienced only in cyclones)                             9
*/

typedef enum {
  NA = -1,
  GLASSY_SEA = 0,
  CALM_SEA = 1,
  SMOOTH_SEA = 2,
  SLIGHT_WAVES = 3,
  MODERATE_WAVES = 4,
  ROUGH_SEA = 5,
  VERY_ROUGH_SEA = 6,
  HIGH_WAVES = 7,
  VERY_HIGH_WAVES = 8,
  PHENOMENAL_WAVES = 9,
} wave_category_WMO_sea_state_code;

#endif
