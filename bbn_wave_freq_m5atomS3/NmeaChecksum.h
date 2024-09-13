#ifndef NmeaChecksum_h
#define NmeaChecksum_h

#define NMEA_END_CHAR_1    '\r'

/*
  NMEA-0183 checksum
 */

uint8_t nmea0183_checksum(const char *sentence);

uint8_t nmea0183_checksum(const char *sentence) {
  const char *n = sentence + 1;
  uint8_t chk = 0;
  /* While current char isn't '*' or sentence ending (newline) */
  while ('*' != *n && NMEA_END_CHAR_1 != *n && '\0' != *n) {
    chk ^= (uint8_t) *n;
    n++;
  }
  return chk;
}

#endif
