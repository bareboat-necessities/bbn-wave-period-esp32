#ifndef NmeaChecksum_h
#define NmeaChecksum_h

/*
  NMEA-0183 checksum
 */

int nmea0183_checksum(char *nmea_data_before_asterisk);

int nmea0183_checksum(char *nmea_data_before_asterisk) {
    int crc = 0;
    int i;
    // the first $ sign 
    for (i = 1; i < strlen(nmea_data); i ++) {
        crc ^= nmea_data[i];
    }
    return crc;
}

#endif
