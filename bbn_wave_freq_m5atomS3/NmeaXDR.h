#ifndef NmeaXDR_h
#define NmeaXDR_h

/*
  XDR - Transducer Measurement
  https://gpsd.gitlab.io/gpsd/NMEA.html#_xdr_transducer_measurement
  https://www.eye4software.com/hydromagic/documentation/articles-and-howtos/handling-nmea0183-xdr/

  Format: $--XDR,a,x.x,a,c--c, ..... *hh<CR><LF>
  Example:

  $HCXDR,A,171,D,PITCH,A,-37,D,ROLL,G,367,,MAGX,G,2420,,MAGY,G,-8984,,MAGZ*41
  $SDXDR,C,23.15,C,WTHI*70

  Transducer Types:
     A - Angular displacement
     C - Temperature
     D - Depth
     F - Frequency
     H - Humidity
     N - Force
     P - Pressure
     R - Flow
     B - Absolute humidity
     G - Generic
     I - Current
     L - Salinity
     S - Switch, valve
     T - Tachometer
     U - Voltage
     V - Volume
     could be more

  Unit of measurement                                     
     "" - could be empty!                                 
     A - Amperes                                          
     B - Bars | Binary                                    
     C - Celsius                                          
     D - Degrees                                          
     H - Hertz                                            
     I - liters/second                                    
     K - Kelvin | Density, kg/m3 kilogram per cubic metre 
     M - Meters | Cubic Meters (m3)                       
     N - Newton                                           
     P - Percent of full range | Pascal                   
     R - RPM                                              
     S - Parts per thousand                               
     V - Volts                                            
     could be more
*/

/*
  Bareboat Necessities Sensors NMEA-0183 XDR Sentences:

  NMEA-0183 Sender 
    BB

  Transducer name suffix:
    1 - method #1
    2 - method #2
      
  Heave (vertical displacement)
    Transducer type: D (Depth)
    Unit of measurement: M (meters)
    Transducer name prefix:
      DHI - displacement max
      DLO - displacement min
      DAV - displacement average (bias)
      DRT - displacement in real time
      DRG - displacement range (wave height)

  Vertical acceleration (from observer point of view):
    Transducer type: N (Force)
    Unit of measurement: P - Percent of g (accel of free fall)
    Transducer name:
      AHI - vertical acceleration max
      ALO - vertical acceleration min
      ART - vertical acceleration in real time
      ARG - vertical acceleration range 
      ABI - vertical acceleration sensor bias 
      
  Wave frequency (from observer point of view):
    Transducer type: F (Frequency)
    Unit of measurement: H (Hertz)
    Transducer name:
      FHI - frequency max
      FLO - frequency min
      FRT - frequency in real time
      FAV - frequency average

  IMU Sample rate:
    Transducer type: F (Frequency)
    Unit of measurement: H (Hertz)
    Transducer name:
      SRT - sample rate

  Examples:

  Accel:
  $BBXDR,N,100.0300,P,ARG1*NN
  $BBXDR,N,-0.001,P,ABI1*NN

  Freq:
  $BBXDR,F,0.2450,H,FAV1*NN
  $BBXDR,F,0.2500,H,FRT1*NN

  Heave:
  $BBXDR,D,0.1000,M,DHI2*NN
  $BBXDR,D,-0.1000,M,DLO2*NN
  $BBXDR,D,0.2000,M,DRG1*NN
  $BBXDR,D,0.0,M,DAV1*NN
  $BBXDR,D,-0.0030,M,DRT2*NN

*/

#include "NmeaChecksum.h"

void gen_nmea0183_xdr(const char *nmea_fmt, float value) {
  char nmea_part[82];
  snprintf(nmea_part, 76, nmea_fmt, value);
  int checksum = nmea0183_checksum(nmea_part);
  Serial.printf("%s*%02d\r\n", nmea_part, checksum);
}

#endif
