ARDUINO_HOME=/opt/arduino-1.8.19

CC=g++
CPPFLAGS=-I./..  -I${ARDUINO_HOME}/libraries/ArxTypeTraits/ -I${ARDUINO_HOME}/libraries/ArduinoJson/extras/tests/Helpers/ -I${ARDUINO_DIRECTORIES_DATA}/packages/esp32/hardware/esp32/3.1.2/cores/esp32/

all:	tests

tests:	tests.o
	$(CC) -o tests tests.o

clean:
	rm -f *.o tests tests.exe
