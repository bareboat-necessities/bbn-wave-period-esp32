ARDUINO_HOME=/opt/arduino-1.8.19

CC=g++
CPPFLAGS=-O3 -ffast-math -funroll-loops -I./..  -I${ARDUINO_HOME}/libraries/ArxTypeTraits/ -I${ARDUINO_HOME}/libraries/ArduinoJson/extras/tests/Helpers/ -I${ARDUINO_DIRECTORIES_DATA}/packages/esp32/hardware/esp32/3.1.2/cores/esp32/

.PHONY: all
all:	tests tests_freq

tests:	tests.o
	$(CC) $(LDFLAGS) -o $@ $^

tests_freq: tests_freq.o
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CC) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

.PHONY:	clean
clean:
	rm -f *.o tests tests.exe tests_freq tests_freq.exe
