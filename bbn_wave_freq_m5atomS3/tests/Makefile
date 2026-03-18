CC=g++
#CPPFLAGS=-O3 -march=native -fsanitize=address -fno-omit-frame-pointer -funroll-loops -fno-finite-math-only -I./.. -I/usr/include/eigen3
CPPFLAGS=-O3 -funroll-loops -fno-finite-math-only -I./.. -I/usr/include/eigen3

.PHONY: all check
all:	tests tests_freq tests_sea adaptive_wave_detrender_test

tests:	tests.o
	$(CC) $(LDFLAGS) -o $@ $^

tests_freq: tests_freq.o
	$(CC) $(LDFLAGS) -o $@ $^

tests_sea: tests_sea.o
	$(CC) $(LDFLAGS) -o $@ $^

adaptive_wave_detrender_test: adaptive_wave_detrender_test.o
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CC) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

.PHONY:	clean adaptive_wave_detrender_check
check: adaptive_wave_detrender_check

adaptive_wave_detrender_check: adaptive_wave_detrender_test
	./adaptive_wave_detrender_test ../data-sim/detrend.csv adaptive_wave_detrender_test_output.csv

clean:
	rm -f *.o *.csv tests tests.exe tests_freq tests_freq.exe tests_sea tests_sea.exe adaptive_wave_detrender_test adaptive_wave_detrender_test.exe
