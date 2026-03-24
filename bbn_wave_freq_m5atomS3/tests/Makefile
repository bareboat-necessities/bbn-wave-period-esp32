CC=g++
#CPPFLAGS=-O3 -march=native -fsanitize=address -fno-omit-frame-pointer -funroll-loops -fno-finite-math-only -I./.. -I/usr/include/eigen3
CPPFLAGS=-O3 -std=c++20 -funroll-loops -fno-finite-math-only -I./.. -I/usr/include/eigen3

.PHONY: all check
all:	tests tests_freq tests_sea adaptive_wave_detrender_test adaptive_wave_detrender_sim_test adaptive_vertical_pii_mahony_test calibrate_imu_test

tests:	tests.o
	$(CC) $(LDFLAGS) -o $@ $^

tests_freq: tests_freq.o
	$(CC) $(LDFLAGS) -o $@ $^

tests_sea: tests_sea.o
	$(CC) $(LDFLAGS) -o $@ $^

adaptive_wave_detrender_test: adaptive_wave_detrender_test.o
	$(CC) $(LDFLAGS) -o $@ $^

adaptive_wave_detrender_sim_test: adaptive_wave_detrender_sim_test.o
	$(CC) $(LDFLAGS) -o $@ $^

adaptive_vertical_pii_mahony_test: adaptive_vertical_pii_mahony_test.o
	$(CC) $(LDFLAGS) -o $@ $^

calibrate_imu_test: calibrate_imu_test.o
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CC) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

.PHONY:	clean adaptive_wave_detrender_check adaptive_wave_detrender_sim_check calibrate_imu_check
check: adaptive_wave_detrender_check adaptive_wave_detrender_sim_check calibrate_imu_check

adaptive_wave_detrender_check: adaptive_wave_detrender_test
	./adaptive_wave_detrender_test ../data-sim/detrend.csv adaptive_wave_detrender_test_output.csv

adaptive_wave_detrender_sim_check: adaptive_wave_detrender_sim_test
	./adaptive_wave_detrender_sim_test adaptive_wave_detrender_sim_output.csv adaptive_wave_detrender_sim_summary.csv

calibrate_imu_check: calibrate_imu_test
	./calibrate_imu_test calibrate_imu_test_output.csv calibrate_imu_test_summary.csv

clean:
	rm -f *.o *.csv tests tests.exe tests_freq tests_freq.exe tests_sea tests_sea.exe adaptive_wave_detrender_test adaptive_wave_detrender_test.exe adaptive_wave_detrender_sim_test adaptive_wave_detrender_sim_test.exe adaptive_vertical_pii_mahony_test adaptive_vertical_pii_mahony_test.exe calibrate_imu_test calibrate_imu_test.exe
