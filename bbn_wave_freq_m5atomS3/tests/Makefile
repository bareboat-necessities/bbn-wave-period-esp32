CC=g++
CPPFLAGS=-O3 -march=native -fsanitize-address -fno-omit-frame-pointer -funroll-loops -fno-finite-math-only -I./.. -I/usr/include/eigen3

.PHONY: all
all:	tests tests_freq tests_sea

tests:	tests.o
	$(CC) $(LDFLAGS) -o $@ $^

tests_freq: tests_freq.o
	$(CC) $(LDFLAGS) -o $@ $^

tests_sea: tests_sea.o
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CC) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

.PHONY:	clean
clean:
	rm -f *.o *.csv tests tests.exe tests_freq tests_freq.exe tests_sea tests_sea.exe
