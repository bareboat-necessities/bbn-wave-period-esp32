CC = g++

# Base flags
BASEFLAGS = -O3 -std=c++20 -funroll-loops -fno-finite-math-only -I./.. -I/usr/include/eigen3

# Detect platform
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
    CPPFLAGS = $(BASEFLAGS) -march=native
else ifeq ($(UNAME_S),Darwin)   # macOS
    CPPFLAGS = $(BASEFLAGS) -march=native
else
    CPPFLAGS = $(BASEFLAGS)    # Windows / Cygwin / others
endif

LDFLAGS  =

# Targets
TARGETS = waves_sim freq_track

all: $(TARGETS)

waves_sim: waves_sim.o
	$(CC) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

freq_track: freq_track.o
	$(CC) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGETS) *.exe *.csv
