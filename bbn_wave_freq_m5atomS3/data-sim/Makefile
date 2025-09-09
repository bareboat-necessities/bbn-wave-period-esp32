CC = g++

# Base flags
BASEFLAGS = -O3 -std=c++20 -funroll-loops -fno-finite-math-only -I./.. -I/usr/include/eigen3

# Detect platform
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
    CXXFLAGS = $(BASEFLAGS) -march=native
else ifeq ($(UNAME_S),Darwin)   # macOS
    CXXFLAGS = $(BASEFLAGS) -march=native
else
    CXXFLAGS = $(BASEFLAGS)    # Windows / Cygwin / others
endif

LDFLAGS  =

# Targets
TARGETS = waves_sim freq_track sea_reg qmekf_sim w3d_sim

all: $(TARGETS)

waves_sim: waves_sim.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

freq_track: freq_track.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

sea_reg: sea_reg.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

qmekf_sim: qmekf_sim.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

w3d_sim: w3d_sim.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGETS) *.exe *.csv
