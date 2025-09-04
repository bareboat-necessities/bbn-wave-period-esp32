CC = g++

# Base flags
BASEFLAGS = -O3 -funroll-loops -fno-finite-math-only -I./.. -I/usr/include/eigen3

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
TARGET   = waves_sim
SRC      = waves_sim.cpp
OBJ      = $(SRC:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET) $(TARGET).exe *.csv
