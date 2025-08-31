CC=g++
CPPFLAGS=-O3 -march=native -funroll-loops -fno-finite-math-only -I./.. -I/usr/include/eigen3

.PHONY:	all
all:	fenton_plots wave_dir_plots

fenton_plots:	fenton_plots.o
	$(CC) -o fenton_plots fenton_plots.o

wave_dir_plots:	wave_dir_plots.o
	$(CC) -o wave_dir_plots wave_dir_plots.o

.PHONY:	clean
clean:
	rm -f *.o fenton_plots fenton_plots.exe wave_dir_plots wave_dir_plots.exe
