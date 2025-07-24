CC=g++
CPPFLAGS=-I./.. -I/usr/include/eigen3

fenton_plots:	fenton_plots.o
	$(CC) -o fenton_plots fenton_plots.o

wave_dir_plots:	wave_dir_plots.o
	$(CC) -o wave_dir_plots wave_dir_plots.o

clean:
	rm -f *.o fenton_plots fenton_plots.exe wave_dir_plots wave_dir_plots.exe
