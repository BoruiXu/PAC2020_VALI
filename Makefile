CC      = icpc
CFLAGS  = -std=c++11 -O3 -fopenmp -mavx2 -mfma


all: logVS

logVS: main.o
        $(CC) -o $@ $^ $(CFLAGS)

main.o: main.cpp
        $(CC) -c $(CFLAGS) $<

.PHONY: clean

clean: 
        rm -f *.o
        rm -f logVS
