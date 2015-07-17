NVCC=/usr/bin/nvcc
CFLAGS=-Wall -Werror
NVFLAGS=-g --compiler-options "${CFLAGS}" -v
OBJECTS=gol.o

gol : $(OBJECTS) 
	$(NVCC) $(NVFLAGS) $(OBJECTS) -o $@

%.d : %.cu
	$(NVCC) -M $(NVFLAGS) $< > $*.d

-include $(OBJECTS:.o=.d)

%.o : %.cu
	$(NVCC) -c $(NVFLAGS) $< -o $@

clean : 
	rm -f *.o *.d gol
