NVCC=nvcc
CFLAGS=-Wall -Werror
NVFLAGS=-g --compiler-options "${CFLAGS}"
OBJECTS=gol.o

gol : $(OBJECTS) 
	$(NVCC) $(NVFLAGS) $(OBJECTS) -o $@

deps:
	apt-get install \
		build-essential \
		nvidia-cuda-toolkit \
		\

%.d : %.cu
	$(NVCC) -M $(NVFLAGS) $< > $*.d

-include $(OBJECTS:.o=.d)

%.o : %.cu
	$(NVCC) -c $(NVFLAGS) $< -o $@

clean : 
	rm -f *.o *.d gol
