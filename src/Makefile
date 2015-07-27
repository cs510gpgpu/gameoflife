NVCC=nvcc
CFLAGS=-Wall -Werror
NVFLAGS=-g --compiler-options "${CFLAGS}"
OBJECTS=gol.o common.o

all : gol

gol : $(OBJECTS) 
	$(NVCC) $(NVFLAGS) $(OBJECTS) -lGL -lGLU -lglut -o $@

-include $(OBJECTS:.o=.d)

%.o : %.cu
	$(NVCC) -M $(NVFLAGS) $< > $*.d
	$(NVCC) -c $(NVFLAGS) $< -o $@

clean : 
	find -iregex '.*\.\(o\|d\)$$' -exec rm {} \;
	rm gol
