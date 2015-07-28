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


test: gol
	__GL_SYNC_TO_VBLANK=0 ./gol

clean : 
	find -iregex '.*\.\(o\|d\)$$' -exec rm {} \;
	rm gol
