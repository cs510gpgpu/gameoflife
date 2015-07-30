NVCC=nvcc
CFLAGS=-Wall -Werror
NVFLAGS=-g --compiler-options "${CFLAGS}"
OBJECTS=gol.o common.o
RIPOBJS=ripple.o common.o
RIPCPUOBJS=ripple_cpu.o common.o

all : gol ripple ripple_cpu

gol : $(OBJECTS) 
	$(NVCC) $(NVFLAGS) $(OBJECTS) -lGL -lGLU -lglut -o $@

-include %(OBJECTS:.o=.d)

ripple : $(RIPOBJS)
	$(NVCC) $(NVFLAGS) $(RIPOBJS) -lGL -lGLU -lglut -o $@

-include $(RIPOBJS:.o=.d)

ripple_cpu : $(RIPCPUOBJS)
	$(NVCC) $(NVFLAGS) $(RIPCPUOBJS) -lGL -lGLU -lglut -o $@

-include $(RIPCPUOBJS:.o=.d)

%.o : %.cu
	$(NVCC) -M $(NVFLAGS) $< > $*.d
	$(NVCC) -c $(NVFLAGS) $< -o $@


test: gol
	__GL_SYNC_TO_VBLANK=0 ./gol

clean : 
	find -iregex '.*\.\(o\|d\)$$' -exec rm {} \;
	rm gol
	rm ripple
