NVCC=nvcc
CFLAGS=-Wall -Werror
NVFLAGS=-lineinfo -use_fast_math  -g --compiler-options "${CFLAGS}"
OBJECTS=gol.o common.o argtable3.o
RIPOBJS=ripple.o common.o argtable3.o
FLAMEOBJS=flames.o common.o argtable3.o

all : gol ripple flames

gol : $(OBJECTS) 
	$(NVCC) $(NVFLAGS) $(OBJECTS) -lGL -lGLU -lglut -o $@

-include %(OBJECTS:.o=.d)

ripple : $(RIPOBJS)
	$(NVCC) $(NVFLAGS) $(RIPOBJS) -lGL -lGLU -lglut -o $@

-include $(RIPOBJS:.o=.d)

flames : $(FLAMEOBJS)
	$(NVCC) $(NVFLAGS) $(FLAMEOBJS) -lGL -lGLU -lglut -o $@

-include $(FLAMEOBJS:.o=.d)

%.o : %.cu
	$(NVCC) -M $(NVFLAGS) $< > $*.d
	$(NVCC) -c $(NVFLAGS) $< -o $@

test: gol
	__GL_SYNC_TO_VBLANK=0 ./gol

clean : 
	find -iregex '.*\.\(o\|d\)$$' -exec rm {} \;
	rm -f gol
	rm -f ripple
	rm -f flames
