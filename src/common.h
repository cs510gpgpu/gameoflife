#ifndef COMMON_H
#define COMMON_H

enum MODES {
    PROFILE_NONE,
    PROFILE_GPU,
    PROFILE_CPU,    
};

typedef struct Args_t {
    MODES mode;
    int height, width, blockwidth, profile, timeout;
} * Args;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
int processArgs(const char * progname, char ** argv, int argc, Args args);
void timeout(Args args, int frames);

#endif
