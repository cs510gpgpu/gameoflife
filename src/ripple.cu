// Significant portions shamelessly stolen from Kandrot & Sanders CUDA by example

#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h> // for printf
#include <time.h> // for nanosleep
#include "common.h"
#include "nv/gpu_anim.h"

#ifdef _WIN32
#include <chrono>
#include <thread>
#endif
/* assuring that any block size will be divisible by warps size */
#define THREADS_IN_WARP 32
#define TILE_WIDTH ((THREADS_IN_WARP) - 2)

#define SCREEN_WIDTH 1024
#define SCREEN_HEIGHT 768

#define TILE_DIM 32

__global__ void compute_ripple_bitmap(uchar4* bitmap, int ticks)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + y * blockDim.x * gridDim.x;

    float fx = x - SCREEN_WIDTH/2;
    float fy = y - SCREEN_HEIGHT/2;
    float d = sqrtf(fx * fx + fy * fy);
    unsigned char grey = (unsigned char) (128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / 
                                                            (d/10.0f + 1.0f));

    bitmap[offset].x = grey;
    bitmap[offset].y = grey;
    bitmap[offset].z = grey;
    bitmap[offset].w = 255;
}

void generate_frame(uchar4 * bitmap, void *, int ticks) {
    dim3 grids(SCREEN_WIDTH/TILE_DIM, SCREEN_HEIGHT/TILE_DIM);
    dim3 threads(TILE_DIM, TILE_DIM);
    compute_ripple_bitmap<<<grids, threads>>>(bitmap, ticks);
}

int main(int argc, char **argv) {    
    GPUAnimBitmap bitmap(SCREEN_WIDTH, SCREEN_HEIGHT, NULL);
    bitmap.anim_and_exit((void (*)(uchar4*,void*,int))generate_frame, NULL);
}

