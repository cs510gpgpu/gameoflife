// Significant portions shamelessly stolen from Kandrot & Sanders CUDA by example

#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h> // for printf
#include <time.h> // for nanosleep
#include "common.h"
#include "nv/gpu_anim.h"
#include "nv/cpu_anim.h"

#ifdef _WIN32
#include <chrono>
#include <thread>
#endif

struct GPUDataBlock {
    int HEIGHT;
    int WIDTH;
    int block_width;
};

__global__ void compute_ripple_bitmap(uchar4* bitmap, int ticks, int WIDTH, int HEIGHT)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + y * blockDim.x * gridDim.x;

    float fx = x - WIDTH/2;
    float fy = y - HEIGHT/2;
    float d = sqrtf(fx * fx + fy * fy);
    unsigned char grey = (unsigned char) (128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / 
                                                            (d/10.0f + 1.0f));

    bitmap[offset].x = grey;
    bitmap[offset].y = grey;
    bitmap[offset].z = grey;
    bitmap[offset].w = 255;
}

void generate_frame(uchar4 * bitmap, GPUDataBlock * d, int ticks) {
    dim3 grids(d->WIDTH/d->block_width, d->HEIGHT/d->block_width);
    dim3 threads(d->block_width, d->block_width);
    compute_ripple_bitmap<<<grids, threads>>>(bitmap, ticks, d->WIDTH, d->HEIGHT);
}

struct CPUDataBlock {
    uchar4 *dev_bitmap;
    CPUAnimBitmap *bitmap;
    int HEIGHT;
    int WIDTH;
    int block_width;
};

void generate_frame_cpu(CPUDataBlock * d, int ticks) {
    dim3 grids(d->WIDTH/d->block_width, d->HEIGHT/d->block_width);
    dim3 threads(d->block_width, d->block_width);
    compute_ripple_bitmap<<<grids, threads>>>(d->dev_bitmap, ticks, d->WIDTH, d->HEIGHT);

    gpuErrchk(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost));
}

void cleanup_cpu(CPUDataBlock *d) {
    cudaFree(d->dev_bitmap);
}

int main(int argc, char **argv) {    
    int WIDTH = 1024;
    int HEIGHT = 768;
    int profile = 0;
    int block_width = 32;
    MODES mode = PROFILE_NONE;
    processArgs("ripple", argv, argc, &mode, &HEIGHT, &WIDTH, &block_width, &profile);

    switch(mode) {
    case PROFILE_NONE:
        printf("Set a profile mode. \"None\" is unimplemented.\n");
        break;
    case PROFILE_GPU:
        {
            GPUDataBlock data;            
            data.HEIGHT = HEIGHT;
            data.WIDTH = WIDTH;
            data.block_width = block_width;
        	GPUAnimBitmap bitmap(data.WIDTH, data.HEIGHT, &data);
            bitmap.anim_and_exit((void (*)(uchar4*,void*,int))generate_frame, NULL);
        }
        break;
    case PROFILE_CPU:
        {
            CPUDataBlock data;
            data.HEIGHT = HEIGHT;
            data.WIDTH = WIDTH;
            data.block_width = block_width;
            CPUAnimBitmap bitmap(data.WIDTH, data.HEIGHT, &data);
            data.bitmap = &bitmap;
            gpuErrchk(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));
            bitmap.anim_and_exit((void (*)(void*,int))generate_frame_cpu, (void(*)(void*))cleanup_cpu);
        }
        break;
    default:
        printf("Unhandled mode by ripple.\n");
        exit(1);
    }
}
