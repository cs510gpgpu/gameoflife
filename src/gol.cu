/* Original Author: Christopher Mitchell <chrism@lclark.edu>
 * Date: 2011-07-15
 * Heavy modification by Mitch Souders <msouders@cs.pdx.edu> as part of a homework assignment
 * to compute the Game Of Life using CUDA.
 */

#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h> // for printf
#include <time.h> // for nanosleep
#include "common.h"
#include "nv/gpu_anim.h"
#include "nv/cpu_anim.h"
#include <assert.h>

#ifdef _WIN32
#include <chrono>
#include <thread>
#endif

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

struct Args_t globalArgs;

void fill_board(int *board, int elements) {
    int i;
    for (i=0; i<elements; i++)
        board[i] = (rand() % 3) >> 1;
		//This fix is nessary since windows rand() produces a number whose least siginificant bit
		//	repeats every 161072 causing patterns to emerge and end the simulation more quickly
		//  on some screen resolutions. 
}

void print_board(int *board, int WIDTH, int HEIGHT) {
    int x, y;
    for (y=0; y<HEIGHT; y++) {
        for (x=0; x<WIDTH; x++) {
            char c = board[y * WIDTH + x] ? '#':' ';
            printf("%c", c);
        }
        printf("\n");
    }
    printf("-----\n");
}

/* simply swaps pointers for two boards when they need to be
   in different locations */
void swap_board(int ** b1, int ** b2)
{
    int * tmp = *b1;
    *b1 = *b2;
    *b2 = tmp;
}

__global__ void tile_compute_gol_bitmap(uchar4* bitmap, int * next, int * board, int WIDTH, int HEIGHT)
{
    __shared__ int tile[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tileX = (blockIdx.x * (blockDim.x - 2)) + (threadIdx.x -1);
    int tileY = (blockIdx.y * (blockDim.y - 2)) + (threadIdx.y -1);

    /* fill the tile board with each item */
    int * target = &tile[threadIdx.x][threadIdx.y];
    *target = board[((tileY + HEIGHT) % HEIGHT) * WIDTH + ((tileX + WIDTH) % WIDTH)];

    if (tx == 0 || ty == 0 || tx >= TILE_WIDTH - 1 || ty >= TILE_WIDTH - 1) {
        return; /* these threads do not contribute to answer */
    } else if (tileX >= WIDTH || tileY >= HEIGHT) {
        return; /* these are threads that extend over the edge since the board is not evenly divided by blocks */
    }
    __syncthreads(); /* all spots in `tile' should be full after sync */
    int num_neighbors = 0;
    /* does this manual loop unrolling improve speed over accessing the
       constants in the constant memory? worth a shot */

#if DEBUG_GPU
    assert(tx > 0 && tx < TILE_WIDTH - 1);
    assert(ty > 0 && ty < TILE_WIDTH - 1);
#endif
    
    num_neighbors += tile[tx +-1][ty + 1];
    num_neighbors += tile[tx + 0][ty + 1];
    num_neighbors += tile[tx + 1][ty + 1];
    num_neighbors += tile[tx +-1][ty + 0];
    num_neighbors += tile[tx + 1][ty + 0];
    num_neighbors += tile[tx +-1][ty +-1];
    num_neighbors += tile[tx + 0][ty +-1];
    num_neighbors += tile[tx + 1][ty +-1];

    int offset = tileY * WIDTH + tileX;
    int * next_target = &next[offset];
#if DEBUG_GPU
    assert(offset >= 0 && offset < HEIGHT * WIDTH);
#endif
    /* compute final result */
    *next_target = ((*target && (num_neighbors == 2)) || num_neighbors == 3);

    if (bitmap) {
        int color = *next_target ? 255 : 0;
        bitmap[offset].x = color;
        bitmap[offset].y = color;
        bitmap[offset].z = color;
        bitmap[offset].w = 255;
    }

}
    
/* returns true if boards are equivalent */
bool compare_board(int * b1, int * b2, int len)
{
    for (int i = 0; i < len; i++) {
        if (b1[i] != b2[i]) {
            return false;
        }
    }
    return true;
}

/* copies a board from src to target */
void copy_board(int * target, int * src, int len)
{
    memcpy(target, src, len * sizeof(int));
}

struct GPUDataBlock {
    int HEIGHT;
    int WIDTH;
    int * board;
    int * dev_board;
    int * dev_next;
    int profile;
    int block_width;
};

struct CPUDataBlock {
    uchar4 * dev_bitmap;
    GPUDataBlock gpu;
    CPUAnimBitmap * bitmap;
};


void generate_frame_gpu( uchar4* outputBitmap, GPUDataBlock *d, int ticks ) {
    static int iterations = 0;
    dim3 threadsPerBlock(d->block_width, d->block_width);
    dim3 numBlocks(ceil((float)d->WIDTH / (d->block_width - 2)),ceil((float)d->HEIGHT / (d->block_width - 2)));
    assert(d->block_width == TILE_WIDTH);
    
    tile_compute_gol_bitmap<<<numBlocks, threadsPerBlock>>>(outputBitmap, d->dev_next, d->dev_board, d->WIDTH, d->HEIGHT);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() ); /* wait for computation to complete */

    /* swap the two boards to allow memory to already be in the correct location */
    swap_board(&d->dev_next, &d->dev_board);

    timeout(&globalArgs, iterations++);
}

void generate_frame_cpu(CPUDataBlock *d, int ticks )
{
    generate_frame_gpu(d->dev_bitmap, &d->gpu, ticks);
    gpuErrchk(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost));
}

void cleanup_gpu(GPUDataBlock * d)
{
    free(d->board);
    gpuErrchk(cudaFree(d->dev_board));
    gpuErrchk(cudaFree(d->dev_next));
}

void cleanup_cpu(CPUDataBlock *d)
{
    cleanup_gpu(&d->gpu);
    gpuErrchk(cudaFree(d->dev_bitmap));
}

void gpu_exit( GPUDataBlock *d )
{
    cleanup_gpu(d);
}

void cpu_exit(CPUDataBlock *d)
{
    cleanup_cpu(d);
}

void init_gpu_datablock(GPUDataBlock * d)
{    
    d->HEIGHT = globalArgs.height;
    d->WIDTH = globalArgs.width;
    d->profile = globalArgs.profile;
    d->block_width = globalArgs.blockwidth;
    d->board = (int *)malloc(sizeof(int) * d->HEIGHT * d->WIDTH);
    
    fill_board(d->board, d->HEIGHT * d->WIDTH);
    
    gpuErrchk(cudaMalloc((void **) &d->dev_board, sizeof(int) * d->WIDTH * d->HEIGHT));
    gpuErrchk(cudaMalloc((void **) &d->dev_next, sizeof(int) * d->WIDTH * d->HEIGHT));
  
    gpuErrchk(cudaMemcpy(d->dev_board, d->board, sizeof(int) * d->WIDTH * d->HEIGHT, cudaMemcpyHostToDevice));
    
    gpuErrchk( cudaDeviceSynchronize() ); /* wait for mem to be copied */
}

void init_cpu_datablock(CPUDataBlock * d)
{
    init_gpu_datablock(&d->gpu);
}

int main(int argc, char *argv[]) {
    processArgs("gol", argv, argc, &globalArgs);
    srand(time(NULL));
 
    switch(globalArgs.mode) {
    case PROFILE_NONE:
        printf("Profile None is unimplemented\n");
        break;
    case PROFILE_GPU:
        {
            GPUDataBlock   d;
            GPUAnimBitmap bitmap( globalArgs.width, globalArgs.height, &d );
            init_gpu_datablock(&d);    
            bitmap.anim_and_exit( (void (*)(uchar4*,void*,int))generate_frame_gpu,
                (void (*)(void*))cleanup_gpu );
        }
        break;
    case PROFILE_CPU:
        {
            CPUDataBlock data;
            init_gpu_datablock(&data.gpu);
            CPUAnimBitmap bitmap(data.gpu.WIDTH, data.gpu.HEIGHT, &data);
            data.bitmap = &bitmap;
            gpuErrchk(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));
            bitmap.anim_and_exit((void (*)(void*,int))generate_frame_cpu, (void(*)(void*))cleanup_cpu);
        }
        break;
    default:
        printf("Unhandled mode by game of life.\n");
        exit(1);
    }

    return 0;
}
