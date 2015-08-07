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
#include <assert.h>

#ifdef _WIN32
#include <chrono>
#include <thread>
#endif

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

const int offsets[8][2] = {{-1, 1},{0, 1},{1, 1},
                           {-1, 0},       {1, 0},
                           {-1,-1},{0,-1},{1,-1}};

__constant__ int cuda_offsets[8][2];

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

void step(int * next, int * current, int WIDTH, int HEIGHT) {
    // coordinates of the cell we're currently evaluating
    int x, y;
    // offset index, neighbor coordinates, alive neighbor count
    int i, nx, ny, num_neighbors;

    // write the next board state
    for (y=0; y<HEIGHT; y++) {
        for (x=0; x<WIDTH; x++) {
            // count this cell's alive neighbors
            num_neighbors = 0;
            for (i=0; i<8; i++) {
                // To make the board torroidal, we use modular arithmetic to
                // wrap neighbor coordinates around to the other side of the
                // board if they fall off.
                nx = (x + offsets[i][0] + WIDTH) % WIDTH;
                ny = (y + offsets[i][1] + HEIGHT) % HEIGHT;
                if (current[ny * WIDTH + nx]) {
                    num_neighbors++;
                }
            }

            // apply the Game of Life rules to this cell
            next[y * WIDTH + x] = 0;
            if ((current[y * WIDTH + x] && num_neighbors==2) ||
                    num_neighbors==3) {
                next[y * WIDTH + x] = 1;
            }
        }
    }
}

/* the naive approach is more or less a copy/pasted version of the original algorithm
   which has minor changes to work within a kernel space */
__global__ void naive_compute_gol(int * next, int * board, int WIDTH, int HEIGHT)
{    
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }
    
    int num_neighbors = 0;
    for (int i=0; i<8; i++) {
        int nx = (x + cuda_offsets[i][0] + WIDTH) % WIDTH;
        int ny = (y + cuda_offsets[i][1] + HEIGHT) % HEIGHT;
        if (board[ny * WIDTH + nx]) {
            num_neighbors++;
        }
    }
    
    next[y * WIDTH + x] = 0;
    if ((board[y * WIDTH + x] && num_neighbors==2) ||
        num_neighbors==3) {
        next[y * WIDTH + x] = 1;
    }
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

/* the naive game of life, which uses constant memory for offsets */
void gol_naive_device(int * board, int iterations, int WIDTH, int HEIGHT, int block_width)
{
    int * dev_board;
    int * dev_next;
    int i;
    gpuErrchk(cudaMalloc((void **) &dev_board, sizeof(int) * WIDTH * HEIGHT));
    gpuErrchk(cudaMalloc((void **) &dev_next, sizeof(int) * WIDTH * HEIGHT));
    
    gpuErrchk(cudaMemcpy(dev_board, board, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(cuda_offsets, offsets, sizeof(cuda_offsets) ));

    dim3 threadsPerBlock(block_width, block_width);
    dim3 numBlocks(ceil((float)WIDTH / block_width),ceil((float)HEIGHT / block_width));

    gpuErrchk( cudaDeviceSynchronize() ); /* wait for mem to be copied? */
    for (i = 0; i < iterations; i++) {
        tile_compute_gol_bitmap<<<numBlocks, threadsPerBlock>>>(NULL, dev_next, dev_board, WIDTH, HEIGHT);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() ); /* wait for computation to complete */

        /* swap the two boards to allow memory to already be in the correct location */
        swap_board(&dev_next, &dev_board);
    }

    /* only copy memory from device when the user program wants it (after the iterations) */
    gpuErrchk(cudaMemcpy(board, dev_board, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(dev_board));
    gpuErrchk(cudaFree(dev_next));
}

/* the tiled game of life management function */
void gol_device(int * board, int iterations, int WIDTH, int HEIGHT, int block_width)
{
    int * dev_board;
    int * dev_next;
    int i;
    gpuErrchk(cudaMalloc((void **) &dev_board, sizeof(int) * WIDTH * HEIGHT));
    gpuErrchk(cudaMalloc((void **) &dev_next, sizeof(int) * WIDTH * HEIGHT));
    
    gpuErrchk(cudaMemcpy(dev_board, board, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyHostToDevice));
    /* this will "over" allocate threads, to fully populate shared memory on each block */
    dim3 threadsPerBlock(block_width, block_width);
    dim3 numBlocks(ceil((float)WIDTH / (block_width - 2)),ceil((float)HEIGHT / (block_width - 2)));

    gpuErrchk( cudaDeviceSynchronize() ); /* wait for mem to be copied? */
    for (i = 0; i < iterations; i++) {
        tile_compute_gol_bitmap<<<numBlocks, threadsPerBlock>>>(NULL, dev_next, dev_board, WIDTH, HEIGHT);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() ); /* wait for computation to complete */

        /* swap the two boards to allow memory to already be in the correct location */
        swap_board(&dev_next, &dev_board);
    }

    /* only copy memory from device when the user program wants it (after the iterations) */
    gpuErrchk(cudaMemcpy(board, dev_board, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(dev_board));
    gpuErrchk(cudaFree(dev_next));
}

void animate(int * board, int WIDTH, int HEIGHT, int block_width, int do_delay, int profile) {
	#ifndef _WIN32
	struct timespec delay = {0, 125000000}; // 0.125 seconds
    struct timespec remaining;
	#endif
	
	int iteration = 0;
    while (1) {
        iteration++;
        if (!profile) {
            printf("Iteration: %d\n", iteration);
            print_board(board, WIDTH, HEIGHT);
        }
        gol_device(board, 1, WIDTH, HEIGHT, block_width);
        // We sleep only because textual output is slow and the console needs
        // time to catch up. We don't sleep in the graphical X11 version.
        if (do_delay) {
#ifdef _WIN32
            std::this_thread::sleep_for(std::chrono::milliseconds(125));
#else
            nanosleep(&delay, &remaining);
#endif
        }
        timeout(&globalArgs, iteration);
    }
}

struct DataBlock {
    int HEIGHT;
    int WIDTH;
    int * dev_board;
    int * dev_next;
    cudaEvent_t start, stop;
    int frames;
    float totalTime;
    int profile;
    int block_width;
};


void anim_gpu( uchar4* outputBitmap, DataBlock *d, int ticks ) {
    gpuErrchk( cudaEventRecord( d->start, 0 ) );
    dim3 threadsPerBlock(d->block_width, d->block_width);
    dim3 numBlocks(ceil((float)d->WIDTH / (d->block_width - 2)),ceil((float)d->HEIGHT / (d->block_width - 2)));
    assert(d->block_width == TILE_WIDTH);
    
    tile_compute_gol_bitmap<<<numBlocks, threadsPerBlock>>>(outputBitmap, d->dev_next, d->dev_board, d->WIDTH, d->HEIGHT);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() ); /* wait for computation to complete */
    gpuErrchk( cudaEventRecord( d->stop, 0 ) );
    gpuErrchk( cudaEventSynchronize( d->stop ) );
    float   elapsedTime;
    gpuErrchk( cudaEventElapsedTime( &elapsedTime, d->start, d->stop ) );
    /* swap the two boards to allow memory to already be in the correct location */
    swap_board(&d->dev_next, &d->dev_board);

    d->totalTime += elapsedTime;
    d->frames++;
    if ((d->frames & 0x7F) == 0 && !d->profile) {
        printf( "(%d) Average Time per frame:  %3.1f ms\n",
            d->frames, d->totalTime/d->frames );
    }
    timeout(&globalArgs, d->frames);
}

void anim_exit( DataBlock *d ) {
    gpuErrchk(cudaFree(d->dev_board));
    gpuErrchk(cudaFree(d->dev_next));
    gpuErrchk( cudaEventDestroy( d->start ) );
    gpuErrchk( cudaEventDestroy( d->stop ) );
}

void gpu_gameoflife(int WIDTH, int HEIGHT, int block_width, int * board, int profile)
{
    DataBlock   d;
    GPUAnimBitmap bitmap( WIDTH, HEIGHT, &d );

    d.HEIGHT = HEIGHT;
    d.WIDTH = WIDTH;
    d.frames = 0;
    d.totalTime = 0;
    d.profile = profile;
    d.block_width = block_width;
    
    gpuErrchk( cudaEventCreate( &d.start ) );
    gpuErrchk( cudaEventCreate( &d.stop ) );
    
    gpuErrchk(cudaMalloc((void **) &d.dev_board, sizeof(int) * WIDTH * HEIGHT));
    gpuErrchk(cudaMalloc((void **) &d.dev_next, sizeof(int) * WIDTH * HEIGHT));
    
    gpuErrchk(cudaMemcpy(d.dev_board, board, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyHostToDevice));

    gpuErrchk( cudaDeviceSynchronize() ); /* wait for mem to be copied */
    
    bitmap.anim_and_exit( (void (*)(uchar4*,void*,int))anim_gpu,
        (void (*)(void*))anim_exit );
}

void plain_old(int WIDTH, int HEIGHT, int block_width, int * default_board, int * default_next, int * cuda_board, int profile)
{
    int elements = WIDTH * HEIGHT;
    // Sanity Check CUDA for 10 Steps (each checked)
    for (int i = 0; i < 10; i++) {
        step(default_next, default_board, WIDTH, HEIGHT);
        gol_device(cuda_board, 1, WIDTH, HEIGHT, block_width);
        
        if (!compare_board(cuda_board, default_next, elements)) {
            print_board(cuda_board, WIDTH, HEIGHT);
            print_board(default_next, WIDTH, HEIGHT);
            printf("SANITY: Boards do not match (iteration %d)!\n", i);
            exit(1);
        }

        swap_board(&default_next, &default_board);
    }

    //Sanity Check CUDA with 10 unmonitored (completely on device) steps
    int unmonitored = 10;
    for (int i = 0; i < unmonitored; i++) {
        step(default_next, default_board, WIDTH, HEIGHT);
        swap_board(&default_next, &default_board);
    }
    gol_device(cuda_board, unmonitored, WIDTH, HEIGHT, block_width);
    if (!compare_board(cuda_board, default_board, elements)) {
        print_board(cuda_board, WIDTH, HEIGHT);
        print_board(default_board, WIDTH, HEIGHT);
        printf("SANITY: Boards do not match (after 10 on device iterations)!\n");
        exit(1);
    }

    /* it appears to be sane, run the animation routine */
    animate(cuda_board, WIDTH, HEIGHT, block_width, 1, profile);
}


int main(int argc, char *argv[]) {
    processArgs("gol", argv, argc, &globalArgs);
    
	int elements = globalArgs.width * globalArgs.height;
    
    int * default_board = (int *)malloc(sizeof(int) * elements);
    int * default_next = (int *)malloc(sizeof(int) * elements);

    int * cuda_board = (int *)malloc(sizeof(int) * elements);
    srand(time(NULL));
 
    fill_board(default_board, elements);
    copy_board(cuda_board, default_board, elements);

    switch(globalArgs.mode) {
    case PROFILE_NONE:
        plain_old(globalArgs.width, globalArgs.height, globalArgs.blockwidth, default_board, default_next, cuda_board, globalArgs.profile);
        break;
    case PROFILE_GPU:
        gpu_gameoflife(globalArgs.width, globalArgs.height, globalArgs.blockwidth, cuda_board, globalArgs.profile);
        break;
    case PROFILE_CPU:
        animate(cuda_board, globalArgs.width, globalArgs.height, globalArgs.blockwidth, 0, globalArgs.profile);
        break;
    default:
        printf("Unhandled mode by game of life.\n");
        exit(1);
    }

    /* cleanup no longer needed boards */
    free(default_board);
    free(default_next);   
    free(cuda_board);

    return 0;
}
