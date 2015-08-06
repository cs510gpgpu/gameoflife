// Significant portions shamelessly stolen from Kandrot & Sanders CUDA by example

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

#define SAMPLES (2000000)
#define FRACTAL_SIZE 20
#define WARP_SIZE 32

typedef struct Point_t {
    float x;
    float y;
} * Point;

typedef struct Coef_t {
    float a, b,c, d, e, f;
    uchar4 color;
} * Coef;

struct GPUDataBlock {
    Point points;
    Point dev_points;
    int HEIGHT;
    int WIDTH;
    int block_width;
};

typedef struct Fractal_t {
    struct Coef_t coef[FRACTAL_SIZE];
    int n;
} * Fractal;

void PlaneHammersley(struct Point_t *result, int n)
{
    float p, u, v;
    int k, kk;
    for (k=0 ; k<n ; k++) {
        u = 0;
        for (p=0.5, kk=k ; kk ; p*=0.5, kk>>=1) {
            if (kk & 1) {
                u += p;
            }
        }
        v = (k + 0.5) / n;
        result[k].x = u;
        result[k].y = v;
    }
}

/* https://github.com/jameswmccarty/Fractal-Flame-Algorithm-in-C/blob/master/fractal.c */
float randf(float lo, float hi)
{
    return ((lo) + (((hi)-(lo)) * drand48()));
}

int random_bit (void)
{
  return rand() & 01;
}

void contractive_map(Coef c)
{
    float a, b, d, e;
    do {
        do {
            a = drand48 ();
            d = randf(a * a, 1);
            if (random_bit ()) {                
                d = -d;
            }
        } while ((a * a + d * d) > 1);
        do {
            b = drand48 ();
            e = randf (b * b, 1);
            if (random_bit ()) {
                e = -e;
            }
        } while ((b * b + e * e) > 1);
    } while ((a * a + b * b + d * d + e * e) >
        (1 + (a * e - d * b) * (a * e - d * b)));

    c->a = a;
    c->b = b;
    c->c = randf (-1, 1);
    c->d = d;
    c->e = e;
    c->f = randf (-1, 1);
}

void init_fractal(Fractal f, int n)
{
    f->n = FRACTAL_SIZE;
    for (int n = 0; n < f->n; n++) {
        Coef c = &f->coef[n];
        contractive_map(c);
        c->color.x = randf(64,255);
        c->color.y = randf(64,255);
        c->color.z = randf(64,255);
        printf("a %f b %f c %f d %f e %f f %f\n", c->a, c->b, c->c, c->d, c->e, c->f);
    }
}

void clean_fractal(Fractal f)
{
    f->n = 0;
}
 
__device__ float r(Point in)
{
    return sqrt((float)(powf(in->x,2) + powf(in->y,2)));
}

__device__ float theta(Point in)
{
    return atanf((float)in->x / in->y);
}
    
__device__ void v_0(Point out, Point in)
{
    *out = *in;
}

__device__ void v_1(Point out, Point in)
{
    out->x = sinf(in->x);
    out->y = sinf(in->y);
}

__device__ void v_2(Point out, Point in)
{
    out->x = 1.0 / pow(r(in),2) * in->x;
    out->y = 1.0 / pow(r(in),2) * in->y;
}

__device__ void v_3(Point out, Point in)
{
    out->x = in->x * sinf(powf(r(in), 2)) - in->y * cosf(powf(r(in), 2));
    out->y = in->x * cosf(powf(r(in), 2)) - in->y * sinf(powf(r(in), 2));    
}

#if OMEGA_IMPLEMENTED
__device__ void v_13(Point out, Point in)
{
    out->x = powf(r(in), 0.5) * cosf(theta()/2 + omega());
    out->y = powf(r(in), 0.5) * sinf(theta()/2 + omega());
}
#endif

__device__ void v_18(Point out, Point in)
{
    out->x = expf(in->x - 1) * cospif(in->y);
    out->y = expf(in->y - 1) * sinpif(in->y);
}

typedef void (*V_func)(Point out, Point in);

__device__ void v_19(Point out, Point in)
{
    float theta_0 = theta(in);
    out->x = pow(r(in), sin(theta_0)) * cos(theta_0);
    out->x = pow(r(in), sin(theta_0)) * sin(theta_0);
}

__device__ void nextColor(uchar4 * out, Coef coef)
{
#define C(_color_) out->_color_ = (((unsigned int)out-> _color_ + coef->color._color_) / 2)
    C(x);
    C(y);
    C(z);
    out->w = 255;
#undef C
}

__device__ void toSpace(Point out, int x, int y, int width, int height)
{
    out->x = (x - width / 2.0) / (width / 2.0);
    out->y = (y - height / 2.0) / (height / 2.0);
}

void fromSpace2(int * x, int * y, Point in, int width, int height)
{
    *x = in->x * (width / 2.0) + width / 2.0;
    *y = in->y * (height / 2.0) + height / 2.0;
}


__device__ void fromSpace(int * x, int * y, Point in, int width, int height)
{
    *x = in->x * (width / 2.0) + width / 2.0;
    *y = in->y * (height / 2.0) + height / 2.0;
}

__constant__ struct Fractal_t cuda_fractal;
__global__ void compute_flames(uchar4* bitmap, struct Point_t *points, int pt_offset, int fn_offset, int i, int WIDTH, int HEIGHT)
{
    if (threadIdx.x + blockIdx.x * blockDim.x >= SAMPLES) {
        return;
    }
    Point p = &points[(threadIdx.x + blockIdx.x * blockDim.x + pt_offset) % SAMPLES];
    struct Point_t old = *p;
    Coef c = &cuda_fractal.coef[i];

    const V_func fn[] = {v_0, v_1, v_2, v_3, v_18, v_19};
    int fn_cnt = (sizeof(fn) / sizeof(V_func));
    int fn_idx = ((threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE + fn_offset) % fn_cnt;
    
    p->x = c->a * old.x + c->b * old.y + c->c;
    p->y = c->d * old.x + c->e * old.y + c->f;
    old = *p;
    fn[fn_idx](p, &old);

    if (bitmap) {
        int x,y;
        fromSpace(&x, &y, p, WIDTH, HEIGHT);
        if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
            nextColor(&bitmap[x  + y * WIDTH], &cuda_fractal.coef[i]);
        }
    }
}



__global__ void clearScreen(uchar4* bitmap, int WIDTH, int HEIGHT)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + y * blockDim.x * gridDim.x;

    if (x < WIDTH && y < HEIGHT) {
        int grey = 0;
        bitmap[offset].x = grey;
        bitmap[offset].y = grey;
        bitmap[offset].z = grey;
        bitmap[offset].w = 255;
    }
}
    
void generate_frame(uchar4 * bitmap, GPUDataBlock * d, int ticks) {
    dim3 grids(ceil(SAMPLES/d->block_width), 1);
    dim3 threads(1024, 1);
    int i = randf(0, FRACTAL_SIZE);
    int rand_offset = rand(), pt_offset = rand();
    static int iterations = 0;
    if (iterations == 0){
        dim3 grids(ceil((float)d->WIDTH/d->block_width), ceil((float)d->HEIGHT/d->block_width));
        dim3 threads(d->block_width, d->block_width);
        clearScreen<<<grids, threads >>>(bitmap, d->WIDTH, d->HEIGHT);
        gpuErrchk(cudaDeviceSynchronize());
    }
    if ( iterations < 17) {
        compute_flames<<<grids, threads>>>(NULL, d->dev_points, pt_offset, rand_offset, i, d->WIDTH, d->HEIGHT);
    } else {
        compute_flames<<<grids, threads>>>(bitmap, d->dev_points, pt_offset, rand_offset, i, d->WIDTH, d->HEIGHT);
    }
    iterations++;

}

struct CPUDataBlock {
    uchar4 *dev_bitmap;
    CPUAnimBitmap *bitmap;
    int HEIGHT;
    int WIDTH;
    int block_width;
};

void generate_frame_cpu(CPUDataBlock * d, int ticks) {
#if 0
    dim3 grids(ceil((float)d->WIDTH/d->block_width), ceil((float)d->HEIGHT/d->block_width));
    dim3 threads(d->block_width, d->block_width);
    compute_flames<<<grids, threads>>>(d->dev_bitmap, ticks, d->WIDTH, d->HEIGHT);

    gpuErrchk(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost));
#endif
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
    srand(time(NULL));
    srand48 (random ());
 
    struct Fractal_t f;
    init_fractal(&f, 5);
    
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
            data.points = (Point)malloc(sizeof(struct Point_t) * SAMPLES);
            gpuErrchk(cudaMalloc((void **) &data.dev_points, sizeof(struct Point_t) * SAMPLES));
            PlaneHammersley(data.points, SAMPLES);
            gpuErrchk(cudaMemcpy(data.dev_points, data.points, sizeof(struct Point_t) * SAMPLES, cudaMemcpyHostToDevice));
        	GPUAnimBitmap bitmap(data.WIDTH, data.HEIGHT, &data);
            cudaMemcpyToSymbol(cuda_fractal, &f, sizeof(struct Fractal_t) );
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
