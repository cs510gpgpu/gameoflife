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

#define FRACTAL_SIZE 10
#define WARP_SIZE 32

#define SQUARE(_n_) ((_n_) * (_n_))
#define SAMPLES(w, h) ((w) * (h))

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

void cleanup_gpu(struct GPUDataBlock * d )
{
    free(d->points);
    gpuErrchk(cudaFree(d->dev_points));
}

__device__ void v_0(Point out, Point in);
__device__ void v_1(Point out, Point in);
__device__ void v_2(Point out, Point in);
__device__ void v_3(Point out, Point in);
__device__ void v_18(Point out, Point in);
__device__ void v_19(Point out, Point in);

#define DEVICE_FN {v_0 , v_1, v_2, v_3, v_18, v_19};
typedef void (*V_func)(Point out, Point in);
const V_func all_fn[] = DEVICE_FN;
const int all_fn_size = sizeof(all_fn) / sizeof(V_func);
struct Args_t globalArgs;

typedef struct Fractal_t {
    struct Coef_t coef[all_fn_size];
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
        result[k].x = 2*u - 1;
        result[k].y = 2*v - 1;
        assert(result[k].x <= 1 && result[k].x >= -1);
        assert(result[k].y <= 1 && result[k].y >= -1);
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

void contractive_map(Coef cf)
{
    double a, b, d, e;
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
    cf->a = a;
    cf->b = b;
    cf->c = randf (-1, 1);
    cf->d = d;
    cf->e = e;
    cf->f = randf (-1, 1);
}

void init_fractal(Fractal f, int n)
{
    f->n = FRACTAL_SIZE;
    for (int n = 0; n < all_fn_size; n++) {
        Coef c = &f->coef[n];
        contractive_map(c);
        printf("a %f b %f c %f d %f e %f f %f\n", c->a, c->b, c->c, c->d, c->e, c->f);
        
        f->coef[n].color.x = randf(64,255);
        f->coef[n].color.y = randf(64,255);
        f->coef[n].color.z = randf(64,255);
    }
}

void clean_fractal(Fractal f)
{
    f->n = 0;
}
 
__device__ float r(Point in)
{
    return sqrtf((SQUARE(in->x) + SQUARE(in->y)));
}

__device__ float theta(Point in)
{
    return atanf(in->x / in->y);
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
    float recip = 1.0 / SQUARE(r(in));
    out->x = recip * in->x;
    out->y = recip * in->y;
}

__device__ void v_3(Point out, Point in)
{
    float sqr_r_in = SQUARE(r(in));
    float sin_sq_r_in = sinf(sqr_r_in);
    float cos_sq_r_in = cosf(sqr_r_in);
    out->x = in->x * sin_sq_r_in - in->y * cos_sq_r_in;
    out->y = in->x * cos_sq_r_in + in->y * sin_sq_r_in;    
}

#if OMEGA_IMPLEMENTED
__device__ void v_13(Point out, Point in)
{
    out->x = sqrtf(r(in)) * cosf(theta()/2 + omega());
    out->y = sqrtf(r(in)) * sinf(theta()/2 + omega());
}
#endif

__device__ void v_18(Point out, Point in)
{
    float expf_0 = expf(in->x - 1);
    out->x = expf_0 * cospif(in->y);
    out->y = expf_0 * sinpif(in->y);
}

__device__ void v_19(Point out, Point in)
{
    float theta_0 = theta(in);
    float cos_theta = cosf(theta_0);
    float sin_theta = sinf(theta_0);
    float p_sinf = powf(r(in), sin_theta);
    out->x = p_sinf * cos_theta;
    out->y = p_sinf * sin_theta;
}

__device__ void nextColor(uchar4 * out, uchar4 * color)
{
#define C(_color_) out->_color_ = (((unsigned int)out-> _color_ + color->_color_) / 2)
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

__device__ void fromSpace(int * x, int * y, Point in, int width, int height)
{
    *x = in->x * (width / 2.0) + width / 2.0;
    *y = in->y * (height / 2.0) + height / 2.0;
}

__constant__ struct Fractal_t cuda_fractal;



__global__ void compute_flames(uchar4* bitmap, struct Point_t *points, int pt_offset, int fn_offset, int i, int WIDTH, int HEIGHT)
{
    if (threadIdx.x + blockIdx.x * blockDim.x >= SAMPLES(WIDTH, HEIGHT)) {
        return;
    }
    int pt_idx = (threadIdx.x + blockIdx.x * blockDim.x + pt_offset) % SAMPLES(WIDTH, HEIGHT);
    Point p = &points[pt_idx];
    struct Point_t old = *p;
    struct Point_t tmp = old;
    
    int fn_idx = ((threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE + fn_offset) % all_fn_size;
    Coef c = &cuda_fractal.coef[fn_idx];
    tmp.x = c->a * old.x + c->b * old.y + c->c;
    tmp.y = c->d * old.x + c->e * old.y + c->f;
    old = tmp;
    V_func fn[] = DEVICE_FN;
    fn[fn_idx](p, &old);
    
    if (bitmap) {
        int x,y;
        fromSpace(&x, &y, p, WIDTH, HEIGHT);
        if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
            nextColor(&bitmap[x  + y * WIDTH], &cuda_fractal.coef[fn_idx].color);
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
    static int iterations = 0;
    if (iterations == 0) {
        dim3 grids(ceil((float)d->WIDTH/32), ceil((float)d->HEIGHT/32));
        dim3 threads(32, 32);
        clearScreen<<<grids, threads >>>(bitmap, d->WIDTH, d->HEIGHT);
        gpuErrchk(cudaDeviceSynchronize());
    }

    dim3 grids(ceil((float)SAMPLES(d->WIDTH, d->HEIGHT)/d->block_width), 1);
    dim3 threads(d->block_width, 1);
    int i = randf(0, all_fn_size);
    int rand_offset = rand(), pt_offset = rand();
    compute_flames<<<grids, threads>>>(iterations < 17 ? NULL : bitmap,
        d->dev_points, pt_offset, rand_offset, i, d->WIDTH, d->HEIGHT);
    gpuErrchk(cudaDeviceSynchronize());
    iterations++;
    timeout(&globalArgs, iterations);
}

struct CPUDataBlock {
    uchar4 *dev_bitmap;
    CPUAnimBitmap *bitmap;
    struct GPUDataBlock g;
};

void generate_frame_cpu(CPUDataBlock * d, int ticks)
{
    generate_frame(d->dev_bitmap, &d->g, ticks);
    gpuErrchk(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost));
}

void cleanup_cpu(struct CPUDataBlock *d)
{
    cleanup_gpu(&d->g);
    gpuErrchk(cudaFree(d->dev_bitmap));
}

void init_gpu_datablock(GPUDataBlock *d)
{
    struct Fractal_t f;
    init_fractal(&f, 5);
    d->HEIGHT = globalArgs.height;
    d->WIDTH = globalArgs.width;
    d->block_width = globalArgs.blockwidth;
    int samples = SAMPLES(d->WIDTH, d->HEIGHT);
    d->points = (Point)malloc(sizeof(struct Point_t) * samples);
    gpuErrchk(cudaMalloc((void **) &d->dev_points, sizeof(struct Point_t) * samples));
    PlaneHammersley(d->points, samples);
    gpuErrchk(cudaMemcpy(d->dev_points, d->points, sizeof(struct Point_t) * samples, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(cuda_fractal, &f, sizeof(struct Fractal_t) ));
}

int main(int argc, char **argv) {    
    processArgs("flames", argv, argc, &globalArgs);
    
    srand(time(NULL));
    srand48(time(NULL));
 
    switch(globalArgs.mode) {
    case PROFILE_NONE:
        printf("Set a profile mode. \"None\" is unimplemented.\n");
        break;
    case PROFILE_GPU:
        {
            GPUDataBlock data;
            init_gpu_datablock(&data);
        	GPUAnimBitmap bitmap(data.WIDTH, data.HEIGHT, &data);        
            bitmap.anim_and_exit((void (*)(uchar4*,void*,int))generate_frame, (void(*)(void*))cleanup_gpu);
        }
        break;
    case PROFILE_CPU:
        {
            CPUDataBlock data;
            init_gpu_datablock(&data.g);
            CPUAnimBitmap bitmap(data.g.WIDTH, data.g.HEIGHT, &data);
            data.bitmap = &bitmap;
            gpuErrchk(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));
            bitmap.anim_and_exit((void (*)(void*,int))generate_frame_cpu, (void(*)(void*))cleanup_cpu);
        }
        break;
    default:
        printf("Unhandled mode by flames.\n");
        exit(1);
    }
}
