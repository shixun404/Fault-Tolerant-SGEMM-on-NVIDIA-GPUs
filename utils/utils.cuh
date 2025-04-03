#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#define CUDA_CALLER(call) do{\
  cudaError_t cuda_ret = (call);\
  if(cuda_ret != cudaSuccess){\
    printf("CUDA Error at line %d in file %s\n", __LINE__, __FILE__);\
    printf("  Error message: %s\n", cudaGetErrorString(cuda_ret));\
    printf("  In the function call %s\n", #call);\
    exit(1);\
  }\
}while(0)
#define CEIL_DIV(m,n) ( (m) + (n) - 1 ) / (n)



class saxpy_timer
{
public:
    saxpy_timer() { reset(); }
    void reset() {
    t0_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed(bool reset_timer=false) {
    std::chrono::high_resolution_clock::time_point t =
            std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(t - t0_);
    if (reset_timer)
        reset();
    return time_span.count();
    }
    double elapsed_msec(bool reset_timer=false) {
    return elapsed(reset_timer) * 1000;
    }
private:
    std::chrono::high_resolution_clock::time_point t0_;
};

//__global__ void fill(float *a , float x, int N);

cudaDeviceProp getDetails(int deviceId);

void generate_random_vector(float* target, int n);

void copy_vector(float *src, float *dest, int n);

bool verify_vector(float *vec1, float *vec2, int n);

void fill_vector(float*, float, int);

void copy_matrix(float *src, float *dest, int n);

void generate_random_matrix(float* target, int n);

bool verify_matrix(float*, float*, int m, int n);

void cpu_gemm(float alpha, float beta, float *mat1, float*mat2, int max_size, float* mat3);

void print_matrix(float*, int);
