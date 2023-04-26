#include <stdio.h>
//#include "../kernels.cuh"
#define m 8
#define kk_max 1024
#define tab(t, a, b)t.x += a.x * b;t.y += a.y * b;  t.z += a.z * b;t.w += a.w * b;  
    
// #define tcab(t, c, alpha, beta) c = alpha * t + beta * c;
#define tcab(t, c, alpha, beta) \
    c.x = alpha * t.x + beta * c.x;\
    c.y = alpha * t.y + beta * c.y;\
    c.z = alpha * t.z + beta * c.z;\
    c.w = alpha * t.w + beta * c.w;
    
__global__  __launch_bounds__(256) void sgemm_huge(int M, int N, int K, float *B, float *A, float *C, float alpha, float beta){
    __shared__ float shared_A[2][1024]; // blockDim * 2 for sublocks of A and B
    __shared__ float shared_B[2][1024];
    float* sa, *sb;
    sa = (float*)shared_A;
    sb = (float*)shared_B;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int wid = (tx >> 5);
    int wid_b = (wid >> 2), wid_a = (wid & 3);
    int inter_warp_id_b = ((tx&31) >> 2);
    int inter_warp_id_a = ((tx&31) & 3);
    int i1 = (wid_b << 6) + (inter_warp_id_b << 3) + (bx<<7);
    int j1 = (wid_a << 5) + (inter_warp_id_a << 3) + (by<<7);
    float4 t[16], bb0[2],aa0[2],bb1[2], aa1[2], C1[16], pre_A, pre_B;
    int idx = tx & 31, idy = tx >> 5;
    memset(t, 0, sizeof(t));
    int idx_4 = (idx<<2), idy_128 = (idy << 7), by_128 = (by << 7);
    A = A + idx_4 + (by << 7) + idy * N;
    B = B + idx_4 + (bx << 7) + idy * M;
    pre_B = *(float4*)B;
    pre_A = *(float4*)A;
    //int shared_offset = 0;
    ((float4*)sb)[tx] = pre_B;
    ((float4*)sa)[tx] = pre_A;
    // sa[(tx>>1) + ((((tx&1)<<2) + 0)<<7)]= pre_A.x;
    // sa[(tx>>1) + ((((tx&1)<<2)+1)<<7) ]= pre_A.y;
    // sa[(tx>>1) + ((((tx&1)<<2) + 2)<<7)]= pre_A.z; 
    // sa[(tx>>1) + ((((tx&1)<<2) + 3)<<7)]= pre_A.w;
    __syncthreads();
    bb0[0] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3));
    bb1[0] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3) + 4);
    aa0[0] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3));
    aa1[0] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3) + 4);

    for(int k = 0; k < K; k += 8){
        B += (M << 3);
        A += (N << 3); 
        int shared_offset = ((((k>>3) + 1)&1)<<10);
        pre_B = *(float4*)B;
        pre_A = *(float4*)A;
        #pragma unroll
        for(int kk = 0; kk < 8; kk+=1){
            int prefetch_next = ((kk + 1)&1);
            int prefetch = ((kk)&1);
            int kk_ = ((kk + 1)&7);
            bb0[prefetch_next] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3) + (kk_<<7));
            bb1[prefetch_next] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3) + 4 + (kk_<<7));
            aa0[prefetch_next] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3) + (kk_<<7));
            aa1[prefetch_next] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3) + 4 + (kk_<<7));
            tab(t[0], bb0[prefetch], aa0[prefetch].x);
            tab(t[1], bb1[prefetch], aa0[prefetch].x);
            tab(t[2], bb0[prefetch], aa0[prefetch].y);
            tab(t[3], bb1[prefetch], aa0[prefetch].y);
            tab(t[4], bb0[prefetch], aa0[prefetch].z);
            tab(t[5], bb1[prefetch], aa0[prefetch].z);
            tab(t[6], bb0[prefetch], aa0[prefetch].w);
            tab(t[7], bb1[prefetch], aa0[prefetch].w);

            tab(t[8], bb0[prefetch], aa1[prefetch].x);
            tab(t[9], bb1[prefetch], aa1[prefetch].x);
            tab(t[10], bb0[prefetch], aa1[prefetch].y);
            tab(t[11], bb1[prefetch], aa1[prefetch].y);
            tab(t[12], bb0[prefetch], aa1[prefetch].z);
            tab(t[13], bb1[prefetch], aa1[prefetch].z);
            tab(t[14], bb0[prefetch], aa1[prefetch].w);
            tab(t[15], bb1[prefetch], aa1[prefetch].w);

        }
    sb = (float*)shared_B + shared_offset;
    sa = (float*)shared_A + shared_offset;
    ((float4*)sb)[tx] = pre_B;
    ((float4*)sa)[tx] = pre_A;
    __syncthreads();
    bb0[0] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3));
    bb1[0] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3) + 4);
    aa0[0] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3));
    aa1[0] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3) + 4);
    }
    C1[0] = *(float4*)(C + i1 + j1 * M);
    C1[1] = *(float4*)(C + i1 + 4 + j1 * M);
    
    C1[2] = *(float4*)(C + i1 + (j1 + 1) * M);
    C1[3] = *(float4*)(C + i1 + 4 + (j1 + 1) * M);

    C1[4] = *(float4*)(C + i1 + (j1 + 2) * M);
    C1[5] = *(float4*)(C + i1 + 4 + (j1 + 2) * M);

    C1[6] = *(float4*)(C + i1 + (j1 + 3) * M);
    C1[7] = *(float4*)(C + i1 + 4 + (j1 + 3) * M);

    C1[8] = *(float4*)(C + i1 + (j1 + 4) * M);
    C1[9] = *(float4*)(C + i1 + 4 + (j1 + 4) * M);
    
    C1[10] = *(float4*)(C + i1 + (j1 + 5) * M);
    C1[11] = *(float4*)(C + i1 + 4 + (j1 + 5) * M);

    C1[12] = *(float4*)(C + i1 + (j1 + 6) * M);
    C1[13] = *(float4*)(C + i1 + 4 + (j1 + 6) * M);

    C1[14] = *(float4*)(C + i1 + (j1 + 7) * M);
    C1[15] = *(float4*)(C + i1 + 4 + (j1 + 7) * M);
    
    tcab(t[0], C1[0], alpha, beta);
    tcab(t[1], C1[1], alpha, beta);
    tcab(t[2], C1[2], alpha, beta);
    tcab(t[3], C1[3], alpha, beta);
    tcab(t[4], C1[4], alpha, beta);
    tcab(t[5], C1[5], alpha, beta);
    tcab(t[6], C1[6], alpha, beta);
    tcab(t[7], C1[7], alpha, beta);
    tcab(t[8], C1[8], alpha, beta);
    tcab(t[9], C1[9], alpha, beta);
    tcab(t[10], C1[10], alpha, beta);
    tcab(t[11], C1[11], alpha, beta);
    tcab(t[12], C1[12], alpha, beta);
    tcab(t[13], C1[13], alpha, beta);
    tcab(t[14], C1[14], alpha, beta);
    tcab(t[15], C1[15], alpha, beta);
    *(float4*)(C + i1 + j1 * M) = C1[0]; 
    *(float4*)(C + i1 + 4 + j1 * M) = C1[1];


    *(float4*)(C + i1 + (j1 + 1) * M) = C1[2];
    *(float4*)(C + i1 + 4 + (j1 + 1) * M) = C1[3];
    
    *(float4*)(C + i1 + (j1 + 2) * M) = C1[4];
    *(float4*)(C + i1 + 4 + (j1 + 2) * M) = C1[5];
    
    *(float4*)(C + i1 + (j1 + 3) * M) = C1[6];
    *(float4*)(C + i1 + 4 + (j1 + 3) * M) = C1[7];
    
    *(float4*)(C + i1 + (j1 + 4) * M) = C1[8];
    *(float4*)(C + i1 + 4 + (j1 + 4) * M) = C1[9];

    *(float4*)(C + i1 + (j1 + 5) * M) = C1[10];
    *(float4*)(C + i1 + 4 + (j1 + 5) * M) = C1[11];

    *(float4*)(C + i1 + (j1 + 6) * M) = C1[12];
    *(float4*)(C + i1 + 4 + (j1 + 6) * M) = C1[13];

    *(float4*)(C + i1 + (j1 + 7) * M) = C1[14];
    *(float4*)(C + i1 + 4 + (j1 + 7) * M) = C1[15];
}