#include <stdio.h>
//#include "../kernels.cuh"
#define m 8
#define err_bound 3e-1
#define kk_max 1024
#define tab(t, a, b)t.x += a.x * b;t.y += a.y * b;  t.z += a.z * b;t.w += a.w * b;  

#define checksum_8(t, a, b) t += a.x; t += a.y; t += a.z; t += a.w; t += b.x; t += b.y; t += b.z; t += b.w;
#define negative_checksum_8(t, a, b) t -= a.x; t -= a.y; t -= a.z; t -= a.w; t -= b.x; t -= b.y; t -= b.z; t -= b.w;
#define saxpy(t1,t2, a, b1,b2) t1.x += a * b1.x; t1.y += a * b1.y; t1.z += a * b1.z; t1.w += a * b1.w; \
                               t2.x += a * b2.x; t2.y += a * b2.y; t2.z += a * b2.z; t2.w += a * b2.w;

// #define tcab(t, c, alpha, beta) c = alpha * t + beta * c;
#define tcab(t, c, alpha, beta) \
    c.x = alpha * t.x + beta * c.x;\
    c.y = alpha * t.y + beta * c.y;\
    c.z = alpha * t.z + beta * c.z;\
    c.w = alpha * t.w + beta * c.w;

#define copy_float4(a, b) \
    a.x = b.x;\
    a.y = b.y;\
    a.z = b.z;\
    a.w = b.w;

#define float4_set_zero(t) t.x = 0.; t.y = 0.; t.z = 0.; t.w = 0.; 
#define comp_and_record(checksum, r, offset) \
    if(checksum.x > err_bound) r = 0 + offset; \ 
    if(checksum.y > err_bound) r = 1 + offset; \ 
    if(checksum.z > err_bound) r = 2 + offset; \ 
    if(checksum.w > err_bound) r = 3 + offset; 

// #define print_float4(a, b, id) if(a.x != a.x || a.y != a.y || a.z != a.z || a.w != a.w || b.x != b.x || b.y != b.y || b.z != b.z || b.w != b.w)printf("%d, %f, %f, %f, %f, %f, %f, %f, %f\n", id,  a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
#define print_float4(a, b, id) printf("%d, %f, %f, %f, %f, %f, %f, %f, %f\n", id,  a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
#define warp_shfl_down(a, i) \
    a.x += __shfl_down_sync(0xffffffff, a.x, i, 32); \
    a.y += __shfl_down_sync(0xffffffff, a.y, i, 32); \
    a.z += __shfl_down_sync(0xffffffff, a.z, i, 32); \
    a.w += __shfl_down_sync(0xffffffff, a.w, i, 32);

__global__  __launch_bounds__(256) void ft_sgemm_huge_warp(int M, int N, int K, float *A, float *B, float *C, float alpha, float beta){
    __shared__ float shared_A[2][1024]; // blockDim * 2 for sublocks of A and B
    __shared__ float shared_B[2][1024];
    // double shared buffer to store checksum
    __shared__ float shared_A_r[2][32];
    __shared__ float shared_B_c[2][16];
    float* sa, *sb;
    float* sAr, *sBc;
    sAr = (float*)shared_A_r;
    sBc = (float*)shared_B_c;
    sa = (float*)shared_A;
    sb = (float*)shared_B;
    int tx = threadIdx.x;
    float aa0, bb0, bb1;
    float tmp;
    int bx = blockIdx.x, by = blockIdx.y;
    int wid = (tx >> 5);
    int wid_b = (wid >> 2), wid_a = (wid & 3);
    int inter_warp_id_b = ((tx&31) >> 2);
    int inter_warp_id_a = ((tx&31) & 3);
    int i1 = (wid_b << 6) + (inter_warp_id_b << 3) + (bx<<7);
    int j1 = (wid_a << 5) + (inter_warp_id_a << 3) + (by<<7);
    int A_r_checksum_id = ((tx&31) >> 2);
    float4 t[16], bb[4],aa[4], C1[16], pre_A, pre_B, pre_A_sum; //C_c[2], C_r[2], C_c1[2], C_r1[2];
    float C_r[2],C_c; 
    float checksum_B_c[2], checksum_A_r[2];
    float A_r[2] = {0,0}, B_c = 0, checksum_sum = 0; 
    int idx = tx & 31, idy = tx >> 5;
    int r = -1, c = -1;
    int tmp_1 =  (((tx & 31) >> 2) & 1 );
    memset(t, 0, sizeof(t));
    C_c = 0;
    memset(C_r, 0, sizeof(C_r));
    memset(checksum_B_c, 0, sizeof(checksum_B_c));
    memset(checksum_A_r, 0, sizeof(checksum_A_r));
    int idx_4 = (idx<<2), idy_128 = (idy << 7), by_128 = (by << 7);
    A = A + ((tx>>1) + by_128) * N + ((tx & 1) << 2);
    B = B + idx_4 + (bx << 7) + idy * N;
    pre_B = *(float4*)B;
    pre_A = *(float4*)A;
    //int shared_offset = 0;
    ((float4*)sb)[tx] = pre_B;
    sa[(tx>>1) + ((((tx&1)<<2) + 0)<<7)]= pre_A.x;
    sa[(tx>>1) + ((((tx&1)<<2)+1)<<7) ]= pre_A.y;
    sa[(tx>>1) + ((((tx&1)<<2) + 2)<<7)]= pre_A.z; 
    sa[(tx>>1) + ((((tx&1)<<2) + 3)<<7)]= pre_A.w;
    __syncthreads();
    bb[0] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3));
    bb[1] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3) + 4);
    aa[0] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3));
    aa[1] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3) + 4);

    for(int k = 0; k < K; k += 8){
        B += (N<<3);
        A += 8; 
        int shared_offset = ((((k>>3) + 1)&1)<<10);
        int shared_checksum_offset = ((((k>>3) + 1)&1)<<4);
        pre_B = *(float4*)B;
        pre_A = *(float4*)A;

        #pragma unroll
        for(int kk = 0; kk < 8; kk+=1){
            int prefetch_next = ((kk + 1)&1);
            int prefetch = ((kk)&1);
            int kk_ = ((kk + 1)&7);

            bb[prefetch_next * 2 + 0] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3) + (kk_<<7));
            bb[prefetch_next * 2 + 1] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3) + 4 + (kk_<<7));
            aa[prefetch_next * 2 + 0] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3) + (kk_<<7));
            aa[prefetch_next * 2 + 1] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3) + 4 + (kk_<<7));
            bb0 = *(sb + (wid_b << 6) + (tx&31) * 2 + (kk_<<7));
            bb1 = *(sb + (wid_b << 6) + (tx&31) * 2 + 1 + (kk_<<7));
            aa0 = *(sa + (wid_a << 5) + (tx&31) + (kk_<<7));
            checksum_B_c[prefetch_next] = *(sBc + wid_b + (kk_<<1));
            checksum_A_r[prefetch_next] = *(sAr + wid_a + (kk_<<2));

            tab(t[0], bb[prefetch * 2], aa[prefetch * 2 + 0].x);
            tab(t[1], bb[prefetch * 2 + 1], aa[prefetch * 2 + 0].x);
            tab(t[2], bb[prefetch * 2], aa[prefetch * 2 + 0].y);  
            tab(t[3], bb[prefetch * 2 + 1], aa[prefetch * 2 + 0].y);
            tab(t[4], bb[prefetch * 2], aa[prefetch * 2 + 0].z);
            tab(t[5], bb[prefetch * 2 + 1], aa[prefetch * 2 + 0].z);
            tab(t[6], bb[prefetch * 2], aa[prefetch * 2 + 0].w);
            tab(t[7], bb[prefetch * 2 + 1], aa[prefetch * 2 + 0].w);

            tab(t[8], bb[prefetch * 2], aa[prefetch * 2 + 1].x);
            tab(t[9], bb[prefetch * 2 + 1], aa[prefetch * 2 + 1].x);
            tab(t[10], bb[prefetch * 2], aa[prefetch * 2 + 1].y);   
            tab(t[11], bb[prefetch * 2 + 1], aa[prefetch * 2 + 1].y);
            tab(t[12], bb[prefetch * 2], aa[prefetch * 2 + 1].z);
            tab(t[13], bb[prefetch * 2 + 1], aa[prefetch * 2 + 1].z);
            tab(t[14], bb[prefetch * 2], aa[prefetch * 2 + 1].w);
            tab(t[15], bb[prefetch * 2 + 1], aa[prefetch * 2 + 1].w);

            
            // C_r[0] += checksum_A_r[prefetch] * *((float*)(bb + prefetch * 2) + inter_warp_id_a * 2);
            // C_r[1] += checksum_A_r[prefetch] * *((float*)(bb + prefetch * 2) + inter_warp_id_a * 2 + 1);
            // C_c += checksum_B_c[prefetch] * *((float*)(aa + prefetch * 2) + inter_warp_id_b);
            C_r[0] += checksum_A_r[prefetch] * bb0;
            C_r[1] += checksum_A_r[prefetch] * bb1;
            C_c += checksum_B_c[prefetch] * aa0;
            
            // C_r[0] += checksum_A_r[prefetch] * *((float*)(bb + prefetch * 2) + 0 * 2);
            // C_r[1] += checksum_A_r[prefetch] * *((float*)(bb + prefetch * 2) + 0 * 2 + 1);
            // C_r[0] += checksum_A_r[prefetch] * *((float*)(bb + prefetch * 2) + 0 * 2 + 2);
            // C_r[1] += checksum_A_r[prefetch] * *((float*)(bb + prefetch * 2) + 0 * 2 + 3);
            // C_r[0] += checksum_A_r[prefetch] * *((float*)(bb + prefetch * 2) + 0 * 2 + 4);
            // C_r[1] += checksum_A_r[prefetch] * *((float*)(bb + prefetch * 2) + 0 * 2 + 5);
            // C_r[0] += checksum_A_r[prefetch] * *((float*)(bb + prefetch * 2) + 0 * 2 + 6);
            // C_r[1] += checksum_A_r[prefetch] * *((float*)(bb + prefetch * 2) + 0 * 2 + 7);
            // C_c += checksum_B_c[prefetch] * *((float*)(aa + prefetch * 2) + 0);
            // C_c += checksum_B_c[prefetch] * *((float*)(aa + prefetch * 2) + 1);
            // C_c += checksum_B_c[prefetch] * *((float*)(aa + prefetch * 2) + 2);
            // C_c += checksum_B_c[prefetch] * *((float*)(aa + prefetch * 2) + 3);
            // C_c += checksum_B_c[prefetch] * *((float*)(aa + prefetch * 2) + 4);
            // C_c += checksum_B_c[prefetch] * *((float*)(aa + prefetch * 2) + 5);
            // C_c += checksum_B_c[prefetch] * *((float*)(aa + prefetch * 2) + 6);
            // C_c += checksum_B_c[prefetch] * *((float*)(aa + prefetch * 2) + 7);
            
        }
        B_c = pre_B.x + pre_B.y + pre_B.z + pre_B.w;
        B_c += __shfl_down_sync(0xffffffff, B_c, 1, 32);
        B_c += __shfl_down_sync(0xffffffff, B_c, 2, 32);
        B_c += __shfl_down_sync(0xffffffff, B_c, 4, 32);
        B_c += __shfl_down_sync(0xffffffff, B_c, 8, 32);
        
        sBc = (float*)shared_B_c + shared_checksum_offset;
        // Attention! This branch cause warp divergence.
        if(((tx&31) & 15) == 0){
            //shared_checksum_offset
            // sAr[int(tx / 2)] = A_r;
            sBc[int(tx / 16)] = B_c;
        }
        sb = (float*)shared_B + shared_offset;
        sa = (float*)shared_A + shared_offset;
        ((float4*)sb)[tx] = pre_B;
        sa[(tx>>1) + ((((tx&1)<<2) + 0)<<7)]= pre_A.x;
        sa[(tx>>1) + ((((tx&1)<<2)+1)<<7) ]= pre_A.y;
        sa[(tx>>1) + ((((tx&1)<<2) + 2)<<7)]= pre_A.z; 
        sa[(tx>>1) + ((((tx&1)<<2) + 3)<<7)]= pre_A.w;
        __syncthreads();        
        
        A_r[0] = pre_A.x + pre_A.y + pre_A.z + pre_A.w;
        A_r[0] += __shfl_down_sync(0xffffffff, A_r[0], 1, 32);
        A_r[0] += __shfl_down_sync(0xffffffff, A_r[0], 2, 32);
        A_r[0] += __shfl_down_sync(0xffffffff, A_r[0], 4, 32);

        sAr = (float*)shared_A_r + shared_checksum_offset * 2;
        // Attention! This branch cause warp divergence.
        if(((tx&31) & 7) == 0){
            //shared_checksum_offset
            sAr[int(tx / 8)] = A_r[0];
        }
        if((k % 256) == 0){
            C_c += 1.0;
            C_r[0] += 1.0;
            checksum_sum = checksum_A_r[0] + checksum_A_r[1] + checksum_B_c[0] + checksum_B_c[1] + A_r[0] + A_r[1];
            checksum_sum -= C_c;
            checksum_sum -= C_r[0];
            checksum_sum -= C_r[1];
            if((checksum_sum - err_bound > 0) || (checksum_sum + err_bound < 0)){
            r = -1, c = -1;
            r = 0, c = 0;
            *(((float*)(t + c * 2 + r / 4)) + r % 4) += checksum_sum;
            }
            
        }
        
        
        bb[0] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3));
        bb[1] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3) + 4);
        aa[0] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3));
        aa[1] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3) + 4);
    }
    C1[0] = *(float4*)(C + i1 + j1 * N);
    C1[1] = *(float4*)(C + i1 + 4 + j1 * N);
    
    C1[2] = *(float4*)(C + i1 + (j1 + 1) * N);
    C1[3] = *(float4*)(C + i1 + 4 + (j1 + 1) * N);

    C1[4] = *(float4*)(C + i1 + (j1 + 2) * N);
    C1[5] = *(float4*)(C + i1 + 4 + (j1 + 2) * N);

    C1[6] = *(float4*)(C + i1 + (j1 + 3) * N);
    C1[7] = *(float4*)(C + i1 + 4 + (j1 + 3) * N);

    C1[8] = *(float4*)(C + i1 + (j1 + 4) * N);
    C1[9] = *(float4*)(C + i1 + 4 + (j1 + 4) * N);
    
    C1[10] = *(float4*)(C + i1 + (j1 + 5) * N);
    C1[11] = *(float4*)(C + i1 + 4 + (j1 + 5) * N);

    C1[12] = *(float4*)(C + i1 + (j1 + 6) * N);
    C1[13] = *(float4*)(C + i1 + 4 + (j1 + 6) * N);

    C1[14] = *(float4*)(C + i1 + (j1 + 7) * N);
    C1[15] = *(float4*)(C + i1 + 4 + (j1 + 7) * N);
    
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
    *(float4*)(C + i1 + j1 * N) = C1[0]; 
    *(float4*)(C + i1 + 4 + j1 * N) = C1[1];


    *(float4*)(C + i1 + (j1 + 1) * N) = C1[2];
    *(float4*)(C + i1 + 4 + (j1 + 1) * N) = C1[3];
    
    *(float4*)(C + i1 + (j1 + 2) * N) = C1[4];
    *(float4*)(C + i1 + 4 + (j1 + 2) * N) = C1[5];
    
    *(float4*)(C + i1 + (j1 + 3) * N) = C1[6];
    *(float4*)(C + i1 + 4 + (j1 + 3) * N) = C1[7];
    
    *(float4*)(C + i1 + (j1 + 4) * N) = C1[8];
    *(float4*)(C + i1 + 4 + (j1 + 4) * N) = C1[9];

    *(float4*)(C + i1 + (j1 + 5) * N) = C1[10];
    *(float4*)(C + i1 + 4 + (j1 + 5) * N) = C1[11];

    *(float4*)(C + i1 + (j1 + 6) * N) = C1[12];
    *(float4*)(C + i1 + 4 + (j1 + 6) * N) = C1[13];

    *(float4*)(C + i1 + (j1 + 7) * N) = C1[14];
    *(float4*)(C + i1 + 4 + (j1 + 7) * N) = C1[15];
}