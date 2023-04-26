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


__global__  __launch_bounds__(256) void ft_sgemm_huge_thread(int M, int N, int K, float *A, float *B, float *C, float alpha, float beta){
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
    float4 t[16], bb0[2],aa0[2],bb1[2], aa1[2], C1[16], pre_A, pre_B, C_c[2], C_r[2], C_c1[2], C_r1[2];
    float A_r = 0, B_c = 0., checksum_sum = 0; 
    int idx = tx & 31, idy = tx >> 5;
    memset(t, 0, sizeof(t));
    memset(C_c, 0, sizeof(C_c));
    memset(C_r, 0, sizeof(C_r));
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
    bb0[0] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3));
    bb1[0] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3) + 4);
    aa0[0] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3));
    aa1[0] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3) + 4);

    for(int k = 0; k < K; k += 8){
        B += (N<<3);
        A += 8; 
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
            A_r = 0, B_c = 0;
            checksum_8(A_r, aa0[prefetch], aa1[prefetch]);
            checksum_8(B_c, bb0[prefetch], bb1[prefetch]);
            saxpy(C_r[0], C_r[1], A_r, bb0[prefetch], bb1[prefetch]);
            saxpy(C_c[0], C_c[1], B_c, aa0[prefetch], aa1[prefetch]);

        }
        sb = (float*)shared_B + shared_offset;
        sa = (float*)shared_A + shared_offset;
        ((float4*)sb)[tx] = pre_B;
        sa[(tx>>1) + ((((tx&1)<<2) + 0)<<7)]= pre_A.x;
        sa[(tx>>1) + ((((tx&1)<<2)+1)<<7) ]= pre_A.y;
        sa[(tx>>1) + ((((tx&1)<<2) + 2)<<7)]= pre_A.z; 
        sa[(tx>>1) + ((((tx&1)<<2) + 3)<<7)]= pre_A.w;
        __syncthreads();        
        if((k % 256) == 0){
            copy_float4(C_c1[0], C_c[0]);copy_float4(C_c1[1], C_c[1]);
            copy_float4(C_r1[0], C_r[0]);copy_float4(C_r1[1], C_r[1]);
            
            
            // tcab(C_c[0], C_c1[0], 1.0, 0.0); tcab(C_c[1], C_c1[1], 1.0, 0.0);
            // tcab(C_r[0], C_r1[0], 1.0, 0.0); tcab(C_r[1], C_r1[1], 1.0, 0.0);
            // print_float4(C_r[0], C_r[1], gridDim.x * 128);
            // print_float4(C_c[0], C_c[1], gridDim.x * 128);
            negative_checksum_8(C_c1[0].x, t[0], t[1])
            negative_checksum_8(C_c1[0].y, t[2], t[3])
            negative_checksum_8(C_c1[0].z, t[4], t[5])
            negative_checksum_8(C_c1[0].w, t[6], t[7])
            negative_checksum_8(C_c1[1].x, t[8], t[9])
            negative_checksum_8(C_c1[1].y, t[10], t[11])
            negative_checksum_8(C_c1[1].z, t[12], t[13])
            negative_checksum_8(C_c1[1].w, t[14], t[15])
            
            // C_r_ref[0] = t[0] + t[2] + t[4] + t[6] + t[8] + t[10] + t[12] + t[14];
            tcab(t[0], C_r1[0], -1.0, 1.0)
            tcab(t[2], C_r1[0], -1.0, 1.0)
            tcab(t[4], C_r1[0], -1.0, 1.0)
            tcab(t[6], C_r1[0], -1.0, 1.0)
            tcab(t[8], C_r1[0], -1.0, 1.0)
            tcab(t[10], C_r1[0], -1.0, 1.0)
            tcab(t[12], C_r1[0], -1.0, 1.0)
            tcab(t[14], C_r1[0], -1.0, 1.0)
            
            // C_r_ref[1] = t[1] + t[3] + t[5] + t[7] + t[9] + t[11] + t[13] + t[15];
            tcab(t[1], C_r1[1], -1.0, 1.0)
            tcab(t[3], C_r1[1], -1.0, 1.0)
            tcab(t[5], C_r1[1], -1.0, 1.0)
            tcab(t[7], C_r1[1], -1.0, 1.0)
            tcab(t[9], C_r1[1], -1.0, 1.0)
            tcab(t[11], C_r1[1], -1.0, 1.0)
            tcab(t[13], C_r1[1], -1.0, 1.0)
            tcab(t[15], C_r1[1], -1.0, 1.0)
            
            
            C_r1[1].x = 1;
            C_c1[0].x = 1;
            checksum_sum = 0;
            checksum_8(checksum_sum, C_r1[0], C_r1[1]);
            checksum_8(checksum_sum, C_c1[0], C_c1[1]);

            //\printf("%d \n", checksum_sum);
            // 16 comparisons
            bool verified = (checksum_sum - err_bound > 0) || (checksum_sum + err_bound < 0);
            //if(verified){
                            
            // print_float4(C_r1[0], C_r1[1], gridDim.x * 128);
            // print_float4(C_c1[0], C_c1[1], gridDim.x * 128);
            int r = 0, c = 0;
            //t[c * 2 + r / 4].x += 0;
            //}
            *(((float*)(t + c * 2 + r / 4)) + r % 4) += checksum_sum;
        }
        bb0[0] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3));
        bb1[0] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3) + 4);
        aa0[0] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3));
        aa1[0] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3) + 4);
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