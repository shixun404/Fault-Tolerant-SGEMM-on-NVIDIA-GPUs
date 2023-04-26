#include <stdio.h>
//#include "../kernels.cuh"
#define m 8


#define kk_max 1024
#define tab(t, a, b)t.x += a.x * b;t.y += a.y * b;  t.z += a.z * b;t.w += a.w * b;  

#define checksum_8(t, a, b) t = a.x; t += a.y; t += a.z; t += a.w; t += b.x; t += b.y; t += b.z; t += b.w;
#define fabs_checksum_8(t, a, b) t = fabsf(a.x); t += fabsf(a.y); t += fabsf(a.z); t += fabsf(a.w); t += fabsf(b.x); t += fabsf(b.y); t += fabsf(b.z); t += fabsf(b.w);
#define negative_checksum_8(t, a, b) t -= a.x; t -= a.y; t -= a.z; t -= a.w; t -= b.x; t -= b.y; t -= b.z; t -= b.w;
#define saxpy(alpha, a,b) b.x = alpha * a.x; b.y = alpha * a.y; b.z = alpha * a.z; b.w = alpha * a.w;
#define correct_t(t_4, col, row_4, err_correct) \
    t_4.x += int((col * row_4.x) / (err_correct * err_correct)) * col;\
    t_4.y += int((col * row_4.y) / (err_correct * err_correct)) * col;\
    t_4.z += int((col * row_4.z) / (err_correct * err_correct)) * col;\
    t_4.w += int((col * row_4.w) / (err_correct * err_correct)) * col;

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

#define print_float4(a, b, id) printf("%f, %f, %f, %f, %f, %f, %f, %f, %f\n", id,  a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
#define warp_shfl_down(a, i) \
    a.x += __shfl_down_sync(0xffffffff, a.x, i, 32); \
    a.y += __shfl_down_sync(0xffffffff, a.y, i, 32); \
    a.z += __shfl_down_sync(0xffffffff, a.z, i, 32); \
    a.w += __shfl_down_sync(0xffffffff, a.w, i, 32);

__global__  __launch_bounds__(256) void ft_sgemm_huge(int M, int N, int K, float *B, float *A, float *C, float alpha, float beta){
    __shared__ float shared[4][1024]; 
    float* sa, *sb;
    float* sAr, *sBc;
    sAr = (float*)(shared) + 1024;
    sBc = (float*)(shared) + 1024 + 2048;
    sa = (float*)shared;
    sb = (float*)shared + 2048; 
    int tx = threadIdx.x;
    
    float2 bb0[2];
    float aa0[2];
    float tmp;
    float base[8] = {1., 2., 3., 4., 5., 6., 7., 8.};
    int bx = blockIdx.x, by = blockIdx.y;
    int wid = (tx >> 5);
    int wid_b = (wid >> 2), wid_a = (wid & 3);
    int inter_warp_id_b = ((tx&31) >> 2);
    int inter_warp_id_a = ((tx&31) & 3);
    
    int tx_injec = 127;
    float err_bound1 =95;
    float error_inject = ((tx==tx_injec)?100.0:0.0);
    int i1 = (wid_b << 6) + (inter_warp_id_b << 3) + (bx<<7);
    int j1 = (wid_a << 5) + (inter_warp_id_a << 3) + (by<<7);
    int A_r_checksum_id = ((tx&31) >> 2);
    float4 t[16], bb[4],aa[4], C1[16], pre_A, pre_B, pre_A_sum, C_c1[2], C_r1[2];
    float4 block_level_B_c = {0, 0, 0, 0}, block_level_A_r = {0, 0, 0, 0};
    
    float C_c; 
    float checksum_B_c[2], checksum_A_r[2];
    float A_r[2] = {0.,0.}, B_c = 0, checksum_sum = 0; 
    int idx = tx & 31, idy = tx >> 5;
    int r = -1, c = -1;
    int tmp_1 =  (((tx & 31) >> 2) & 1 );
    memset(t, 0, sizeof(t));
    C_c = 0.0;
    memset(C_c1, 0, sizeof(C_c1));
    memset(C_r1, 0, sizeof(C_r1));
    memset(checksum_B_c, 0, sizeof(checksum_B_c));
    memset(checksum_A_r, 0, sizeof(checksum_A_r));
    int idx_4 = (idx<<2), idy_128 = (idy << 7), by_128 = (by << 7);
    A = A + idx_4 + (by << 7) + idy * N;
    B = B + idx_4 + (bx << 7) + idy * M;
    pre_B = *(float4*)B;
    pre_A = *(float4*)A;
    //int shared_offset = 0;
    ((float4*)sb)[tx] = pre_B;
    ((float4*)sa)[tx] = pre_A;
    B_c = pre_B.x + pre_B.y + pre_B.z + pre_B.w;
    B_c += __shfl_xor_sync(0xffffffff, B_c, 1, 32);
    B_c += __shfl_xor_sync(0xffffffff, B_c, 2, 32);
    B_c += __shfl_xor_sync(0xffffffff, B_c, 4, 32);
    B_c += __shfl_xor_sync(0xffffffff, B_c, 8, 32);
    B_c += __shfl_xor_sync(0xffffffff, B_c, 16, 32);

    
    __syncthreads();        
    A_r[0] = pre_A.x + pre_A.y + pre_A.z + pre_A.w;
    A_r[0] += __shfl_xor_sync(0xffffffff, A_r[0], 1, 32);
    A_r[0] += __shfl_xor_sync(0xffffffff, A_r[0], 2, 32);
    A_r[0] += __shfl_xor_sync(0xffffffff, A_r[0], 4, 32);
    A_r[0] += __shfl_xor_sync(0xffffffff, A_r[0], 8, 32);
    A_r[0] += __shfl_xor_sync(0xffffffff, A_r[0], 16, 32);    
    saxpy(B_c, pre_A, block_level_B_c);
        
    *(((float4*)sBc) + tx) = block_level_B_c;
    saxpy(A_r[0], pre_B, block_level_A_r);
    
    *(((float4*)sAr) + tx) = block_level_A_r;
    __syncthreads();
    int offset_2048 = (tx > 127)?0:2048;
    int offset_2048_ = (tx > 127)?2048:0;
    int tx_128 = (tx & 127);
    int tx_div_128_mul_4 =  4 * ((tx > 127)?0:1);
    int tx_div_128_mul_7 = 7 * ((tx > 127)?1:0);
    sAr += offset_2048 + (tx & 127);
    C_c += *(sAr + (0 << 7));
    C_c += *(sAr + (1 << 7));
    C_c += *(sAr + (2 << 7));
    C_c += *(sAr + (3 << 7));
    C_c += *(sAr + (4 << 7));
    C_c += *(sAr + (5 << 7));
    C_c += *(sAr + (6 << 7));
    C_c += *(sAr + (7 << 7));
    bb[0] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3));
    bb[1] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3) + 4);
    aa[0] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3));
    aa[1] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3) + 4);
    for(int k = 0; k < K; k += 8){
        B += (M<<3);
        A += (N<<3); 
        int shared_offset = ((((k>>3) + 1)&1)<<10);
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
        }
        if(((k+8) %128) == 0){
            t[1].y += 1.0 * error_inject;
            checksum_8(C_c1[0].x, t[0], t[1])
            checksum_8(C_c1[0].y, t[2], t[3])
            checksum_8(C_c1[0].z, t[4], t[5])
            checksum_8(C_c1[0].w, t[6], t[7])
            checksum_8(C_c1[1].x, t[8], t[9])
            checksum_8(C_c1[1].y, t[10], t[11])
            checksum_8(C_c1[1].z, t[12], t[13])
            checksum_8(C_c1[1].w, t[14], t[15])
            
            tcab(t[0], C_r1[0], 1.0, 0.0)
            tcab(t[2], C_r1[0], 1.0, 1.0)
            tcab(t[4], C_r1[0], 1.0, 1.0)
            tcab(t[6], C_r1[0], 1.0, 1.0)
            tcab(t[8], C_r1[0], 1.0, 1.0)
            tcab(t[10], C_r1[0], 1.0, 1.0)
            tcab(t[12], C_r1[0], 1.0, 1.0)
            tcab(t[14], C_r1[0], 1.0, 1.0)
            tcab(t[1], C_r1[1], 1.0, 0.0)
            tcab(t[3], C_r1[1], 1.0, 1.0)
            tcab(t[5], C_r1[1], 1.0, 1.0)
            tcab(t[7], C_r1[1], 1.0, 1.0)
            tcab(t[9], C_r1[1], 1.0, 1.0)
            tcab(t[11], C_r1[1], 1.0, 1.0)
            tcab(t[13], C_r1[1], 1.0, 1.0)
            tcab(t[15], C_r1[1], 1.0, 1.0)

            __syncthreads();
            
            float* s = ((float*)(shared) + ((wid_b) << 3) + (wid_a << 9) + inter_warp_id_b + (inter_warp_id_a << 7) + 0);
            float* s_ = ((float*)(shared) + 2048 + ((wid_a) << 2) + (wid_b << 10) + inter_warp_id_a + (inter_warp_id_b << 7) + 0);
            *s = C_c1[0].x;
            *(s + (1 << 4)) = C_c1[0].y;
            *(s + (2 << 4)) = C_c1[0].z;
            *(s + (3 << 4)) = C_c1[0].w;
            *(s + (4 << 4)) = C_c1[1].x;
            *(s + (5 << 4)) = C_c1[1].y;
            *(s + (6 << 4)) = C_c1[1].z;
            *(s + (7 << 4)) = C_c1[1].w;
            *s_ = C_r1[0].x;
            *(s_ + (1 << 4)) = C_r1[0].y;
            *(s_ + (2 << 4)) = C_r1[0].z;
            *(s_ + (3 << 4)) = C_r1[0].w;
            *(s_ + (4 << 4)) = C_r1[1].x;
            *(s_ + (5 << 4)) = C_r1[1].y;
            *(s_ + (6 << 4)) = C_r1[1].z;
            *(s_ + (7 << 4)) = C_r1[1].w;
            
            __syncthreads();
            float C_c_ = C_c;
            if (tx < 128){
                float4 r_ = *((float4*)((float*)shared + ((tx&127) << 4)));
                C_c_ -= r_.x;
                C_c_ -= r_.y;
                C_c_ -= r_.z;
                C_c_ -= r_.w;
                r_ = *((float4*)((float*)shared + ((tx&127) << 4) + 4));
                C_c_ -= r_.x;
                C_c_ -= r_.y;
                C_c_ -= r_.z;
                C_c_ -= r_.w;
                r_ = *((float4*)((float*)shared + ((tx&127) << 4) + 8));
                C_c_ -= r_.x;
                C_c_ -= r_.y;
                C_c_ -= r_.z;
                C_c_ -= r_.w;
                r_ = *((float4*)((float*)shared + ((tx&127) << 4) + 12));
                C_c_ -= r_.x;
                C_c_ -= r_.y;
                C_c_ -= r_.z;
                C_c_ -= r_.w;
            }
            else{
                float4 r_ = *((float4*)((float*)shared + 2048 + ((tx&127) << 4)));
                C_c_ -= r_.x;
                C_c_ -= r_.y;
                C_c_ -= r_.z;
                C_c_ -= r_.w;
                r_ = *((float4*)((float*)shared  + 2048 + ((tx&127) << 4) + 4));
                C_c_ -= r_.x;
                C_c_ -= r_.y;
                C_c_ -= r_.z;
                C_c_ -= r_.w;
                r_ = *((float4*)((float*)shared  + 2048 + ((tx&127) << 4) + 8));
                C_c_ -= r_.x;
                C_c_ -= r_.y;
                C_c_ -= r_.z;
                C_c_ -= r_.w;
                r_ = *((float4*)((float*)shared  + 2048 + ((tx&127) << 4) + 12));
                C_c_ -= r_.x;
                C_c_ -= r_.y;
                C_c_ -= r_.z;
                C_c_ -= r_.w;
            }
            __syncthreads();
            *((float*)shared + tx) = C_c_;
            __syncthreads();
            float error = 0.0, error1 = 0.0;
            float errsum = 0.0, errsum1 = 0.0; 
            float checksum_1 = 0.0;
            float4 tmp_col[2], tmp_row[2];
            tmp_row[0] =  (*((float4*)((float*)shared + 128 + (wid_b * 64 + inter_warp_id_b * 8))));
            tmp_row[1] =  (*((float4*)((float*)shared + 128 + (wid_b * 64 + inter_warp_id_b * 8) + 4)));
            tmp_col[0] =  (*((float4*)((float*)shared + (wid_a * 32 + inter_warp_id_a * 8))));
            tmp_col[1] =  (*((float4*)((float*)shared + (wid_a * 32 + inter_warp_id_a * 8) + 4)));
            
            if(blockIdx.x == 0 && blockIdx.y == 0 && tx == tx_injec)
            // if(tmp_row[1].y > err_bound1 && tmp_col[0].x > err_bound1)
            printf("tx: %d, k: %d, %f, %f\n", tx, k+8, tmp_row[1].y, tmp_col[0].x);
            correct_t(t[0], tmp_col[0].x, tmp_row[0], err_bound1);
            correct_t(t[1], tmp_col[0].x, tmp_row[1], err_bound1);
            correct_t(t[2], tmp_col[0].y, tmp_row[0], err_bound1);
            correct_t(t[3], tmp_col[0].y, tmp_row[1], err_bound1);
            correct_t(t[4], tmp_col[0].z, tmp_row[0], err_bound1);
            correct_t(t[5], tmp_col[0].z, tmp_row[1], err_bound1);
            correct_t(t[6], tmp_col[0].w, tmp_row[0], err_bound1);
            correct_t(t[7], tmp_col[0].w, tmp_row[1], err_bound1);

            correct_t(t[8],  tmp_col[1].x, tmp_row[0], err_bound1);
            correct_t(t[9],  tmp_col[1].x, tmp_row[1], err_bound1);
            correct_t(t[10], tmp_col[1].y, tmp_row[0], err_bound1);
            correct_t(t[11], tmp_col[1].y, tmp_row[1], err_bound1);
            correct_t(t[12], tmp_col[1].z, tmp_row[0], err_bound1);
            correct_t(t[13], tmp_col[1].z, tmp_row[1], err_bound1);
            correct_t(t[14], tmp_col[1].w, tmp_row[0], err_bound1);
            correct_t(t[15], tmp_col[1].w, tmp_row[1], err_bound1);
            
            __syncthreads();
        }


        sb = (float*)shared + 2048 + shared_offset;
        sa = (float*)shared + shared_offset;
        ((float4*)sb)[tx] = pre_B;
        ((float4*)sa)[tx] = pre_A;
        int shared_offset_ = ((((k>>3))&1)<<10);
        sAr = (float*)shared + shared_offset_;
        sBc = (float*)shared + 2048 + shared_offset_;

        B_c = pre_B.x + pre_B.y + pre_B.z + pre_B.w;
        // *(sBc + tx) = B_c;
        B_c += __shfl_xor_sync(0xffffffff, B_c, 1, 32);
        B_c += __shfl_xor_sync(0xffffffff, B_c, 2, 32);
        B_c += __shfl_xor_sync(0xffffffff, B_c, 4, 32);
        B_c += __shfl_xor_sync(0xffffffff, B_c, 8, 32);
        B_c += __shfl_xor_sync(0xffffffff, B_c, 16, 32);

        
        __syncthreads();        
        A_r[0] = pre_A.x + pre_A.y + pre_A.z + pre_A.w;
        // *(sAr + tx) = A_r[0];
        A_r[0] += __shfl_xor_sync(0xffffffff, A_r[0], 1, 32);
        A_r[0] += __shfl_xor_sync(0xffffffff, A_r[0], 2, 32);
        A_r[0] += __shfl_xor_sync(0xffffffff, A_r[0], 4, 32);
        A_r[0] += __shfl_xor_sync(0xffffffff, A_r[0], 8, 32);
        A_r[0] += __shfl_xor_sync(0xffffffff, A_r[0], 16, 32);
        saxpy(B_c, pre_A, block_level_B_c);
        
        *(((float4*)sBc) + tx) = block_level_B_c;
        saxpy(A_r[0], pre_B, block_level_A_r);
        
        *(((float4*)sAr) + tx) = block_level_A_r;
        __syncthreads();
        sAr += offset_2048 + tx_128;
        C_c += *(sAr + (0 << 7));
        C_c += *(sAr + (1 << 7));
        C_c += *(sAr + (2 << 7));
        C_c += *(sAr + (3 << 7));
        C_c += *(sAr + (4 << 7));
        C_c += *(sAr + (5 << 7));
        C_c += *(sAr + (6 << 7));
        C_c += *(sAr + (7 << 7));

        bb[0] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3));
        bb[1] = *(float4*)(sb + (wid_b << 6) + (inter_warp_id_b << 3) + 4);
        aa[0] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3));
        aa[1] = *(float4*)(sa + (wid_a << 5) + (inter_warp_id_a << 3) + 4);
        
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