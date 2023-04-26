#include <stdio.h>

#define m 8
#define kk_max 1024
#define ms_tall 128
#define ns_tall 32
#define ks_tall 8
#define mw_tall 64
#define nw_tall 16
#define mr_tall 8
#define nr_tall 4
#define tab(t, a, b)t.x += a.x * b;t.y += a.y * b;  t.z += a.z * b;t.w += a.w * b;  
    
// #define tcab(t, c, alpha, beta) c = alpha * t + beta * c;
#define tcab(t, c, alpha, beta) \
    c.x = alpha * t.x + beta * c.x;\
    c.y = alpha * t.y + beta * c.y;\
    c.z = alpha * t.z + beta * c.z;\
    c.w = alpha * t.w + beta * c.w;
    
__global__  __launch_bounds__(256) void ft_sgemm_tall_struct(int N, int K, float *A, float *B, float *C, float alpha, float beta){
    // ms_tall = 128, ns_tall = 32, ks_tall = 8
    // mw_tall = 64, nw_tall = 16
    // mr_tall = 8, nr_tall = 4
    // blockId, warpId, and threadIdx
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x; 
    // initial global read column
    int k = 0;
    // block row range: blockIdx.x * ms_tall ~ blockIdx.x * ms_tall + ms_tall - 1
    // warp row id:  

    // global memory read
    // tile A size = ms_tall x ks_tall = 64 * 8, col major
    // tile B size = ns_tall x ks_tall = 64 * 8, row major
    // init double buffer with size ms_tall * ks_tall * 2 + ns_tall * ks_tall * 2 = 2560 in shared memory
    // [buffer_A_1, buffer_A_2, buffer_B_1, buffer_B_2]
    __shared__ float sAB[ms_tall * ks_tall * 2 + ns_tall * ks_tall * 2]; 
    int buffer_A_offset = 0;
    int buffer_B_offset = 2 * ms_tall * ks_tall;
    // tile A global offset
    // block bx read tile A with rows in [bx * ms_tall, bx * ms_tall + ms_tall - 1]
    A += bx * ms_tall;

    // tile B global offset
    // block bx read tile A with rows in [bx * ms_tall, bx * ms_tall + ms_tall - 1]
    B += by * ns_tall;

    // tile A inner offset.
    // Each thread load (128 * 8) / 128 = 8 floats from A.
    int load_tile_A_num_floats_one_thread = (int)((ms_tall * ks_tall) / blockDim.x);
    // number of threads to load a column of tile A: 128 floats / 8 floats = 16 threads,
    int load_tile_A_num_threads_one_col = (int)(ms_tall / load_tile_A_num_floats_one_thread);
    // thread tx load 8 floats with rows = [(tx % 16 threads) * 8, (tx % 16 threads) * 8 + 7],
    //                              col  = (tx / 16 threads) of tile A
    A += (tx % load_tile_A_num_threads_one_col) * (load_tile_A_num_floats_one_thread) + (int)(tx / load_tile_A_num_threads_one_col) * N;

    // tile B inner offset.
    // each thread load (32 * 8) / 128 = 2 floats from B.
    int load_tile_B_num_floats_one_thread = (int)((ns_tall * ks_tall) / blockDim.x);
    // number of threads to load a column of tile B: 32 floats / 2 floats = 16 threads,
    int load_tile_B_num_threads_one_col = (int)(ns_tall / load_tile_B_num_floats_one_thread);
    // thread tx load 8 floats with rows = [(tx % 16 threads) * 2, (tx % 16 threads) * 2 + 1],
    //                              col  = (tx / 16 threads) of tile A
    B += (tx % load_tile_B_num_threads_one_col) * (load_tile_B_num_floats_one_thread) + (int)(tx / load_tile_B_num_threads_one_col) * N;

    // prefetch the vector from A and B in global memory 
    // 
    // float{4 if ms * ks / num_thread >= 4 else 2} prefetch_vector_tile_A[{ms * ks / (4 * num_thread)}];
    // float{4 if ns * ks / num_thread >= 4 else 2} prefetch_vector_tile_B[{ns * ks / (4 * num_thread)}]
    float4 prefetch_vector_tile_A[2];
    float2 prefetch_vector_tile_B;
    prefetch_vector_tile_A[0] = *((float4*)A);
    prefetch_vector_tile_A[1] = *((float4*)A + 1);
    prefetch_vector_tile_B    = *((float2*)B);

    // offset to store the prefetch vector
    int offset_store_prefetch = ((k / ks_tall) & 1);
    
    // get the pointer to prefetched buffer A and prefetched buffer B
    float* buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms_tall * ks_tall;
    float* buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns_tall * ks_tall;

    // store the vectors in the prefetched buffer A and prefetched buffer B
    *(((float4*)buffer_A) + 2 * tx) = prefetch_vector_tile_A[0];
    *(((float4*)buffer_A) + 2 * tx + 1) = prefetch_vector_tile_A[1];
    *(((float2*)buffer_B) + tx) = prefetch_vector_tile_B;

    __syncthreads();
    
    // warp size mw_tall x nw_tall (64 x 16)
    //           ----------------------------------
    //          |               vec B              |      
    //           ----------------------------------                 
    //  -----    ----------------- -----------------   -             -
    // |     |  |     warp 0      |     warp 1      |  | mw_tall = 64     | ms_tall = 128
    // | vec |  |                 |                 |  |             | 
    // |     |   ----------------- -----------------   -             |
    // |  A  |  |     warp 2      |     warp 3      |                |   
    // |     |  |                 |                 |                |
    //  -----    ----------------- -----------------                 -
    //           nw_tall = 16
    //          |-----------------|
    //          
    //          |-----------------------------------|
    //                         ns_tall = 32

    // numbers of warp along A vector and B vector
    int num_warp_A = int(ms_tall / mw_tall);
    int num_warp_B = int(ns_tall / nw_tall);
    
    // 1D warp id =  tx / 32
    int id_warp = (int)(tx / 32);
    
    // 2D warp arrangement, row major
    // 2D warp idB = 1D warp id % num_warp_B
    //         idA = 1D warp id / num_warp_B    
    int idB_warp = id_warp % num_warp_B;
    int idA_warp = int(id_warp / num_warp_B);
    
    // offset for the warp tile
    // offset vec A = 2D warp idA * mw_tall
    // offset vec B = 2D warp idB * nw_tall
    int offset_vec_A_warp = idA_warp * mw_tall;
    int offset_vec_B_warp = idB_warp * nw_tall;

    // inner warp thread arrangement 1, row major
    //          warp 0
    //      --------------              -
    //     |  0  1  2  3  |   mr_tall = 8    |  mw_tall = 64  
    //     |  4  5  6  7  |             |
    //     |  8  9 10 11  |             |
    //     | 12 13 14 15  |             |    
    //     | 16 17 18 19  |             |
    //     | 20 21 22 23  |             |
    //     | 24 25 26 27  |             |
    //     | 28 29 30 31  |             |
    //      --------------              -
    //      nr_tall = 4
    //      nw_tall = nr_tall * 4 = 16

    //2D thread idB = tx % (nw_tall / nr_tall)
    //          idA = tx / (nw_tall / nr_tall)
    int idB_thread = ((tx & 31) % ((int)(nw_tall / nr_tall)));
    int idA_thread = int((tx & 31) / (nw_tall / nr_tall));

    // offset for the threads
    // offset vec A = 2D thread idA * mr_tall
    // offset vec B = 2D thread idA * nr_tall
    int offset_vec_A_thread = idA_thread * mr_tall;
    int offset_vec_B_thread = idB_thread * nr_tall;

    // load two vectors with size 4 from buffer A and buffer B into registers
    // initial the registers, to store two vectors with size mr_tall and nr_tall
    // prefetch with the double buffer
    float4 vec_A[4];
    float4 vec_B[2];
    float res[32];
    memset(res, 0, sizeof(res));
    // initial outer product column
    int kk = -1;
      
    // offset of register store for prefetching
    int offset_prefetch_register_kk = ((kk + 1) & 1);
    
    // offset of register to use 
    int offset_register_kk = 0;
    
    // offset of vec A and vec B w.r.t kk:
    int offset_load_vec_A_kk = ((kk + 1) % ks_tall) * ms_tall;
    int offset_load_vec_B_kk = ((kk + 1) % ks_tall) * ns_tall;
    
    // load the vectors from buffer to registers
    vec_A[offset_prefetch_register_kk * 2    ] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
    vec_A[offset_prefetch_register_kk * 2 + 1] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk + 4);
    vec_B[offset_prefetch_register_kk        ] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);
    

    // ABFT
    float4 block_level_A_c[2];
    float2 block_level_B_r[1];
    float A_c = prefetch_vector_tile_A[0].x + prefetch_vector_tile_A[0].y + prefetch_vector_tile_A[0].z + prefetch_vector_tile_A[0].w;
    A_c += prefetch_vector_tile_A[1].x + prefetch_vector_tile_A[1].y + prefetch_vector_tile_A[1].z + prefetch_vector_tile_A[1].w;
    float B_r = prefetch_vector_tile_B.x + prefetch_vector_tile_B.y;
    A_c += __shfl_xor_sync(0xffffffff, A_c, 1, 32);
    A_c += __shfl_xor_sync(0xffffffff, A_c, 2, 32);
    A_c += __shfl_xor_sync(0xffffffff, A_c, 4, 32);
    A_c += __shfl_xor_sync(0xffffffff, A_c, 8, 32);
    B_r += __shfl_xor_sync(0xffffffff, B_r, 1, 32);
    B_r += __shfl_xor_sync(0xffffffff, B_r, 2, 32);
    B_r += __shfl_xor_sync(0xffffffff, B_r, 4, 32);
    B_r += __shfl_xor_sync(0xffffffff, B_r, 8, 32);
    
    // saxpy
    block_level_B_r[0].x = prefetch_vector_tile_B.x * A_c;
    block_level_B_r[0].y = prefetch_vector_tile_B.y * A_c;

    block_level_A_c[0].x = prefetch_vector_tile_A[0].x * B_r;
    block_level_A_c[0].y = prefetch_vector_tile_A[0].y * B_r;
    block_level_A_c[0].z = prefetch_vector_tile_A[0].z * B_r;
    block_level_A_c[0].w = prefetch_vector_tile_A[0].w * B_r;

    block_level_A_c[1].x = prefetch_vector_tile_A[1].x * B_r;
    block_level_A_c[1].y = prefetch_vector_tile_A[1].y * B_r;
    block_level_A_c[1].z = prefetch_vector_tile_A[1].z * B_r;
    block_level_A_c[1].w = prefetch_vector_tile_A[1].w * B_r;

    // store into buffer

    // offset to store the saxpy result
    int offset_store_checksum = (((k / ks_tall) + 1) & 1);
    
    // get the pointer to prefetched buffer A and prefetched buffer B
    float* checksum_buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_checksum * ms_tall * ks_tall;
    float* checksum_buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_checksum * ns_tall * ks_tall;

    *(((float4*)checksum_buffer_A) + tx * 2) = block_level_A_c[0];
    *(((float4*)checksum_buffer_A) + tx * 2 + 1) = block_level_A_c[1];
    *(((float2*)checksum_buffer_B) + tx) = block_level_B_r[0];

    __syncthreads(); 
    // offset C checksum each thread
    int offset_A_B = (tx < (3 * blockDim.x / 4)) ? (buffer_A_offset + offset_store_checksum * ms_tall * ks_tall): (buffer_B_offset + offset_store_checksum * ns_tall * ks_tall);
    int ws = (tx < (3 * blockDim.x / 4)) ? ms_tall: ns_tall;
    int ws_1 = 2;//(tx < (3 * blockDim.x / 4)) ? 2: 2;
    int ws_2 = 1;//(tx < (3 * blockDim.x / 4)) ? 1: 1;
    offset_A_B +=  (tx & (int)(ws / ws_1 - 1)) * ws_1;
    float checksum[2] = {0., 0.};
    float2 tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 0));
    checksum[0] +=  tmp.x;
    checksum[1] +=  tmp.y;
    tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 1));
    checksum[0] +=  tmp.x;
    checksum[1] +=  tmp.y;
    tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 2));
    checksum[0] +=  tmp.x;
    checksum[1] +=  tmp.y;
    tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 3));
    checksum[0] +=  tmp.x;
    checksum[1] +=  tmp.y;
    tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 4));
    checksum[0] +=  tmp.x;
    checksum[1] +=  tmp.y;
    tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 5));
    checksum[0] +=  tmp.x;
    checksum[1] +=  tmp.y;
    tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 6));
    checksum[0] +=  tmp.x;
    checksum[1] +=  tmp.y;
    tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 7));
    checksum[0] +=  tmp.x;
    checksum[1] +=  tmp.y;


    
    // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 0));
    // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 0));
    // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 1));
    // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 1));
    // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 2));
    // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 2));
    // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 3));
    // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 3));
    // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 4));
    // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 4));
    // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 5));
    // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 5));
    // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 6));
    // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 6));
    // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 7));
    // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 7));


    // K loop
    for(k = 0; k < K; k += ks_tall){
        // tile A abd tile B global offsets move forward ks_tall columns
        A += ks_tall * N; 
        B += ks_tall * N; 
        // prefetch the vector from A and B in global memory 
        prefetch_vector_tile_A[0] = *((float4*)A);  
        prefetch_vector_tile_A[1] = *((float4*)A + 1);
        prefetch_vector_tile_B    = *((float2*)B);

        // inner k loop, 8
        for(kk = 0; kk < ks_tall; ++kk){
            offset_register_kk = ((kk) & 1);
            offset_prefetch_register_kk = ((kk + 1) & 1);
    
            // offset of vec A and vec B w.r.t kk:
            offset_load_vec_A_kk = ((kk + 1) % ks_tall) * ms_tall;
            offset_load_vec_B_kk = ((kk + 1) % ks_tall) * ns_tall;
            
            // load the vectors from buffer to registers
            vec_A[offset_prefetch_register_kk * 2    ] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
            vec_A[offset_prefetch_register_kk * 2 + 1] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk + 4);
            vec_B[offset_prefetch_register_kk        ] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);

            res[ 0] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk].x;
            res[ 1] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk].y;
            res[ 2] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk].z;
            res[ 3] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk].w;

            res[ 4] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk].x;
            res[ 5] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk].y;
            res[ 6] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk].z;
            res[ 7] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk].w;

            res[ 8] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk].x;
            res[ 9] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk].y;
            res[10] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk].z;
            res[11] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk].w;

            res[12] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk].x;
            res[13] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk].y;
            res[14] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk].z;
            res[15] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk].w;

            res[16] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk].x;
            res[17] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk].y;
            res[18] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk].z;
            res[19] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk].w;

            res[20] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk].x;
            res[21] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk].y;
            res[22] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk].z;
            res[23] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk].w;

            res[24] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk].x;
            res[25] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk].y;
            res[26] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk].z;
            res[27] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk].w;

            res[28] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk].x;
            res[29] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk].y;
            res[30] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk].z;
            res[31] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk].w;
        }
        
        if(k % 256 == 0){
            res[(tx&1)] += checksum[0] + checksum[1];
        }

        // update offset to store the prefetch vector
        offset_store_prefetch = (((int)(k / ks_tall) + 1) & 1);
        
        // update the pointer to prefetched buffer A and prefetched buffer B
        buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms_tall * ks_tall;
        buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns_tall * ks_tall;
        
        
        // store the vectors in the prefetched buffer A and prefetched buffer B
        *(((float4*)buffer_A) + 2 * tx) = prefetch_vector_tile_A[0];
        *(((float4*)buffer_A) + 2 * tx + 1) = prefetch_vector_tile_A[1];
        *(((float2*)buffer_B) + tx) = prefetch_vector_tile_B;
        __syncthreads();
        // initial outer product column
        kk = -1;
        
        // offset of register store for prefetching
        offset_prefetch_register_kk = ((kk + 1) & 1);
        
        // offset of vec A and vec B w.r.t kk:
        offset_load_vec_A_kk = ((kk + 1) % ks_tall) * ms_tall;
        offset_load_vec_B_kk = ((kk + 1) % ks_tall) * ns_tall;
        
        // load the vectors from buffer to registers
        vec_A[offset_prefetch_register_kk * 2    ] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
        vec_A[offset_prefetch_register_kk * 2 + 1] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk + 4);
        vec_B[offset_prefetch_register_kk        ] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);
        block_level_A_c[2];
        block_level_B_r[1];
        A_c = prefetch_vector_tile_A[0].x + prefetch_vector_tile_A[0].y + prefetch_vector_tile_A[0].z + prefetch_vector_tile_A[0].w;
        A_c += prefetch_vector_tile_A[1].x + prefetch_vector_tile_A[1].y + prefetch_vector_tile_A[1].z + prefetch_vector_tile_A[1].w;
        B_r = prefetch_vector_tile_B.x + prefetch_vector_tile_B.y;
        A_c += __shfl_xor_sync(0xffffffff, A_c, 1, 32);
        A_c += __shfl_xor_sync(0xffffffff, A_c, 2, 32);
        A_c += __shfl_xor_sync(0xffffffff, A_c, 4, 32);
        A_c += __shfl_xor_sync(0xffffffff, A_c, 8, 32);
        B_r += __shfl_xor_sync(0xffffffff, B_r, 1, 32);
        B_r += __shfl_xor_sync(0xffffffff, B_r, 2, 32);
        B_r += __shfl_xor_sync(0xffffffff, B_r, 4, 32);
        B_r += __shfl_xor_sync(0xffffffff, B_r, 8, 32);
        
        // saxpy
        block_level_B_r[0].x = prefetch_vector_tile_B.x * A_c;
        block_level_B_r[0].y = prefetch_vector_tile_B.y * A_c;

        block_level_A_c[0].x = prefetch_vector_tile_A[0].x * B_r;
        block_level_A_c[0].y = prefetch_vector_tile_A[0].y * B_r;
        block_level_A_c[0].z = prefetch_vector_tile_A[0].z * B_r;
        block_level_A_c[0].w = prefetch_vector_tile_A[0].w * B_r;

        block_level_A_c[1].x = prefetch_vector_tile_A[1].x * B_r;
        block_level_A_c[1].y = prefetch_vector_tile_A[1].y * B_r;
        block_level_A_c[1].z = prefetch_vector_tile_A[1].z * B_r;
        block_level_A_c[1].w = prefetch_vector_tile_A[1].w * B_r;

        // store into buffer

        // offset to store the saxpy result
        offset_store_checksum = (((k / ks_tall) + 1) & 1);
        
        // get the pointer to prefetched buffer A and prefetched buffer B
        checksum_buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_checksum * ms_tall * ks_tall;
        checksum_buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_checksum * ns_tall * ks_tall;

        *(((float4*)checksum_buffer_A) + tx * 2) = block_level_A_c[0];
        *(((float4*)checksum_buffer_A) + tx * 2 + 1) = block_level_A_c[1];
        *(((float2*)checksum_buffer_B) + tx) = block_level_B_r[0];

        __syncthreads();
        // offset C checksum each thread
        offset_A_B = (tx < (3 * blockDim.x / 4)) ? (buffer_A_offset + offset_store_checksum * ms_tall * ks_tall): (buffer_B_offset + offset_store_checksum * ns_tall * ks_tall);
        offset_A_B +=  (tx & (int)(ws / ws_1 - 1)) * ws_1;
        
        tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 0));
        checksum[0] +=  tmp.x;
        checksum[1] +=  tmp.y;
        tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 1));
        checksum[0] +=  tmp.x;
        checksum[1] +=  tmp.y;
        tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 2));
        checksum[0] +=  tmp.x;
        checksum[1] +=  tmp.y;
        tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 3));
        checksum[0] +=  tmp.x;
        checksum[1] +=  tmp.y;
        tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 4));
        checksum[0] +=  tmp.x;
        checksum[1] +=  tmp.y;
        tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 5));
        checksum[0] +=  tmp.x;
        checksum[1] +=  tmp.y;
        tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 6));
        checksum[0] +=  tmp.x;
        checksum[1] +=  tmp.y;
        tmp = *((float2*)((float*)(sAB) + offset_A_B + ws * 7));
        checksum[0] +=  tmp.x;
        checksum[1] +=  tmp.y;


        // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 0));
        // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 0));
        // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 1));
        // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 1));
        // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 2));
        // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 2));
        // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 3));
        // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 3));
        // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 4));
        // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 4));
        // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 5));
        // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 5));
        // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 6));
        // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 6));
        // checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 7));
        // checksum[1] +=  *(((float*)(sAB) + offset_A_B + ws_2 + ws * 7));
    }
    
    C += bx * ms_tall + offset_vec_A_warp + offset_vec_A_thread;
    C += (by * ns_tall + offset_vec_B_warp + offset_vec_B_thread) * N;

    float4 C_res[8];
    C_res[ 0] = *((float4 *)(C + 0 + N * 0));
    C_res[ 1] = *((float4 *)(C + 4 + N * 0));
    C_res[ 2] = *((float4 *)(C + 0 + N * 1));
    C_res[ 3] = *((float4 *)(C + 4 + N * 1));
    C_res[ 4] = *((float4 *)(C + 0 + N * 2));
    C_res[ 5] = *((float4 *)(C + 4 + N * 2));
    C_res[ 6] = *((float4 *)(C + 0 + N * 3));
    C_res[ 7] = *((float4 *)(C + 4 + N * 3));
    

    C_res[0].x = alpha * res[0 ] + beta * C_res[0].x;
    C_res[0].y = alpha * res[4 ] + beta * C_res[0].y;
    C_res[0].z = alpha * res[8 ] + beta * C_res[0].z;
    C_res[0].w = alpha * res[12] + beta * C_res[0].w;

    C_res[1].x = alpha * res[16] + beta * C_res[1].x;
    C_res[1].y = alpha * res[20] + beta * C_res[1].y;
    C_res[1].z = alpha * res[24] + beta * C_res[1].z;
    C_res[1].w = alpha * res[28] + beta * C_res[1].w;

    C_res[2].x = alpha * res[1 ] + beta * C_res[2].x;
    C_res[2].y = alpha * res[5 ] + beta * C_res[2].y;
    C_res[2].z = alpha * res[9 ] + beta * C_res[2].z;
    C_res[2].w = alpha * res[13] + beta * C_res[2].w;

    C_res[3].x = alpha * res[17] + beta * C_res[3].x;
    C_res[3].y = alpha * res[21] + beta * C_res[3].y;
    C_res[3].z = alpha * res[25] + beta * C_res[3].z;
    C_res[3].w = alpha * res[29] + beta * C_res[3].w;

    C_res[4].x = alpha * res[2 ] + beta * C_res[4].x;
    C_res[4].y = alpha * res[6 ] + beta * C_res[4].y;
    C_res[4].z = alpha * res[10] + beta * C_res[4].z;
    C_res[4].w = alpha * res[14] + beta * C_res[4].w;

    C_res[5].x = alpha * res[18] + beta * C_res[5].x;
    C_res[5].y = alpha * res[22] + beta * C_res[5].y;
    C_res[5].z = alpha * res[26] + beta * C_res[5].z;
    C_res[5].w = alpha * res[30] + beta * C_res[5].w;

    C_res[6].x = alpha * res[3 ] + beta * C_res[6].x;
    C_res[6].y = alpha * res[7 ] + beta * C_res[6].y;
    C_res[6].z = alpha * res[11] + beta * C_res[6].z;
    C_res[6].w = alpha * res[15] + beta * C_res[6].w;

    C_res[7].x = alpha * res[19] + beta * C_res[7].x;
    C_res[7].y = alpha * res[23] + beta * C_res[7].y;
    C_res[7].z = alpha * res[27] + beta * C_res[7].z;
    C_res[7].w = alpha * res[31] + beta * C_res[7].w;

    *((float4 *)(C + 0 + N * 0)) = C_res[ 0];
    *((float4 *)(C + 4 + N * 0)) = C_res[ 1];
    *((float4 *)(C + 0 + N * 1)) = C_res[ 2];
    *((float4 *)(C + 4 + N * 1)) = C_res[ 3];
    *((float4 *)(C + 0 + N * 2)) = C_res[ 4];
    *((float4 *)(C + 4 + N * 2)) = C_res[ 5];
    *((float4 *)(C + 0 + N * 3)) = C_res[ 6];
    *((float4 *)(C + 4 + N * 3)) = C_res[ 7];
}