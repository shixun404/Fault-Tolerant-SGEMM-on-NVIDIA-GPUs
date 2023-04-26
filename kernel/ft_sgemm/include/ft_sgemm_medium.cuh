#include <stdio.h>

#define m 8
#define kk_max 1024
#define tab(t, a, b)t.x += a.x * b;t.y += a.y * b;  t.z += a.z * b;t.w += a.w * b;  
    
// #define tcab(t, c, alpha, beta) c = alpha * t + beta * c;
#define tcab(t, c, alpha, beta) \
    c.x = alpha * t.x + beta * c.x;\
    c.y = alpha * t.y + beta * c.y;\
    c.z = alpha * t.z + beta * c.z;\
    c.w = alpha * t.w + beta * c.w;
    
__global__  __launch_bounds__(256) void ft_sgemm_medium(int M, int N, int K, float *A, float *B, float *C, float alpha, float beta){
    // ms = ns = 32, ks = 8
    // mw = 16, nw = 32
    // mr = 4, nr = 4
    // thread num 64
    // blockId, warpId, and threadIdx
    int ms = 32, ns = 32, ks = 8, mw = 16, nw = 32, mr = 4, nr = 4;
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x; 
    // initial global read column
    int k = 0;
    // block row range: blockIdx.x * ms ~ blockIdx.x * ms + ms - 1
    // warp row id:  

    // global memory read
    // tile A size = ms x ks = 32 * 8, col major
    // tile B size = ns x ks = 32 * 8, row major
    // init double buffer with size ms * ks * 2 + ns * ks * 2 = 1024 in shared memory
    // [buffer_A_1, buffer_A_2, buffer_B_1, buffer_B_2]
    __shared__ float sAB[1024]; 
    int buffer_A_offset = 0;
    int buffer_B_offset = 2 * ms * ks;
    // tile A global offset
    // block bx read tile A with rows in [bx * ms, bx * ms + ms - 1]
    A += bx * ms;

    // tile B global offset
    // block bx read tile A with rows in [bx * ms, bx * ms + ms - 1]
    B += by * ns;

    // tile A inner offset.
    // Each thread load (32 * 8) / 64 = 4 floats from A.
    int load_tile_A_num_floats_one_thread = (int)((ms * ks) / blockDim.x);
    // number of threads to load a column of tile A: 32 floats / 4 floats = 8 threads,
    int load_tile_A_num_threads_one_col = (int)(ms / load_tile_A_num_floats_one_thread);
    // thread tx load 4 floats with rows = [(tx % 4 threads) * 4, (tx % 4  threads) * 4 + 3],
    //                              col  = (tx / 4 threads) of tile A
    A += (tx % load_tile_A_num_threads_one_col) * (load_tile_A_num_floats_one_thread) + (int)(tx / load_tile_A_num_threads_one_col) * M;

    // tile B inner offset.
    // each thread load (32 * 8) / 64 = 4 floats from B.
    int load_tile_B_num_floats_one_thread = (int)((ns * ks) / blockDim.x);
    // number of threads to load a column of tile B: 32 floats / 4 floats = 8 threads,
    int load_tile_B_num_threads_one_col = (int)(ns / load_tile_B_num_floats_one_thread);
    // thread tx load 4 floats with rows = [(tx % 8 threads) * 4, (tx % 8  threads) * 4 + 3],
    //                              col  = (tx / 8 threads) of tile B
    B += (tx % load_tile_B_num_threads_one_col) * (load_tile_B_num_floats_one_thread) + (int)(tx / load_tile_B_num_threads_one_col) * N;

    // prefetch the vector from A and B in global memory 
    float4 prefetch_vector_tile_A = *((float4*)A);
    float4 prefetch_vector_tile_B = *((float4*)B);

    // offset to store the prefetch vector
    int offset_store_prefetch = ((k / ks) & 1);
    
    // get the pointer to prefetched buffer A and prefetched buffer B
    float* buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
    float* buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;

    // store the vectors in the prefetched buffer A and prefetched buffer B
    *(((float4*)buffer_A) + tx) = prefetch_vector_tile_A;
    *(((float4*)buffer_B) + tx) = prefetch_vector_tile_B;

    __syncthreads();
    
    // warp size mw x nw (16 x 32)
    //           -----------------
    //          |      vec B      |
    //           -----------------                 
    //  -----    -----------------    -             -
    // |     |  |     warp 0      |   | mw = 16     | ms = 32
    // | vec |  |                 |   |             | 
    // |     |   -----------------    -             |
    // |  A  |  |     warp 1      |                 | 
    // |     |  |                 |                 |
    //  -----    -----------------                  -
    //              ns = nw = 32

    // numbers of warp along A vector and B vector
    int num_warp_A = int(ms / mw);
    int num_warp_B = int(ns / nw);
    
    // 1D warp id =  tx / 32
    int id_warp = (int)(tx / 32);
    
    // 2D warp arrangement, row major
    // 2D warp idB = 1D warp id % num_warp_B
    //         idA = 1D warp id / num_warp_B    
    int idB_warp = id_warp % num_warp_B;
    int idA_warp = int(id_warp / num_warp_B);
    
    // offset for the warp tile
    // offset vec A = 2D warp idA * mw
    // offset vec B = 2D warp idB * nw
    int offset_vec_A_warp = idA_warp * mw;
    int offset_vec_B_warp = idB_warp * nw;

    // inner warp thread arrangement 1, row major
    //                warp 0
    //      --------------------------             -
    //     |  0  1  2  3  4  5  6  7  |  mr = 4    |  mw = 16  
    //     |  8  9 10 11 12 13 14 15  |            |
    //     | 16 17 18 19 20 21 22 23  |            |
    //     | 24 25 26 27 28 29 30 31  |            |
    //      --------------------------             -
    //      nr = 4
    //      nw = nr * 8 = 32

    //2D thread idB = tx % (nw / nr)
    //          idA = tx / (nw / nr)
    int idB_thread = ((tx & 31) % ((int)(nw / nr)));
    int idA_thread = int((tx & 31) / (nw / nr));

    // offset for the threads
    // offset vec A = 2D thread idA * mr
    // offset vec B = 2D thread idA * nr
    int offset_vec_A_thread = idA_thread * mr;
    int offset_vec_B_thread = idB_thread * nr;

    // load two vectors with size 4 from buffer A and buffer B into registers
    // initial the registers, to store two vectors with size mr and nr
    // prefetch with the double buffer
    float4 vec_A[2];
    float4 vec_B[2];
    float res[16];
    memset(res, 0, sizeof(res));
    // initial outer product column
    int kk = -1;
      
    // offset of register store for prefetching
    int offset_prefetch_register_kk = ((kk + 1) & 1);
    
    // offset of register to use 
    int offset_register_kk = 0;
    
    // offset of vec A and vec B w.r.t kk:
    int offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
    int offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
    
    // load the vectors from buffer to registers
    vec_A[offset_prefetch_register_kk] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
    vec_B[offset_prefetch_register_kk] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);
    

    // ABFT
    float4 block_level_A_c = {0., 0., 0., 0.}, block_level_B_r = {0., 0., 0., 0.};
    float A_c = prefetch_vector_tile_A.x + prefetch_vector_tile_A.y + prefetch_vector_tile_A.z + prefetch_vector_tile_A.w;
    float B_r = prefetch_vector_tile_B.x + prefetch_vector_tile_B.y + prefetch_vector_tile_B.z + prefetch_vector_tile_B.w;
    A_c += __shfl_xor_sync(0xffffffff, A_c, 1, 32);
    A_c += __shfl_xor_sync(0xffffffff, A_c, 2, 32);
    A_c += __shfl_xor_sync(0xffffffff, A_c, 4, 32);
    B_r += __shfl_xor_sync(0xffffffff, B_r, 1, 32);
    B_r += __shfl_xor_sync(0xffffffff, B_r, 2, 32);
    B_r += __shfl_xor_sync(0xffffffff, B_r, 4, 32);
    
    // saxpy
    block_level_A_c.x += prefetch_vector_tile_B.x * A_c;
    block_level_A_c.y += prefetch_vector_tile_B.y * A_c;
    block_level_A_c.z += prefetch_vector_tile_B.z * A_c;
    block_level_A_c.w += prefetch_vector_tile_B.w * A_c;

    block_level_B_r.x += prefetch_vector_tile_A.x * B_r;
    block_level_B_r.y += prefetch_vector_tile_A.y * B_r;
    block_level_B_r.z += prefetch_vector_tile_A.z * B_r;
    block_level_B_r.w += prefetch_vector_tile_A.w * B_r;

    // store into buffer

    // offset to store the saxpy result
    int offset_store_checksum = (((k / ks) + 1) & 1);
    
    // get the pointer to prefetched buffer A and prefetched buffer B
    float* checksum_buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_checksum * ms * ks;
    float* checksum_buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_checksum * ns * ks;

    *(((float4*)checksum_buffer_A) + tx) = block_level_A_c;
    *(((float4*)checksum_buffer_B) + tx) = block_level_B_r;

    __syncthreads();
    // offset C checksum each thread
    int offset_A_B = (tx >= (blockDim.x / 2)) ? (buffer_A_offset + offset_store_checksum * ms * ks): (buffer_B_offset + offset_store_checksum * ns * ks);
    float checksum = 0.;
    checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 0);
    checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 1);
    checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 2);
    checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 3);
    checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 4);
    checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 5);
    checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 6);
    checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 7);

    // K loop
    for(k = 0; k < K; k += ks){
        // tile A abd tile B global offsets move forward ks columns
        A += ks * M; 
        B += ks * N; 
        // prefetch the vector from A and B in global memory 
        prefetch_vector_tile_A = *((float4*)A);
        prefetch_vector_tile_B = *((float4*)B);

        // inner k loop, 8
        for(kk = 0; kk < ks; ++kk){
            offset_register_kk = ((kk) & 1);
            offset_prefetch_register_kk = ((kk + 1) & 1);
    
            // offset of vec A and vec B w.r.t kk:
            offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
            offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
            
            // load the vectors from buffer to registers
            vec_A[offset_prefetch_register_kk] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
            vec_B[offset_prefetch_register_kk] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);

            res[ 0] += vec_A[offset_register_kk].x * vec_B[offset_register_kk].x;
            res[ 1] += vec_A[offset_register_kk].x * vec_B[offset_register_kk].y;
            res[ 2] += vec_A[offset_register_kk].x * vec_B[offset_register_kk].z;
            res[ 3] += vec_A[offset_register_kk].x * vec_B[offset_register_kk].w;

            res[ 4] += vec_A[offset_register_kk].y * vec_B[offset_register_kk].x;
            res[ 5] += vec_A[offset_register_kk].y * vec_B[offset_register_kk].y;
            res[ 6] += vec_A[offset_register_kk].y * vec_B[offset_register_kk].z;
            res[ 7] += vec_A[offset_register_kk].y * vec_B[offset_register_kk].w;

            res[ 8] += vec_A[offset_register_kk].z * vec_B[offset_register_kk].x;
            res[ 9] += vec_A[offset_register_kk].z * vec_B[offset_register_kk].y;
            res[10] += vec_A[offset_register_kk].z * vec_B[offset_register_kk].z;
            res[11] += vec_A[offset_register_kk].z * vec_B[offset_register_kk].w;

            res[12] += vec_A[offset_register_kk].w * vec_B[offset_register_kk].x;
            res[13] += vec_A[offset_register_kk].w * vec_B[offset_register_kk].y;
            res[14] += vec_A[offset_register_kk].w * vec_B[offset_register_kk].z;
            res[15] += vec_A[offset_register_kk].w * vec_B[offset_register_kk].w;
        }
        if(k % 256 == 0 || k == K-1){
            res[(tx&1)] += checksum;
        }
        
        // update offset to store the prefetch vector
        offset_store_prefetch = (((int)(k / ks) + 1) & 1);
        
        // update the pointer to prefetched buffer A and prefetched buffer B
        buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
        buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;
        
        // store the vectors in the prefetched buffer A and prefetched buffer B
        *(((float4*)buffer_A) + tx) = prefetch_vector_tile_A;
        *(((float4*)buffer_B) + tx) = prefetch_vector_tile_B;
        __syncthreads();
        // initial outer product column
        kk = -1;
        
        // offset of register store for prefetching
        offset_prefetch_register_kk = ((kk + 1) & 1);
        
        // offset of vec A and vec B w.r.t kk:
        offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
        offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
        
        // ABFT
        A_c = prefetch_vector_tile_A.x + prefetch_vector_tile_A.y + prefetch_vector_tile_A.z + prefetch_vector_tile_A.w;
        B_r = prefetch_vector_tile_B.x + prefetch_vector_tile_B.y + prefetch_vector_tile_B.z + prefetch_vector_tile_B.w;
        A_c += __shfl_xor_sync(0xffffffff, A_c, 1, 32);
        A_c += __shfl_xor_sync(0xffffffff, A_c, 2, 32);
        A_c += __shfl_xor_sync(0xffffffff, A_c, 4, 32);
        B_r += __shfl_xor_sync(0xffffffff, B_r, 1, 32);
        B_r += __shfl_xor_sync(0xffffffff, B_r, 2, 32);
        B_r += __shfl_xor_sync(0xffffffff, B_r, 4, 32);
        
        // saxpy
        block_level_A_c.x = prefetch_vector_tile_B.x * A_c;
        block_level_A_c.y = prefetch_vector_tile_B.y * A_c;
        block_level_A_c.z = prefetch_vector_tile_B.z * A_c;
        block_level_A_c.w = prefetch_vector_tile_B.w * A_c;

        block_level_B_r.x = prefetch_vector_tile_A.x * B_r;
        block_level_B_r.y = prefetch_vector_tile_A.y * B_r;
        block_level_B_r.z = prefetch_vector_tile_A.z * B_r;
        block_level_B_r.w = prefetch_vector_tile_A.w * B_r;

        // store into buffer

        // offset to store the saxpy result
        offset_store_checksum = (((k / ks)) & 1);
        
        // get the pointer to prefetched buffer A and prefetched buffer B
        checksum_buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_checksum * ms * ks;
        checksum_buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_checksum * ns * ks;

        *(((float4*)checksum_buffer_A) + tx) = block_level_A_c;
        *(((float4*)checksum_buffer_B) + tx) = block_level_B_r;
        
        
        __syncthreads();
        
        // offset C checksum each thread
        // 
        offset_A_B = (tx >= (blockDim.x / 2)) ? (buffer_A_offset + offset_store_checksum * ms * ks): (buffer_B_offset + offset_store_checksum * ns * ks);
        checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 0);
        checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 1);
        checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 2);
        checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 3);
        checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 4);
        checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 5);
        checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 6);
        checksum += *((float*)(sAB) + offset_A_B + (tx & (ms - 1)) + ms * 7);
        
        // load the vectors from buffer to registers
        vec_A[offset_prefetch_register_kk] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
        vec_B[offset_prefetch_register_kk] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);

        

    }
    
    C += bx * ms + offset_vec_A_warp + offset_vec_A_thread;
    C += (by * ns + offset_vec_B_warp + offset_vec_B_thread) * M;

    float4 C_res[4];
    C_res[0] = *((float4 *)C);
    C_res[1] = *((float4 *)(C + M));
    C_res[2] = *((float4 *)(C + 2 * M));
    C_res[3] = *((float4 *)(C + 3 * M));

    C_res[0].x = alpha * res[0 ] + beta * C_res[0].x;
    C_res[0].y = alpha * res[4 ] + beta * C_res[0].y;
    C_res[0].z = alpha * res[8 ] + beta * C_res[0].z;
    C_res[0].w = alpha * res[12] + beta * C_res[0].w;
    
    C_res[1].x = alpha * res[1 ] + beta * C_res[1].x;
    C_res[1].y = alpha * res[5 ] + beta * C_res[1].y;
    C_res[1].z = alpha * res[9 ] + beta * C_res[1].z;
    C_res[1].w = alpha * res[13] + beta * C_res[1].w;

    C_res[2].x = alpha * res[2 ] + beta * C_res[2].x;
    C_res[2].y = alpha * res[6 ] + beta * C_res[2].y;
    C_res[2].z = alpha * res[10] + beta * C_res[2].z;
    C_res[2].w = alpha * res[14] + beta * C_res[2].w;

    C_res[3].x = alpha * res[3 ] + beta * C_res[3].x;
    C_res[3].y = alpha * res[7 ] + beta * C_res[3].y;
    C_res[3].z = alpha * res[11] + beta * C_res[3].z;
    C_res[3].w = alpha * res[15] + beta * C_res[3].w;

    *((float4 *)C) = C_res[0];
    *((float4 *)(C + M)) = C_res[1];
    *((float4 *)(C + 2 * M)) = C_res[2];
    *((float4 *)(C + 3 * M)) = C_res[3];
}