 
#include <stdio.h>  
#define tab(t, a, b)t.x += a.x * b;t.y += a.y * b;  t.z += a.z * b;t.w += a.w * b;  

#define tcab(t, c, alpha, beta) \
    c.x = alpha * t.x + beta * c.x; \
    c.y = alpha * t.y + beta * c.y; \
    c.z = alpha * t.z + beta * c.z; \
    c.w = alpha * t.w + beta * c.w;
    
__global__  __launch_bounds__(128) void sgemm_tall(int M, int N, int K, float *A, float *B, float *C, float alpha, float beta){
    // ms = 128, ns = 32, ks = 8
    // mw = 64, nw = 16
    // mr = 8, nr = 4
    // blockId, warpId, and threadIdx
    
    int ms = 128, ns = 32, ks = 8, mw = 64, nw = 16, mr = 8, nr = 4;
    
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x; 
    // initial global read column
    int k = 0;
    // block row range: blockIdx.x * ms ~ blockIdx.x * ms + ms - 1
    // warp row id:  

    // global memory read
    // tile A size = ms x ks = 64 * 8, col major
    // tile B size = ns x ks = 64 * 8, row major
    // init double buffer with size ms * ks * 2 + ns * ks * 2 = 2560 in shared memory
    // [buffer_A_1, buffer_A_2, buffer_B_1, buffer_B_2]
    
    __shared__ float sAB[2560]; 
    
    int buffer_A_offset = 0;
    int buffer_B_offset = 2 * ms * ks;
    // tile A global offset
    // block bx read tile A with rows in [bx * ms, bx * ms + ms - 1]
    A += bx * ms;

    // tile B global offset
    // block bx read tile A with rows in [bx * ms, bx * ms + ms - 1]
    B += by * ns;

    // tile A inner offset.
    // Each thread load (128 * 8) / 128 = 8 floats from A.
    int load_tile_A_num_floats_one_thread = (int)((ms * ks) / blockDim.x);
    // number of threads to load a column of tile A: 128 floats / 8 floats = 16 threads,
    int load_tile_A_num_threads_one_col = (int)(ms / load_tile_A_num_floats_one_thread);
    // parameter for error injection
    int tx_injec = 17;
    float err_bound1 =9500.0;
    float error_inject = 10000.0;
    
    
    // thread tx load 8 floats with rows = [(tx % 16 threads) * 8, (tx % 16 threads) * 8 + 7],
    //                              col  = (tx / 16 threads) of tile A
    A += (tx % load_tile_A_num_threads_one_col) * (load_tile_A_num_floats_one_thread) + (int)(tx / load_tile_A_num_threads_one_col) * M;

    // tile B inner offset.
    // each thread load (32 * 8) / 128 = 2 floats from B.
    int load_tile_B_num_floats_one_thread = (int)((ns * ks) / blockDim.x);
    // number of threads to load a column of tile B: 32 floats / 2 floats = 16 threads,
    int load_tile_B_num_threads_one_col = (int)(ns / load_tile_B_num_floats_one_thread);
    // thread tx load 8 floats with rows = [(tx % 16 threads) * 2, (tx % 16 threads) * 2 + 1],
    //                              col  = (tx / 16 threads) of tile A
    B += (tx % load_tile_B_num_threads_one_col) * (load_tile_B_num_floats_one_thread) + (int)(tx / load_tile_B_num_threads_one_col) * N;

    // prefetch the vector from A and B in global memory 
    // 
    // float{4 if ms * ks / num_thread >= 4 else 2} prefetch_vector_tile_A[{ms * ks / (4 * num_thread)}];
    // float{4 if ns * ks / num_thread >= 4 else 2} prefetch_vector_tile_B[{ns * ks / (4 * num_thread)}]
    float4 prefetch_vector_tile_A[2];
    float2 prefetch_vector_tile_B[1];
    prefetch_vector_tile_A[0] = *((float4*)A + 0);
    prefetch_vector_tile_A[1] = *((float4*)A + 1);
    prefetch_vector_tile_B[0] = *((float2*)B + 0);
    
    // offset to store the prefetch vector
    int offset_store_prefetch = ((k / ks) & 1);
    
    // get the pointer to prefetched buffer A and prefetched buffer B
    float* buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
    float* buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;

    // store the vectors in the prefetched buffer A and prefetched buffer B
    *(((float4*)buffer_A) + 2 * tx + 0) = prefetch_vector_tile_A[0];
    *(((float4*)buffer_A) + 2 * tx + 1) = prefetch_vector_tile_A[1];
    *(((float2*)buffer_B) + 1 * tx + 0) = prefetch_vector_tile_B[0];
    
    __syncthreads();
    // numbers of warp along A vector and B vector
    int num_warp_A = int(ms / mw);
    int num_warp_B = int(ns / nw);
    
    // 1D warp id =  tx / 32
    int id_warp = (int)(tx / 32);
    
    // 2D warp arrangement, row major
    // 2D warp idB = 1D warp id % num_warp_B
    //         idA = 1D warp id / num_warp_B    
    int idB_warp = id_warp / num_warp_A;
    int idA_warp = int(id_warp % num_warp_A);
    
    // offset for the warp tile
    // offset vec A = 2D warp idA * mw
    // offset vec B = 2D warp idB * nw
    int offset_vec_A_warp = idA_warp * mw;
    int offset_vec_B_warp = idB_warp * nw;


    //2D thread idB = tx % (nw / nr)
    //          idA = tx / (nw / nr)
    int idB_thread = ((tx & 31) / ((int)(mw / mr)));
    int idA_thread = int((tx & 31) % (mw / mr));

    // offset for the threads
    // offset vec A = 2D thread idA * mr
    // offset vec B = 2D thread idA * nr
    int offset_vec_A_thread = idA_thread * mr;
    int offset_vec_B_thread = idB_thread * nr;

    // load two vectors with size 4 from buffer A and buffer B into registers
    // initial the registers, to store two vectors with size mr and nr
    // prefetch with the double buffer
    float4 vec_A[4];
    float4 vec_B[2];
    float4 tmp_row[2];
    float4 tmp_col[1];
    float res[32];
    float C_c[4];
    float C_r[8];
    
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
    vec_A[offset_prefetch_register_kk * 2 + 0] = *((float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + 0);
    vec_A[offset_prefetch_register_kk * 2 + 1] = *((float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + 1);
    vec_B[offset_prefetch_register_kk * 1 + 0] = *((float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk) + 0);
    
    __syncthreads(); 
    // K loop
    for(k = 0; k < K; k += ks){
        // tile A abd tile B global offsets move forward ks columns
        A += ks * M; 
        B += ks * N; 
        // prefetch the vector from A and B in global memory 
        prefetch_vector_tile_A[0] = *((float4*)A + 0);  
        prefetch_vector_tile_A[1] = *((float4*)A + 1);  
        prefetch_vector_tile_B[0] = *((float2*)B + 0);  
        
        // inner k loop, 8
        for(kk = 0; kk < ks; ++kk){
            offset_register_kk = ((kk) & 1);
            offset_prefetch_register_kk = ((kk + 1) & 1);
    
            // offset of vec A and vec B w.r.t kk:
            offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
            offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
            
            // load the vectors from buffer to registers
            vec_A[offset_prefetch_register_kk * 2 + 0] = *((float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + 0);
            vec_A[offset_prefetch_register_kk * 2 + 1] = *((float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + 1);
            vec_B[offset_prefetch_register_kk * 1 + 0] = *((float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk) + 0);
            
            res[0 ] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk * 1 + 0].x;
            res[1 ] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk * 1 + 0].y;
            res[2 ] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk * 1 + 0].z;
            res[3 ] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk * 1 + 0].w;
            
            res[4 ] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk * 1 + 0].x;
            res[5 ] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk * 1 + 0].y;
            res[6 ] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk * 1 + 0].z;
            res[7 ] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk * 1 + 0].w;
            
            res[8 ] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk * 1 + 0].x;
            res[9 ] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk * 1 + 0].y;
            res[10] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk * 1 + 0].z;
            res[11] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk * 1 + 0].w;
            
            res[12] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk * 1 + 0].x;
            res[13] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk * 1 + 0].y;
            res[14] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk * 1 + 0].z;
            res[15] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk * 1 + 0].w;
            
            res[16] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk * 1 + 0].x;
            res[17] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk * 1 + 0].y;
            res[18] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk * 1 + 0].z;
            res[19] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk * 1 + 0].w;
            
            res[20] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk * 1 + 0].x;
            res[21] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk * 1 + 0].y;
            res[22] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk * 1 + 0].z;
            res[23] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk * 1 + 0].w;
            
            res[24] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk * 1 + 0].x;
            res[25] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk * 1 + 0].y;
            res[26] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk * 1 + 0].z;
            res[27] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk * 1 + 0].w;
            
            res[28] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk * 1 + 0].x;
            res[29] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk * 1 + 0].y;
            res[30] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk * 1 + 0].z;
            res[31] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk * 1 + 0].w;
            
            
        }
            
        // update offset to store the prefetch vector
        offset_store_prefetch = (((int)(k / ks) + 1) & 1);
        
        // update the pointer to prefetched buffer A and prefetched buffer B
        buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
        buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;
        // store the vectors in the prefetched buffer A and prefetched buffer B
        *(((float4*)buffer_A) + 2 * tx + 0) = prefetch_vector_tile_A[0];
        *(((float4*)buffer_A) + 2 * tx + 1) = prefetch_vector_tile_A[1];
        *(((float2*)buffer_B) + 1 * tx + 0) = prefetch_vector_tile_B[0];
        __syncthreads();
        // initial outer product column
        kk = -1;
        
        // offset of register store for prefetching
        offset_prefetch_register_kk = ((kk + 1) & 1);
        
        // offset of vec A and vec B w.r.t kk:
        offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
        offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
        
        // load the vectors from buffer to registers
        vec_A[offset_prefetch_register_kk * 2 + 0] = *((float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + 0);
        vec_A[offset_prefetch_register_kk * 2 + 1] = *((float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + 1);
        vec_B[offset_prefetch_register_kk * 1 + 0] = *((float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk) + 0);
        
    __syncthreads(); 
    }
    
    C += bx * ms + offset_vec_A_warp + offset_vec_A_thread;
    C += (by * ns + offset_vec_B_warp + offset_vec_B_thread) * M;
    
    float4 C_res[8];
    
    C_res[0 ] = *((float4 *)(C+ M * 0) + 0 );
    C_res[1 ] = *((float4 *)(C+ M * 0) + 1 );
    C_res[2 ] = *((float4 *)(C+ M * 1) + 0 );
    C_res[3 ] = *((float4 *)(C+ M * 1) + 1 );
    C_res[4 ] = *((float4 *)(C+ M * 2) + 0 );
    C_res[5 ] = *((float4 *)(C+ M * 2) + 1 );
    C_res[6 ] = *((float4 *)(C+ M * 3) + 0 );
    C_res[7 ] = *((float4 *)(C+ M * 3) + 1 );
    
    C_res[0].x = alpha * res[0  ] + beta * C_res[0].x;
    C_res[0].y = alpha * res[4  ] + beta * C_res[0].y;
    C_res[0].z = alpha * res[8  ] + beta * C_res[0].z;
    C_res[0].w = alpha * res[12 ] + beta * C_res[0].w;
    
    C_res[1].x = alpha * res[16 ] + beta * C_res[1].x;
    C_res[1].y = alpha * res[20 ] + beta * C_res[1].y;
    C_res[1].z = alpha * res[24 ] + beta * C_res[1].z;
    C_res[1].w = alpha * res[28 ] + beta * C_res[1].w;
    
    C_res[2].x = alpha * res[1  ] + beta * C_res[2].x;
    C_res[2].y = alpha * res[5  ] + beta * C_res[2].y;
    C_res[2].z = alpha * res[9  ] + beta * C_res[2].z;
    C_res[2].w = alpha * res[13 ] + beta * C_res[2].w;
    
    C_res[3].x = alpha * res[17 ] + beta * C_res[3].x;
    C_res[3].y = alpha * res[21 ] + beta * C_res[3].y;
    C_res[3].z = alpha * res[25 ] + beta * C_res[3].z;
    C_res[3].w = alpha * res[29 ] + beta * C_res[3].w;
    
    C_res[4].x = alpha * res[2  ] + beta * C_res[4].x;
    C_res[4].y = alpha * res[6  ] + beta * C_res[4].y;
    C_res[4].z = alpha * res[10 ] + beta * C_res[4].z;
    C_res[4].w = alpha * res[14 ] + beta * C_res[4].w;
    
    C_res[5].x = alpha * res[18 ] + beta * C_res[5].x;
    C_res[5].y = alpha * res[22 ] + beta * C_res[5].y;
    C_res[5].z = alpha * res[26 ] + beta * C_res[5].z;
    C_res[5].w = alpha * res[30 ] + beta * C_res[5].w;
    
    C_res[6].x = alpha * res[3  ] + beta * C_res[6].x;
    C_res[6].y = alpha * res[7  ] + beta * C_res[6].y;
    C_res[6].z = alpha * res[11 ] + beta * C_res[6].z;
    C_res[6].w = alpha * res[15 ] + beta * C_res[6].w;
    
    C_res[7].x = alpha * res[19 ] + beta * C_res[7].x;
    C_res[7].y = alpha * res[23 ] + beta * C_res[7].y;
    C_res[7].z = alpha * res[27 ] + beta * C_res[7].z;
    C_res[7].w = alpha * res[31 ] + beta * C_res[7].w;
    
    *((float4 *)(C+ M * 0) + 0 ) = C_res[0 ];
    *((float4 *)(C+ M * 0) + 1 ) = C_res[1 ];
    *((float4 *)(C+ M * 1) + 0 ) = C_res[2 ];
    *((float4 *)(C+ M * 1) + 1 ) = C_res[3 ];
    *((float4 *)(C+ M * 2) + 0 ) = C_res[4 ];
    *((float4 *)(C+ M * 2) + 1 ) = C_res[5 ];
    *((float4 *)(C+ M * 3) + 0 ) = C_res[6 ];
    *((float4 *)(C+ M * 3) + 1 ) = C_res[7 ];
    
}
