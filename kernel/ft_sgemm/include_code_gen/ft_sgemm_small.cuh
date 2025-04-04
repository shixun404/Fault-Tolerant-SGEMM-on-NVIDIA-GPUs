 
#include <stdio.h>  
#define tab(t, a, b)t.x += a.x * b;t.y += a.y * b;  t.z += a.z * b;t.w += a.w * b;  

#define tcab(t, c, alpha, beta) \
    c.x = alpha * t.x + beta * c.x; \
    c.y = alpha * t.y + beta * c.y; \
    c.z = alpha * t.z + beta * c.z; \
    c.w = alpha * t.w + beta * c.w;
    
__global__  __launch_bounds__(64) void ft_sgemm_small(int M, int N, int K, float *A, float *B, float *C, float alpha, float beta){
    // ms = 128, ns = 32, ks = 8
    // mw = 64, nw = 16
    // mr = 8, nr = 4
    // blockId, warpId, and threadIdx
    
    int ms = 16, ns = 16, ks = 16, mw = 8, nw = 16, mr = 2, nr = 2;
    
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
    float4 prefetch_vector_tile_A[1];
    float4 prefetch_vector_tile_B[1];
    prefetch_vector_tile_A[0] = *((float4*)A + 0);
    prefetch_vector_tile_B[0] = *((float4*)B + 0);
    
    // offset to store the prefetch vector
    int offset_store_prefetch = ((k / ks) & 1);
    
    // get the pointer to prefetched buffer A and prefetched buffer B
    float* buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
    float* buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;

    // store the vectors in the prefetched buffer A and prefetched buffer B
    *(((float4*)buffer_A) + 1 * tx + 0) = prefetch_vector_tile_A[0];
    *(((float4*)buffer_B) + 1 * tx + 0) = prefetch_vector_tile_B[0];
    
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
    float2 vec_A[2];
    float2 vec_B[2];
    float2 tmp_row[1];
    float2 tmp_col[1];
    float res[4];
    float C_c[2];
    float C_r[2];
    
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
    vec_A[offset_prefetch_register_kk * 1 + 0] = *((float2*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + 0);
    vec_B[offset_prefetch_register_kk * 1 + 0] = *((float2*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk) + 0);
    
    // ABFT
    float4 block_level_A_c[1];
    float4 block_level_B_r[1];
    float A_c = 0., B_r = 0.;
    
    A_c += prefetch_vector_tile_A[0].x; A_c += prefetch_vector_tile_A[0].y; A_c += prefetch_vector_tile_A[0].z; A_c += prefetch_vector_tile_A[0].w; 
    B_r += prefetch_vector_tile_B[0].x; B_r += prefetch_vector_tile_B[0].y; B_r += prefetch_vector_tile_B[0].z; B_r += prefetch_vector_tile_B[0].w; 
    
    A_c += __shfl_xor_sync(0xffffffff, A_c, 1, 32);
    A_c += __shfl_xor_sync(0xffffffff, A_c, 2, 32);
    
    B_r += __shfl_xor_sync(0xffffffff, B_r, 1, 32);
    B_r += __shfl_xor_sync(0xffffffff, B_r, 2, 32);
    
        // saxpy
    block_level_A_c[0].x = prefetch_vector_tile_A[0].x * B_r; 
    block_level_A_c[0].y = prefetch_vector_tile_A[0].y * B_r; 
    block_level_A_c[0].z = prefetch_vector_tile_A[0].z * B_r; 
    block_level_A_c[0].w = prefetch_vector_tile_A[0].w * B_r; 
    
    block_level_B_r[0].x = prefetch_vector_tile_B[0].x * A_c; 
    block_level_B_r[0].y = prefetch_vector_tile_B[0].y * A_c; 
    block_level_B_r[0].z = prefetch_vector_tile_B[0].z * A_c; 
    block_level_B_r[0].w = prefetch_vector_tile_B[0].w * A_c; 
    
    // store into buffer

    // offset to store the saxpy result
    int offset_store_checksum = (((k / ks) + 1) & 1);
    
    // get the pointer to prefetched buffer A and prefetched buffer B
    float* checksum_buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_checksum * ms * ks;
    float* checksum_buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_checksum * ns * ks;
    
    *(((float4*)checksum_buffer_A) + tx * 1 + 0) = block_level_A_c[0];
    
    *(((float4*)checksum_buffer_B) + tx * 1 + 0) = block_level_B_r[0];
    
    __syncthreads(); 
    // offset C checksum each thread
    int offset_A_B = (tx < (1 * blockDim.x / 2)) ? (buffer_A_offset + offset_store_checksum * ms * ks): (buffer_B_offset + offset_store_checksum * ns * ks);
    int ws = (tx < (1 * blockDim.x / 2)) ? ms: ns;
    int ws_ = (tx < (1 * blockDim.x / 2)) ? ns: ms;
    int ws_1 = 1;
    int ws_2[1];
    offset_A_B +=  (tx & (int)(ws / ws_1 - 1)) * ws_1;
    float checksum[1];
    float checksum_[1];
    ws_2[0] = 0;
    checksum[0] = 0.;
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 0 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 1 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 2 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 3 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 4 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 5 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 6 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 7 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 8 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 9 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 10 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 11 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 12 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 13 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 14 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 15 + ws_2[0]));
    
    __syncthreads(); 
    // K loop
    for(k = 0; k < K; k += ks){
        // tile A abd tile B global offsets move forward ks columns
        A += ks * M; 
        B += ks * N; 
        // prefetch the vector from A and B in global memory 
        if(k + ks < K){
        prefetch_vector_tile_A[0] = *((float4*)A + 0);  
        prefetch_vector_tile_B[0] = *((float4*)B + 0);  
        
        }
        // inner k loop, 8
        for(kk = 0; kk < ks; ++kk){
            offset_register_kk = ((kk) & 1);
            offset_prefetch_register_kk = ((kk + 1) & 1);
    
            // offset of vec A and vec B w.r.t kk:
            offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
            offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
            
            // load the vectors from buffer to registers
            vec_A[offset_prefetch_register_kk * 1 + 0] = *((float2*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + 0);
            vec_B[offset_prefetch_register_kk * 1 + 0] = *((float2*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk) + 0);
            
            res[0 ] += vec_A[offset_register_kk * 1 + 0].x * vec_B[offset_register_kk * 1 + 0].x;
            res[1 ] += vec_A[offset_register_kk * 1 + 0].x * vec_B[offset_register_kk * 1 + 0].y;
            
            res[2 ] += vec_A[offset_register_kk * 1 + 0].y * vec_B[offset_register_kk * 1 + 0].x;
            res[3 ] += vec_A[offset_register_kk * 1 + 0].y * vec_B[offset_register_kk * 1 + 0].y;
            
            
        }
        if(((k+8) %(int(K / 20))) == 0){
            if(tx == (int)((k+8) / (int(K / 20)))){
            res[0] += error_inject;
            }
            C_r[0 ] = res[0 ]; C_r[0 ] += res[1 ]; 
            C_r[1 ] = res[2 ]; C_r[1 ] += res[3 ]; 
            
            C_c[0 ] = res[0 ]; C_c[0 ] += res[2 ]; 
            C_c[1 ] = res[1 ]; C_c[1 ] += res[3 ]; 
            
        __syncthreads();
        float* s = ((float*)(sAB) + ((idB_warp) * 8) + (idA_warp * 64) + idB_thread + (idA_thread * 16) + 0);
        float* s_ = ((float*)(sAB) + 128 + ((idA_warp) * 4) + (idB_warp * 128) + idA_thread + (idB_thread * 16) + 0);
        *(s_ + (0 * 8)) = C_c[0];
        *(s_ + (1 * 8)) = C_c[1];
        
        *(s + (0 * 8)) = C_r[0];
        *(s + (1 * 8)) = C_r[1];
        __syncthreads();
        checksum_[0] =  checksum[0];
        float4 r_;
        if (tx < int(1 * blockDim.x / 2)){
            r_ = *((float4*)((float*)sAB + ((tx& 15) * 1 + 0 ) * 8 + 0));
            checksum_[0] -= r_.x;
            checksum_[0] -= r_.y;
            checksum_[0] -= r_.z;
            checksum_[0] -= r_.w;
            r_ = *((float4*)((float*)sAB + ((tx& 15) * 1 + 0 ) * 8 + 4));
            checksum_[0] -= r_.x;
            checksum_[0] -= r_.y;
            checksum_[0] -= r_.z;
            checksum_[0] -= r_.w;
            
        }
        else{
            r_ = *((float4*)((float*)sAB + 128 + ((tx & 15) * 1 + 0) * 8 + 0));
            checksum_[0] -= r_.x;
            checksum_[0] -= r_.y;
            checksum_[0] -= r_.z;
            checksum_[0] -= r_.w;
            r_ = *((float4*)((float*)sAB + 128 + ((tx & 15) * 1 + 0) * 8 + 4));
            checksum_[0] -= r_.x;
            checksum_[0] -= r_.y;
            checksum_[0] -= r_.z;
            checksum_[0] -= r_.w;
            
        }
        __syncthreads();
        *((float*)sAB + (1 - int(tx / int(blockDim.x / 2))) * ns + (tx % (int(ws / 1))) * 1 + 0) = checksum_[0];
        __syncthreads();
        tmp_col[0] = (*((float2*)((float*)sAB + (idB_warp * 16 + idB_thread * 2) + 0)));
        tmp_row[0] = (*((float2*)((float*)sAB + ns + (idA_warp * 8 + idA_thread * 2) + 0)));
        res[0] += int( (fabsf(*((float*)tmp_row + 0)) > err_bound1) && (fabsf(*((float*)tmp_col + 0)) > err_bound1)) * (*((float*)tmp_row + 0));
        res[1] += int( (fabsf(*((float*)tmp_row + 0)) > err_bound1) && (fabsf(*((float*)tmp_col + 1)) > err_bound1)) * (*((float*)tmp_row + 0));
        res[2] += int( (fabsf(*((float*)tmp_row + 1)) > err_bound1) && (fabsf(*((float*)tmp_col + 0)) > err_bound1)) * (*((float*)tmp_row + 1));
        res[3] += int( (fabsf(*((float*)tmp_row + 1)) > err_bound1) && (fabsf(*((float*)tmp_col + 1)) > err_bound1)) * (*((float*)tmp_row + 1));
        __syncthreads();
        }
            
        // update offset to store the prefetch vector
        offset_store_prefetch = (((int)(k / ks) + 1) & 1);
        
        // update the pointer to prefetched buffer A and prefetched buffer B
        buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
        buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;
        // store the vectors in the prefetched buffer A and prefetched buffer B
        *(((float4*)buffer_A) + 1 * tx + 0) = prefetch_vector_tile_A[0];
        *(((float4*)buffer_B) + 1 * tx + 0) = prefetch_vector_tile_B[0];
        __syncthreads();
        // initial outer product column
        kk = -1;
        
        // offset of register store for prefetching
        offset_prefetch_register_kk = ((kk + 1) & 1);
        
        // offset of vec A and vec B w.r.t kk:
        offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
        offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
        
        // load the vectors from buffer to registers
        vec_A[offset_prefetch_register_kk * 1 + 0] = *((float2*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + 0);
        vec_B[offset_prefetch_register_kk * 1 + 0] = *((float2*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk) + 0);
        
        // ABFT
        A_c = 0., B_r = 0.;
        
        A_c += prefetch_vector_tile_A[0].x; A_c += prefetch_vector_tile_A[0].y; A_c += prefetch_vector_tile_A[0].z; A_c += prefetch_vector_tile_A[0].w; 
        B_r += prefetch_vector_tile_B[0].x; B_r += prefetch_vector_tile_B[0].y; B_r += prefetch_vector_tile_B[0].z; B_r += prefetch_vector_tile_B[0].w; 
        
        A_c += __shfl_xor_sync(0xffffffff, A_c, 1, 32);
        A_c += __shfl_xor_sync(0xffffffff, A_c, 2, 32);
        
        B_r += __shfl_xor_sync(0xffffffff, B_r, 1, 32);
        B_r += __shfl_xor_sync(0xffffffff, B_r, 2, 32);
        
        // saxpy
        block_level_A_c[0].x = prefetch_vector_tile_A[0].x * B_r; 
        block_level_A_c[0].y = prefetch_vector_tile_A[0].y * B_r; 
        block_level_A_c[0].z = prefetch_vector_tile_A[0].z * B_r; 
        block_level_A_c[0].w = prefetch_vector_tile_A[0].w * B_r; 
        
        block_level_B_r[0].x = prefetch_vector_tile_B[0].x * A_c; 
        block_level_B_r[0].y = prefetch_vector_tile_B[0].y * A_c; 
        block_level_B_r[0].z = prefetch_vector_tile_B[0].z * A_c; 
        block_level_B_r[0].w = prefetch_vector_tile_B[0].w * A_c; 
        
        // store into buffer

        // offset to store the saxpy result
        offset_store_checksum = (((k / ks)) & 1);
        
        // get the pointer to prefetched buffer A and prefetched buffer B
        float* checksum_buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_checksum * ms * ks;
        float* checksum_buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_checksum * ns * ks;
        
        *(((float4*)checksum_buffer_A) + tx * 1 + 0) = block_level_A_c[0];
        
        *(((float4*)checksum_buffer_B) + tx * 1 + 0) = block_level_B_r[0];
        
        __syncthreads(); 
        // offset C checksum each thread
        offset_A_B = (tx < (1 * blockDim.x / 2)) ? (buffer_A_offset + offset_store_checksum * ms * ks): (buffer_B_offset + offset_store_checksum * ns * ks);
        offset_A_B +=  (tx & (int)(ws / ws_1 - 1)) * ws_1;
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 0 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 1 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 2 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 3 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 4 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 5 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 6 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 7 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 8 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 9 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 10 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 11 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 12 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 13 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 14 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 15 + ws_2[0]));
        
    __syncthreads(); 
    }
    
    C += bx * ms + offset_vec_A_warp + offset_vec_A_thread;
    C += (by * ns + offset_vec_B_warp + offset_vec_B_thread) * M;
    
    float2 C_res[2];
    
    C_res[0 ] = *((float2 *)(C+ M * 0) + 0 );
    C_res[1 ] = *((float2 *)(C+ M * 1) + 0 );
    
    C_res[0].x = alpha * res[0  ] + beta * C_res[0].x;
    C_res[0].y = alpha * res[2  ] + beta * C_res[0].y;
    
    C_res[1].x = alpha * res[1  ] + beta * C_res[1].x;
    C_res[1].y = alpha * res[3  ] + beta * C_res[1].y;
    
    *((float2 *)(C+ M * 0) + 0 ) = C_res[0 ];
    *((float2 *)(C+ M * 1) + 0 ) = C_res[1 ];
    
}
