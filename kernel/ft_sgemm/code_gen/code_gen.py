
import math

def ft_sgemm_code_gen(param,function_name, if_abft=False):
    ms, ns, ks, mw, nw, mr, nr = param
    vector_variable = {'float4':['x','y','z','w'], 'float2':['x','y']}
    
    total_thread_num = (int)((ms / mr) * (ns / nr))
    
    global_read_byte_A = (int)(ms * ks / total_thread_num)
    global_read_byte_B = (int)(ns * ks / total_thread_num)
    
    global_read_vector_type_A = 'float4' if global_read_byte_A >= 4 else 'float2'
    global_read_vector_type_B = 'float4' if global_read_byte_B >= 4 else 'float2'
    
    global_read_vector_A_length = (int)(global_read_byte_A / len(vector_variable[global_read_vector_type_A]))
    global_read_vector_B_length = (int)(global_read_byte_B / len(vector_variable[global_read_vector_type_B]))
    
    shared_read_vector_type_A = 'float4' if mr >= 4 else 'float2'
    shared_read_vector_type_B = 'float4' if nr >= 4 else 'float2'
    
    shared_read_vector_A_length = (int)(mr / len(vector_variable[shared_read_vector_type_A]))
    shared_read_vector_B_length = (int)(nr / len(vector_variable[shared_read_vector_type_B]))
    
    checksum_byte_per_thread = (int)((ms + ns + total_thread_num - 1) / total_thread_num)
    
    global_read_vector_type_C = 'float4' if mr >= 4 else 'float2'
    global_read_byte_C = (mr * nr)
    global_read_vector_C_length = (int)((mr * nr) / len(vector_variable[global_read_vector_type_C]))
    global_read_vector_C_height = (int)(mr /len(vector_variable[global_read_vector_type_C]) )
    ft_sgemm = f''' 
#include <stdio.h>  
'''
    ft_sgemm += f'''#define tab(t, a, b)t.x += a.x * b;t.y += a.y * b;  t.z += a.z * b;t.w += a.w * b;  

#define tcab(t, c, alpha, beta) \\
    c.x = alpha * t.x + beta * c.x; \\
    c.y = alpha * t.y + beta * c.y; \\
    c.z = alpha * t.z + beta * c.z; \\
    c.w = alpha * t.w + beta * c.w;
    
__global__  __launch_bounds__({total_thread_num}) void '''+ f'''{function_name}'''+'''(int M, int N, int K, float *A, float *B, float *C, float alpha, float beta){
    // ms = 128, ns = 32, ks = 8
    // mw = 64, nw = 16
    // mr = 8, nr = 4
    // blockId, warpId, and threadIdx
    ''' + f'''
    int ms = {ms}, ns = {ns}, ks = {ks}, mw = {mw}, nw = {nw}, mr = {mr}, nr = {nr};
    '''+'''
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
    ''' + f'''
    __shared__ float sAB[{ms * ks * 2 + ns * ks * 2}]; 
    ''' + '''
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
    '''+ \
    f'''{global_read_vector_type_A} prefetch_vector_tile_A[{global_read_vector_A_length}];
    {global_read_vector_type_B} prefetch_vector_tile_B[{global_read_vector_B_length}];
    '''
        
    for i in range(global_read_vector_A_length):
        ft_sgemm +=f'''prefetch_vector_tile_A[{i}] = *(({global_read_vector_type_A}*)A + {i});
    ''' 
    for i in range(global_read_vector_B_length):
        ft_sgemm += f'''prefetch_vector_tile_B[{i}] = *(({global_read_vector_type_B}*)B + {i});
    '''
    ft_sgemm += '''
    // offset to store the prefetch vector
    int offset_store_prefetch = ((k / ks) & 1);
    
    // get the pointer to prefetched buffer A and prefetched buffer B
    float* buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
    float* buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;

    // store the vectors in the prefetched buffer A and prefetched buffer B
    ''' 
    for i in range(global_read_vector_A_length):
        ft_sgemm += f'''*((({global_read_vector_type_A}*)buffer_A) + {global_read_vector_A_length} * tx + {i}) = prefetch_vector_tile_A[{i}];
    '''
    for i in range(global_read_vector_B_length):
        ft_sgemm += f'''*((({global_read_vector_type_B}*)buffer_B) + {global_read_vector_B_length} * tx + {i}) = prefetch_vector_tile_B[{i}];
    '''
    
    ft_sgemm += '''
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
    '''
    ft_sgemm += f'''{shared_read_vector_type_A} vec_A[{shared_read_vector_A_length * 2}];
    {shared_read_vector_type_B} vec_B[{shared_read_vector_B_length * 2}];
    {shared_read_vector_type_A} tmp_row[{shared_read_vector_A_length}];
    {shared_read_vector_type_B} tmp_col[{shared_read_vector_B_length}];
    float res[{mr * nr}];
    float C_c[{nr}];
    float C_r[{mr}];
    '''
    ft_sgemm += '''
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
    '''
    for i in range(shared_read_vector_A_length):
        ft_sgemm += f'''vec_A[offset_prefetch_register_kk * {shared_read_vector_A_length} + {i}] = *(({shared_read_vector_type_A}*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + {i});
    '''
    for i in range(shared_read_vector_B_length):
        ft_sgemm += f'''vec_B[offset_prefetch_register_kk * {shared_read_vector_B_length} + {i}] = *(({shared_read_vector_type_B}*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk) + {i});
    '''
    if if_abft:
        ft_sgemm +=f'''
    // ABFT
    {global_read_vector_type_A} block_level_A_c[{global_read_vector_A_length}];
    {global_read_vector_type_B} block_level_B_r[{global_read_vector_B_length}];
    float A_c = 0., B_r = 0.;
    
    '''
    
        for i in range(global_read_vector_A_length):
            for j in vector_variable[global_read_vector_type_A]:
                ft_sgemm += f'''A_c += prefetch_vector_tile_A[{i}].{j}; '''
            ft_sgemm += '''
    '''
        for i in range(global_read_vector_B_length):
            for j in vector_variable[global_read_vector_type_B]:
                ft_sgemm += f'''B_r += prefetch_vector_tile_B[{i}].{j}; '''
            ft_sgemm += '''
    '''
        ft_sgemm +='''
    '''
        for i in range((int)(math.log2(ms / global_read_byte_A))):
            ft_sgemm += f'''A_c += __shfl_xor_sync(0xffffffff, A_c, {2 ** i}, 32);
    '''
        ft_sgemm +='''
    '''
        for i in range((int)(math.log2(ns / global_read_byte_B))):
            ft_sgemm += f'''B_r += __shfl_xor_sync(0xffffffff, B_r, {2 ** i}, 32);
    '''
        
        ft_sgemm +='''
        // saxpy
    '''
        for i in range(global_read_vector_A_length):
            for j in vector_variable[global_read_vector_type_A]:
                ft_sgemm += f'''block_level_A_c[{i}].{j} = prefetch_vector_tile_A[{i}].{j} * B_r; 
    '''
        ft_sgemm +='''
    '''
        for i in range(global_read_vector_B_length):
            for j in vector_variable[global_read_vector_type_B]:
                ft_sgemm += f'''block_level_B_r[{i}].{j} = prefetch_vector_tile_B[{i}].{j} * A_c; 
    '''
        ft_sgemm += '''
    // store into buffer

    // offset to store the saxpy result
    int offset_store_checksum = (((k / ks) + 1) & 1);
    
    // get the pointer to prefetched buffer A and prefetched buffer B
    float* checksum_buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_checksum * ms * ks;
    float* checksum_buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_checksum * ns * ks;
    
    '''
        for i in range(global_read_vector_A_length):
            ft_sgemm += f'''*((({global_read_vector_type_A}*)checksum_buffer_A) + tx * {global_read_vector_A_length} + {i}) = block_level_A_c[{i}];
    '''
        ft_sgemm +='''
    '''
        for i in range(global_read_vector_B_length):
            ft_sgemm += f'''*((({global_read_vector_type_B}*)checksum_buffer_B) + tx * {global_read_vector_B_length} + {i}) = block_level_B_r[{i}];
    '''

        ft_sgemm += f'''
    __syncthreads(); 
    // offset C checksum each thread
    int offset_A_B = (tx < (1 * blockDim.x / 2)) ? (buffer_A_offset + offset_store_checksum * ms * ks): (buffer_B_offset + offset_store_checksum * ns * ks);
    int ws = (tx < (1 * blockDim.x / 2)) ? ms: ns;
    int ws_ = (tx < (1 * blockDim.x / 2)) ? ns: ms;
    int ws_1 = {checksum_byte_per_thread};
    int ws_2[{checksum_byte_per_thread}];
    offset_A_B +=  (tx & (int)(ws / ws_1 - 1)) * ws_1;
    float checksum[{checksum_byte_per_thread}];
    float checksum_[{checksum_byte_per_thread}];
    '''
        for i in range(checksum_byte_per_thread):
            ft_sgemm += f'''ws_2[{i}] = {i};
    checksum[{i}] = 0.;
    '''
        for i in range(ks):
            for j in range(checksum_byte_per_thread):
                ft_sgemm += f'''checksum[{j}] +=  *(((float*)(sAB) + offset_A_B + ws * {i} + ws_2[{j}]));
    '''
    ft_sgemm +='''
    __syncthreads(); 
    // K loop
    for(k = 0; k < K; k += ks){
        // tile A abd tile B global offsets move forward ks columns
        A += ks * M; 
        B += ks * N; 
        // prefetch the vector from A and B in global memory 
        '''
    for i in range(global_read_vector_A_length):
        ft_sgemm += f'''prefetch_vector_tile_A[{i}] = *(({global_read_vector_type_A}*)A + {i});  
        '''
    for i in range(global_read_vector_B_length):
        ft_sgemm += f'''prefetch_vector_tile_B[{i}] = *(({global_read_vector_type_B}*)B + {i});  
        '''

    ft_sgemm += '''
        // inner k loop, 8
        for(kk = 0; kk < ks; ++kk){
            offset_register_kk = ((kk) & 1);
            offset_prefetch_register_kk = ((kk + 1) & 1);
    
            // offset of vec A and vec B w.r.t kk:
            offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
            offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
            
            // load the vectors from buffer to registers
            '''
    for i in range(shared_read_vector_A_length):
        ft_sgemm += f'''vec_A[offset_prefetch_register_kk * {shared_read_vector_A_length} + {i}] = *(({shared_read_vector_type_A}*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + {i});
            '''
    for i in range(shared_read_vector_B_length):
        ft_sgemm += f'''vec_B[offset_prefetch_register_kk * {shared_read_vector_B_length} + {i}] = *(({shared_read_vector_type_B}*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk) + {i});
            '''
    ft_sgemm += '''
            '''
    k = 0
    for i in range(shared_read_vector_A_length):
        for i_ in vector_variable[shared_read_vector_type_A]:
            for j in range(shared_read_vector_B_length):
                for j_ in vector_variable[shared_read_vector_type_B]:
                    ft_sgemm += f'''res[{k:<2}] += vec_A[offset_register_kk * {shared_read_vector_A_length} + {i}].{i_} * vec_B[offset_register_kk * {shared_read_vector_B_length} + {j}].{j_};
            '''
                    k += 1
                ft_sgemm += '''
            '''
    ft_sgemm += '''
        }
        '''
    if if_abft:       
        ft_sgemm +=  '''if(((k+8) %(int(K / 20))) == 0){
            if(tx == (int)((k+8) / (int(K / 20)))){
            res[0] += error_inject;
            }
            '''
        for j in range(mr):
            for i in range(nr):
                if_pm = '+' if i != 0 else ''
                ft_sgemm += f'''C_r[{j:<2}] {if_pm}= res[{(j * nr + i):<2}]; '''
            ft_sgemm += '''
            '''
        ft_sgemm += '''
            '''
        for i in range(nr):
            for j in range(mr):
                if_pm = '+' if j != 0 else ''
                ft_sgemm += f'''C_c[{i:<2}] {if_pm}= res[{(j * nr + i):<2}]; '''
            ft_sgemm += '''
            '''
        ft_sgemm += f'''
        __syncthreads();
        float* s = ((float*)(sAB) + ((idB_warp) * {nw//nr}) + (idA_warp * {ns * mw // nr}) + idB_thread + (idA_thread * {ns // nr * mr}) + 0);
        float* s_ = ((float*)(sAB) + {ms*ns//nr} + ((idA_warp) * {mw//mr}) + (idB_warp * {ms * nw // mr}) + idA_thread + (idB_thread * {ms // mr * nr}) + 0);
        '''
        
        for i in range(nr):
            ft_sgemm += f'''*(s_ + ({i} * {ms // mr})) = C_c[{i}];
        '''
        ft_sgemm += '''
        '''
        for i in range(mr):
            ft_sgemm += f'''*(s + ({i} * {ns // nr})) = C_r[{i}];
        '''
        
        ft_sgemm += '''__syncthreads();
        '''
        for j in range(checksum_byte_per_thread):
                ft_sgemm += f'''checksum_[{j}] =  checksum[{j}];
        '''
        ft_sgemm += '''float4 r_;
        if (tx < int(1 * blockDim.x / 2)){
            '''
        for j in range(checksum_byte_per_thread):
            for i in range(ns // nr // 4):
                ft_sgemm += f'''r_ = *((float4*)((float*)sAB + ((tx& {ms // checksum_byte_per_thread - 1}) * {checksum_byte_per_thread} + {j} ) * {ns // nr} + {i * 4}));
            '''
                for k in range(4):
                    ft_sgemm += f'''checksum_[{j}] -= r_.{vector_variable['float4'][k]};
            '''
        ft_sgemm += '''
        }
        '''
        
        ft_sgemm += '''else{
            '''
        for j in range(checksum_byte_per_thread):
            for i in range(ms // mr // 4):
                ft_sgemm += f'''r_ = *((float4*)((float*)sAB + {ms*ns//nr} + ((tx & {ns // checksum_byte_per_thread - 1}) * {checksum_byte_per_thread} + {j}) * {ms // mr} + {i * 4}));
            '''
                for k in range(4):
                    ft_sgemm += f'''checksum_[{j}] -= r_.{vector_variable['float4'][k]};
            '''
        ft_sgemm += '''
        }
        __syncthreads();
        '''
        for i in range(checksum_byte_per_thread):
            ft_sgemm += f'''*((float*)sAB + (1 - int(tx / int(blockDim.x / 2))) * ns + (tx % (int(ws / {(checksum_byte_per_thread)}))) * {(checksum_byte_per_thread)} + {i}) = checksum_[{i}];
        '''
        
        ft_sgemm += '''__syncthreads();
        '''
        
        '''
        {shared_read_vector_type_A} tmp_row[{shared_read_vector_A_length}];
        {shared_read_vector_type_B} tmp_col[{shared_read_vector_B_length}];
        '''
        for i in range(shared_read_vector_B_length):
            ft_sgemm += f'''tmp_col[{i}] = (*(({shared_read_vector_type_B}*)((float*)sAB + (idB_warp * {nw} + idB_thread * {nr}) + {i * len(vector_variable[shared_read_vector_type_B])})));
        '''
        #(*((float*)tmp_col + {j}))
        for i in range(shared_read_vector_A_length):
            ft_sgemm += f'''tmp_row[{i}] = (*(({shared_read_vector_type_A}*)((float*)sAB + ns + (idA_warp * {mw} + idA_thread * {mr}) + {i * len(vector_variable[shared_read_vector_type_A])})));
        '''
        for i in range(mr):
            for j in range(nr):
                if mr < nr:
                    ft_sgemm += f'''res[{i * nr + j}] += int( (fabsf(*((float*)tmp_row + {i})) > err_bound1) && (fabsf(*((float*)tmp_col + {j})) > err_bound1)) * (*((float*)tmp_col + {j}));
        '''
                else:
                    ft_sgemm += f'''res[{i * nr + j}] += int( (fabsf(*((float*)tmp_row + {i})) > err_bound1) && (fabsf(*((float*)tmp_col + {j})) > err_bound1)) * (*((float*)tmp_row + {i}));
        '''         
        # ft_sgemm += f'''res[{i * nr + j}] += int(( (*((float*)tmp_row + {i})) * (*((float*)tmp_col + {j}))) / (err_bound1 * err_bound1)) * (-error_inject);
        ''' 
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
        '''
        ft_sgemm += '''__syncthreads();
        }
        '''
    ft_sgemm += '''    
        // update offset to store the prefetch vector
        offset_store_prefetch = (((int)(k / ks) + 1) & 1);
        
        // update the pointer to prefetched buffer A and prefetched buffer B
        buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
        buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;
        // store the vectors in the prefetched buffer A and prefetched buffer B
        '''
            
    
    for i in range(global_read_vector_A_length):
        ft_sgemm += f'''*((({global_read_vector_type_A}*)buffer_A) + {global_read_vector_A_length} * tx + {i}) = prefetch_vector_tile_A[{i}];
        '''
    for i in range(global_read_vector_B_length):
        ft_sgemm += f'''*((({global_read_vector_type_B}*)buffer_B) + {global_read_vector_B_length} * tx + {i}) = prefetch_vector_tile_B[{i}];
        '''
    ft_sgemm += '''__syncthreads();
        // initial outer product column
        kk = -1;
        
        // offset of register store for prefetching
        offset_prefetch_register_kk = ((kk + 1) & 1);
        
        // offset of vec A and vec B w.r.t kk:
        offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
        offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
        
        // load the vectors from buffer to registers
        '''
    for i in range(shared_read_vector_A_length):
        ft_sgemm += f'''vec_A[offset_prefetch_register_kk * {shared_read_vector_A_length} + {i}] = *(({shared_read_vector_type_A}*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + {i});
        '''
    for i in range(shared_read_vector_B_length):
        ft_sgemm += f'''vec_B[offset_prefetch_register_kk * {shared_read_vector_B_length} + {i}] = *(({shared_read_vector_type_B}*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk) + {i});
        '''
    if if_abft:
        ft_sgemm +=f'''
        // ABFT
        A_c = 0., B_r = 0.;
        
        '''
        for i in range(global_read_vector_A_length):
            for j in vector_variable[global_read_vector_type_A]:
                ft_sgemm += f'''A_c += prefetch_vector_tile_A[{i}].{j}; '''
            ft_sgemm += '''
        '''
        for i in range(global_read_vector_B_length):
            for j in vector_variable[global_read_vector_type_B]:
                ft_sgemm += f'''B_r += prefetch_vector_tile_B[{i}].{j}; '''
            ft_sgemm += '''
        '''
        ft_sgemm +='''
        '''
        for i in range((int)(math.log2(ms / global_read_byte_A))):
            ft_sgemm += f'''A_c += __shfl_xor_sync(0xffffffff, A_c, {2 ** i}, 32);
        '''
        ft_sgemm +='''
        '''
        for i in range((int)(math.log2(ns / global_read_byte_B))):
            ft_sgemm += f'''B_r += __shfl_xor_sync(0xffffffff, B_r, {2 ** i}, 32);
        '''
        
        ft_sgemm +='''
        // saxpy
        '''
        for i in range(global_read_vector_A_length):
            for j in vector_variable[global_read_vector_type_A]:
                ft_sgemm += f'''block_level_A_c[{i}].{j} = prefetch_vector_tile_A[{i}].{j} * B_r; 
        '''
        ft_sgemm +='''
        '''
        for i in range(global_read_vector_B_length):
            for j in vector_variable[global_read_vector_type_B]:
                ft_sgemm += f'''block_level_B_r[{i}].{j} = prefetch_vector_tile_B[{i}].{j} * A_c; 
        '''
        ft_sgemm += '''
        // store into buffer

        // offset to store the saxpy result
        offset_store_checksum = (((k / ks)) & 1);
        
        // get the pointer to prefetched buffer A and prefetched buffer B
        float* checksum_buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_checksum * ms * ks;
        float* checksum_buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_checksum * ns * ks;
        
        '''
        for i in range(global_read_vector_A_length):
            ft_sgemm += f'''*((({global_read_vector_type_A}*)checksum_buffer_A) + tx * {global_read_vector_A_length} + {i}) = block_level_A_c[{i}];
        '''
        ft_sgemm +='''
        '''
        for i in range(global_read_vector_B_length):
            ft_sgemm += f'''*((({global_read_vector_type_B}*)checksum_buffer_B) + tx * {global_read_vector_B_length} + {i}) = block_level_B_r[{i}];
        '''

        ft_sgemm += f'''
        __syncthreads(); 
        // offset C checksum each thread
        offset_A_B = (tx < (1 * blockDim.x / 2)) ? (buffer_A_offset + offset_store_checksum * ms * ks): (buffer_B_offset + offset_store_checksum * ns * ks);
        offset_A_B +=  (tx & (int)(ws / ws_1 - 1)) * ws_1;
        '''
        for i in range(ks):
            for j in range(checksum_byte_per_thread):
                ft_sgemm += f'''checksum[{j}] +=  *(((float*)(sAB) + offset_A_B + ws * {i} + ws_2[{j}]));
        '''
    ft_sgemm += '''
    __syncthreads(); 
    }
    
    C += bx * ms + offset_vec_A_warp + offset_vec_A_thread;
    C += (by * ns + offset_vec_B_warp + offset_vec_B_thread) * M;
    '''
    ft_sgemm += '''
    '''
    ft_sgemm += f'''{global_read_vector_type_C} C_res[{global_read_vector_C_length}];
    
    '''
    for i in range(global_read_vector_C_length):
        ft_sgemm += f'''C_res[{i:<2}] = *(({global_read_vector_type_C} *)(C+ M * {(int)(i / global_read_vector_C_height)}) + {(i % global_read_vector_C_height)} );
    '''
    ft_sgemm += '''
    '''
    k = 0
    for i in range(global_read_vector_C_length):
        for j in vector_variable[global_read_vector_type_C]:
            ft_sgemm += f'''C_res[{i}].{j} = alpha * res[{(k % mr) * nr + (int)(k / mr):<2} ] + beta * C_res[{i}].{j};
    '''
            k+=1
        ft_sgemm += '''
    '''
    for i in range(global_read_vector_C_length):
        ft_sgemm += f'''*(({global_read_vector_type_C} *)(C+ M * {(int)(i / global_read_vector_C_height)}) + {(i % global_read_vector_C_height)} ) = C_res[{i:<2}];
    '''
    
    ft_sgemm +=   '''
}
'''
    return ft_sgemm
