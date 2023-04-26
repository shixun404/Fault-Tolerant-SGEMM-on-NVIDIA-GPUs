void baseline_ft_sgemm(int num_tests, int M, int N, int K,  cublasHandle_t handle, float* dA, float* dB, float* dC, float* dE, float* dRes, float* dcheck_C_row, float* dcheck_C_col, float* dcheck_A_col_mul_B, float* dcheck_B_row_mul_A, float* dcheck_A_col, float* dcheck_B_row,  float alpha, float beta, float negative_1){
    
    for(int ii = 0; ii < num_tests; ++ii){
        for (int i = 0; i < K; i += 256){

        cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_T, M, N, 256, &alpha, dA + i * M, M, dB + i * N, N, &beta, dC, M);
        cudaDeviceSynchronize();
        // row sum of C
        cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, dC, M, dE, 1, &beta, dcheck_C_row, 1);
        
        // col sum of C
        cublasSgemv(handle, CUBLAS_OP_T, M, N, &alpha, dC, M, dE, 1, &beta, dcheck_C_col, 1);

        // col sum of A
        cublasSgemv(handle, CUBLAS_OP_T, M, 256, &alpha, dA  + i * M, M, dE, 1, &beta, dcheck_A_col, 1);

        // row sum of B
        cublasSgemv(handle, CUBLAS_OP_N, 256, N, &alpha, dB + i * N, N, dE, 1, &beta, dcheck_B_row, 1);
        cudaDeviceSynchronize();
        // col sum of A x B
        cublasSgemv(handle, CUBLAS_OP_T, 256, N, &alpha, dB + i * N, N, dcheck_A_col, 1, &beta, dcheck_A_col_mul_B, 1);

        // row sum of B x A
        cublasSgemv(handle, CUBLAS_OP_N, M, 256, &alpha, dA + i * M, M, dcheck_B_row, 1, &beta, dcheck_B_row_mul_A, 1);
        cudaDeviceSynchronize();
        // verify
        cublasSaxpy(handle, N, &negative_1, dcheck_A_col_mul_B, 1, dcheck_C_col, 1);
        cublasSdot(handle, N, dcheck_C_col, 1, dE, 1, dRes);
        cudaDeviceSynchronize();
        cublasSaxpy(handle, M, &negative_1, dcheck_B_row_mul_A, 1, dcheck_C_row, 1);
        cublasSdot(handle, M, dcheck_C_row, 1, dE, 1, dRes);
        }
    }
}