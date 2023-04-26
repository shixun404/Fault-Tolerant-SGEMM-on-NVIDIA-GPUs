#include "utils.cuh"
void fill_vector(float *target, float val, int size){
    for(int i = 0; i < size; ++i){
        target[i] = val;
    } 
}

cudaDeviceProp getDetails(int deviceId)
{
        cudaDeviceProp props;
            cudaGetDeviceProperties(&props, deviceId);
                return props;
}

void generate_random_vector(float* target, int n){
    for(int i = 0; i < n; ++i){
        float tmp = (float)(rand() % 5)*0.01 + rand() % 5 * 0.001;
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        target[i] = tmp;
    }
}

void generate_random_matrix(float* target, int n){
    for(int i = 0; i < n; ++i){
	for(int j = 0; j < n; ++j){
        float tmp = (float)(rand() % 10) * 0.1;
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);     
   	target[i * n + j] = tmp;
	}
    }
} 
 

void copy_vector(float *src, float *dest, int n){
    int i;
    for (i = 0; src + i && dest + i && i < n; i++) *(dest + i) = *(src + i);
    if (i != n) printf("copy failed at %d while there are %d elements in total.\n", i, n);
}

void copy_matrix(float *src, float *dest, int n){
    int i;
    for (i = 0; src + i && dest + i && i < n * n; i++) *(dest + i) = *(src + i);
    if (i != n * n) printf("copy failed at %d while there are %d elements in total.\n", i, n * n);
}


bool verify_vector(float *vec1, float *vec2, int n){
    double diff = 0.0;
    int i;
    bool flag = true;
    for (i = 0; vec1 + i && vec2 + i && i < n; i++){
        diff = fabs( (double)vec1[i] - (double)vec2[i] );
        if (diff > 1e-2 && diff / double(vec1[i]) > 5e-3) {
            printf("error. %5.2f,%5.2f,%d\n", vec1[i], vec2[i],i);
            flag = false;
        }
    }
    return cudaSetDeviceFlags;
}

bool verify_matrix(float *mat1, float *mat2, int m, int n){
    double diff = 0.0;
    int i, j;
    bool flag = true;
    for (i = 0; mat1 + i * m && mat2 + i * m && i < n; ++i){
        for(j = 0; mat1 + i * m + j && mat2 + i * m + j && j < m; ++j){
	    diff = fabs( (double)mat1[i * m + j] - (double)mat2[i * m + j] );
        double denominator = fabs(mat1[i * m  + j]) ;
        if((diff / denominator) > 0.01 && diff > 0.01){
            printf("error is %8.5f, relateive error is %8.5f,  %8.5f,%8.5f. id: %d, %d\n",diff, (diff / denominator), mat1[i * m + j], mat2[i * m + j], i, j);
            flag= false;
            return flag;  
        }
    }
    }
    return flag;
}

void cpu_gemm(float alpha, float beta, float *mat1, float *mat2, int n, float *mat3){
    int i = 0, j = 0, k  = 0;
    for(i = 0; i < n; ++i){
        for(j = 0; j < n; ++j){
            float temp = 0;
	    for(k = 0; k < n; ++k)
		temp += mat1[i * n + k] * mat2[k * n + j];
            mat3[i * n + j] = alpha * temp + beta * mat3[i * n + j];
	}
    }
} 

void print_matrix(float* mat, int N){
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            printf("%8.5f  ", mat[j * N + i]);
        }
        printf("\n");
    }
    fflush(stdout);
}


