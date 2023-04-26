BINARY_NAME = ft_sgemm
CUDA_PATH   = /usr/local/cuda
CC          = $(CUDA_PATH)/bin/nvcc -arch=sm_75
CFLAGS      = -O3 -std=c++11 
LDFLAGS     = -L$(CUDA_PATH)/lib64 -lcudart -lcublas
INCFLAGS    = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -I. 




SRC         = $(wildcard *.cu)
build : $(BINARY_NAME)

$(BINARY_NAME): %: kernel/%/sgemm.cu  utils/utils.cu 
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS)  $^   -o $@ 

clean:
	rm $(BINARY_NAME)