# Anatomy of High-Performance GEMM with Online Fault Tolerance on GPUs
## Install and compile the code by running the following command.
```
git clone  https://github.com/shixun404/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs.git
mkdir build; cmake ..; make -j
```
## Test the FT-SGEMM by running the following command.
./ft_sgemm `[START_MATRIX_SIZE]` `[END_MATRIX_SIZE]` `[MATRIX_SIZE_INTERVAL]` `[START_KERNEL]` `[END_KERNEL]`

Example Input:
```
./ft_sgemm 1024 6144 512 0 16
```
Example output:
```
Start verification!
kernel 0 finish verified!
kernel 1 finish verified!
kernel 2 finish verified!
kernel 3 finish verified!
kernel 4 finish verified!
kernel 5 finish verified!
kernel 6 finish verified!
kernel 7 finish verified!
kernel 8 finish verified!
kernel 9 finish verified!
kernel 10 finish verified!
kernel 11 finish verified!
kernel 12 finish verified!
kernel 13 finish verified!
kernel 14 finish verified!
kernel 15 finish verified!
kernel 16 finish verified!
################## Performance (GFLOPS) ########################
Matrix Size         |    1024|    1536|    2048|    2560|    3072|    3584|    4096|    4608|    5120|    5632|    6144|
cublas              |    4695|    5357|    4694|    4647|    4590|    4408|    4537|    4477|    4204|    4453|    4129|
kernel_sgemm_small  |    1565|    1094|    1082|     961|     981|     973|    1032|     967|     952|     944|     921|
kernel_sgemm_medium |    2941|    2799|    2301|    2170|    2041|    2084|    2175|    2065|    2061|    2047|    2033|
kernel_sgemm_large  |    4176|    4561|    3837|    3718|    3541|    3376|    3559|    3386|    3373|    3330|    3338|
kernel_sgemm_tall   |    2060|    2382|    2285|    2189|    2086|    2112|    2151|    2093|    2096|    2086|    2083|
kernel_sgemm_wide   |    3977|    3968|    3066|    3100|    3100|    3066|    3176|    3090|    3083|    3074|    3060|
kernel_sgemm_huge   |    4847|    5783|    5020|    4918|    4757|    4742|    4792|    4716|    4730|    4719|    4721|
abft_baseline       |    2058|    2853|    3074|    3248|    3225|    3108|    3170|    3170|    3099|    3176|    3532|
abft_kernel_small   |    1014|     751|     743|     710|     709|     715|     731|     709|     703|     702|     694|
abft_kernel_medium  |    2214|    1869|    1723|    1597|    1619|    1621|    1652|    1612|    1597|    1607|    1602|
abft_kernel_large   |    2839|    3350|    2940|    2762|    2827|    2772|    2835|    2768|    2737|    2772|    2766|
abft_kernel_tall    |    1754|    1801|    1817|    1692|    1705|    1701|    1723|    1691|    1662|    1695|    1693|
abft_kernel_wide    |    2763|    2410|    2295|    2150|    2292|    2301|    2333|    2284|    2230|    2297|    2291|
abft_kernel_huge    |    3811|    4448|    4076|    4024|    3986|    3924|    4005|    3952|    3885|    3955|    3945|
```

### Kernel Parameters:  -fault-tolerance off
|  | `kernel_id` |$m_{tb}$|$n_{tb}$|$k_{tb}$| $m_w$|$n_w$|$m_t$|$n_t$|
|--|-------------|--|--|--|--|--|--|--|
|cuBLAS*|0| - | - | - | - | - | - | - |
|small|1|16 | 16 | 16 | 8 | 16 | 2 | 2 |
|medium|2|32 | 32 | 8 | 16 | 32 | 4 | 4 |
|large|3|64 | 64 | 8 | 32 | 64 | 8 | 8 |
|tall|4|32 | 128 | 8 | 16 | 64 | 4 | 8 |
|huge|5|128 | 128 | 8 | 32 | 64 | 8 | 8 |

### Kernel Parameters:  -fault-tolerance on
|  | `kernel_id` |$m_{tb}$|$n_{tb}$|$k_{tb}$| $m_w$|$n_w$|$m_t$|$n_t$|
|--|-------------|--|--|--|--|--|--|--|
|non-fused ABFT SGEMM|0| - | - | - | - | - | - | - |
|fused ABFT SGEMM small|1|16 | 16 | 16 | 8 | 16 | 2 | 2 |
|fused ABFT SGEMM medium|2|32 | 32 | 8 | 16 | 32 | 4 | 4 |
|fused ABFT SGEMM large|3|64 | 64 | 8 | 32 | 64 | 8 | 8 |
|fused ABFT SGEMM tall|4|32 | 128 | 8 | 16 | 64 | 4 | 8 |
|fused ABFT SGEMM huge|5|128 | 128 | 8 | 32 | 64 | 8 | 8 |


## Generate fused ABFT SGEMM kernel by running the following commands 

```
cd kernel/ft_sgemm/code_gen
bash gen.sh
```

The generated code is located in `kernel/ft_sgemm/include_code_gen`.

# Citation:

To cite this repository:
```
@article{wu2023anatomy,
  title={Anatomy of High-Performance GEMM with Online Fault Tolerance on GPUs},
  author={Wu, Shixun and Zhai, Yujia and Liu, Jinyang and Huang, Jiajun and Jian, Zizhe and Wong, Bryan M and Chen, Zizhong},
  journal={arXiv preprint arXiv:2305.01024},
  year={2023}
}
```
