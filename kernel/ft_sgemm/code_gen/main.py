from code_gen import ft_sgemm_code_gen
if __name__ =="__main__":
    import os
    kernel = os.sys.argv[1]  
    if_abft = int(os.sys.argv[2])
    abft = "ft_" if if_abft else ""
    function_name = f"{abft}sgemm_{kernel}"
    param = {
        "small" : [ 16,  16, 16,  8, 16, 2, 2],
        "medium": [ 32,  32,  8, 16, 32, 4, 4],
        "large" : [ 64,  64,  8, 32, 64, 8, 8],
        "tall"  : [128,  32,  8, 64, 16, 8, 4],
        "wide"  : [ 32, 128,  8, 16, 64, 4, 8],
        "huge"  : [128, 128,  8, 32, 64, 8, 8],
        "test"  : [ 64,  64,  8, 16, 32, 4, 4],
    }
    sgemm_kernel = ft_sgemm_code_gen(param[kernel], function_name, if_abft)
    with open(f"../include_code_gen/{function_name}.cuh", 'w') as f:
        f.write(sgemm_kernel)