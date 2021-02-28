#define KERNEL_RIGHT_BOTTOM_PADD 3
__global__ void make_res_1st(float * dev_src, float * dev_dst, int src_width, int src_height, int kernel_index, int tile_width, int tile_height);
__global__ void make_res_2st(float * dev_src, float * dev_dst, int src_width, int src_height, int kernel_index, int tile_width, int tile_height);
__global__ void make_res_3st(float * dev_src, float * dev_dst, int src_width, int src_height, int kernel_index, int tile_width, int tile_height);
__global__ void make_res_3x3(float * dev_src, float * dev_dst, int src_width, int src_height, int kernel_index, int tile_width, int tile_height);
__global__ void make_res_5x5(float * dev_src, float * dev_dst, int src_width, int src_height, int kernel_index, int tile_width, int tile_height);