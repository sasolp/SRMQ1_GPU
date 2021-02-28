
#define KERNELS_COUNT 30
#define COUNT_OF_SUBMODELS 39
#define Theta 8
#define HorThreads 32/*Warp Size*/
#define VerThreads Theta/*Theta*/
#define KERNEL_RIGHT_BOTTOM_PADD 3
#define Total 33
#define STREAM_COUNT 15/*Theta*/
#define BIN_COUNT 10
#define FIRST_BIN 4/*(BIN_COUNT - 2) /2*/
#define LAST_BIN (FIRST_BIN + 1)
#define MIN_MAX_SYM_COUNT 325	/*B^4 - 2*T*(T+1)*B^2 =====>>>>> B = T * 2 + 1*/
#define SPAM_SYM_COUNT 169	 /*B^2 + 4*T^2*(T+1)^2 =====>>>>> B = T * 2 + 1*/
#define T 2
#define ORDER 4
#define MAX_BLOCKS_HIST_COUNT 144

#define FULL_DIM  625	/* (int)std::pow((float)(2 * T + 1), ORDER);*/
enum RESIDUALS_1st{ L = 0, R = 1, U = 2, D = 3, LU = 4, RU = 5, RD = 6, LD = 7 };
enum RESIDUALS_2st{ dH = 8, dV = 9, dD = 10, dM = 11 };
enum RESIDUALS_3st{ L_ = 12, R_ = 13, U_ = 14, D_3st = 15, LU_ = 16, RU_ = 17, RD_ = 18, LD_ = 19 };
enum RESIDUALS_3x3{ Dl = 20, Du = 21, Dr = 22, Db = 23, D_4st = 24 };
enum RESIDUALS_5x5{ Dl_ = 25, Du_ = 26, Dr_ = 27, Db_ = 28, D_5st = 29 };
enum MINMAX{ MAX = 1, MIN = 0};

extern uint3 blocks ;
extern uint3 threads;
const int HIST_COUNT = 305/*minmax operation*/ + 33/*spam operation*/;

typedef struct PSRM_Features
{
	int index[HIST_COUNT];
	int submodel[HIST_COUNT];
	char name[HIST_COUNT][32];
	int* hists[HIST_COUNT];
	int sub_model_index[COUNT_OF_SUBMODELS];
	int sub_model_count[COUNT_OF_SUBMODELS];
	int last_index;
	int submodel_index;
}PSRM_Features;

#define COOC_HIST_MIN_MAX(res_count, trans, border)\
	cudaMemcpyAsync(residuals[residual_offset], &(residuals_indexes[0][0]) + residual_offset * 5, 5 * sizeof(int), cudaMemcpyHostToDevice, streams[host_features.last_index / (HIST_COUNT / STREAM_COUNT)]);\
	cooc_hist_min_max << <blocks, threads, 16 * 1024, streams[host_features.last_index / (HIST_COUNT / STREAM_COUNT)] >> >(dev_residuals, residuals[residual_offset++], res_count, dev_MINMAXsymmCoord, dev_SPAMsymmCoord, trans, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height, border);
#define COOC_HIST_SPAM(host_residuals_3x3, trans, hist_pos, border)\
 cooc_hist_spam << <blocks, threads, 16 * 1024, streams[host_features.last_index / (HIST_COUNT / STREAM_COUNT)] >> >(host_residuals_3x3, dev_MINMAXsymmCoord, dev_SPAMsymmCoord, trans, hist_pos, host_features.hists[host_features.last_index] , q, src_width, src_height, tile_width, tile_height, border);

void compute_submodels(PSRM_Features &host_features);


void make_models_1st(float **dev_residuals, int** residuals, float ** host_dev_residuals, int *dev_MINMAXsymmCoord, int *dev_SPAMsymmCoord, cudaStream_t streams[], PSRM_Features &host_features, int src_width, int src_height);
void make_models_2st(float **dev_residuals, int** residuals, float ** host_dev_residuals, int *dev_MINMAXsymmCoord, int *dev_SPAMsymmCoord, cudaStream_t streams[], PSRM_Features &host_features, int src_width, int src_height);
void make_models_3st(float **dev_residuals, int** residuals, float ** host_dev_residuals, int *dev_MINMAXsymmCoord, int *dev_SPAMsymmCoord, cudaStream_t streams[], PSRM_Features &host_features, int src_width, int src_height);
void make_models_3x3(float **dev_residuals, int** residuals, float ** host_dev_residuals, int *dev_MINMAXsymmCoord, int *dev_SPAMsymmCoord, cudaStream_t streams[], PSRM_Features &host_features, int src_width, int src_height);
void make_models_5x5(float **dev_residuals, int** residuals, float ** host_dev_residuals, int *dev_MINMAXsymmCoord, int *dev_SPAMsymmCoord, cudaStream_t streams[], PSRM_Features &host_features, int src_width, int src_height);


__global__ void cooc_hist_min_max(float** residuals, int*residuals_index, int residual_count, int *dev_MINMAXsymmCoord, int *dev_SPAMsymmCoord, bool trans, int* out_hist, int q, int src_width, int src_height, int tile_width, int tile_height, int border);
__global__ void cooc_hist_spam(float* first_residual, int *dev_MINMAXsymmCoord, int *dev_SPAMsymmCoord, bool trans, bool hist_pos, int* out_hist, int q, int src_width, int src_height, int tile_width, int tile_height, int border);




