#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "168ModelMaker.cuh"
#include <cmath>
#include <time.h>
#include <string>
#include <vector>
using namespace std;
	int residual_offset = 0;
	int residuals_indexes[HIST_COUNT][5] = {};
	extern uint3 blocks = { 2, 2, 1 };
	extern uint3  threads = { 16, 16, 1 };

__global__ void add_hist(int **hist1, int start, int count)
{
	int index = threadIdx.x;
	int sum = 0;
	int end = start + count;//printf("\nstart : %d end :%d\n", start, end);
	if (index < SPAM_SYM_COUNT * 2)
	{
		for (size_t i = start; i < end; i++)
		{
			sum += hist1[i][index];
		}
		//printf("\n%d\n", sum);
		hist1[start][index] = sum;
	}
	//__syncthreads();
}
void compute_submodels(PSRM_Features &host_features)
{
	int **hists;
	cudaError_t cudaStatus = cudaMalloc((void**)&hists, HIST_COUNT * sizeof(int*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "line 2374: cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(hists, host_features.hists, HIST_COUNT * sizeof(int*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "line 2378: cudaMemcpy failed!");
	}
	for (size_t i = 0; i <= host_features.submodel_index; i++)
	{
		add_hist << < 1, SPAM_SYM_COUNT * 2 >> >(hists, host_features.sub_model_index[i], host_features.sub_model_count[i]);
	}
}

void make_models_1st(float **dev_residuals, int** residuals, float ** host_residuals, int *dev_MINMAXsymmCoord, int *dev_SPAMsymmCoord, cudaStream_t streams[], PSRM_Features &host_features, int src_width, int src_height)
{
	//return;
	
	residual_offset = 0;
	memset( residuals_indexes, 0, HIST_COUNT * 5 * sizeof(int));
	//int tile_width = src_width / blocks.x / threads.x, tile_height = src_height / blocks.y / threads.y;
	int tile_width = 8, tile_height = 8;
	const int q = 1;
	host_features.last_index = -1;
	host_features.submodel_index = -1;
	
	
	/*float **dev_residuals;
	int* residuals[HIST_COUNT];
	for (int i = 0; i < HIST_COUNT; i++)
	{
		cudaStatus = cudaMalloc((void**)&residuals[i], 5 * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc for failed!");
		}

	}
	cudaStatus = cudaMalloc((void**)&dev_residuals, 30 * sizeof(float*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "line 57: cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(dev_residuals, host_residuals, 30 * sizeof(float*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "line 61: cudaMemcpy failed!");
	}*/
	
	/*//L = 0	U = 1	R =	2	D = 3	LU = 4	RU = 5	RD = 6	LD = 7
	[RLq_min, UDq_min, RLq_max, UDq_max] = deal(min(R, L), min(U, D), max(R, L), max(U, D));
	g.min22h = reshape(ProjHistMinMax(RLq_min, 'hor', q) + ProjHistMinMax(UDq_min, 'ver', q), [], 1);
	g.max22h = reshape(ProjHistMinMax(RLq_max, 'hor', q) + ProjHistMinMax(UDq_max, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/

	host_features.index[++host_features.last_index] = 1;host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 0; 
	strcpy(host_features.name[host_features.last_index], "s1_minmax22h"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::R; residuals_indexes[residual_offset][1] = RESIDUALS_1st::L;
	//strcpy(host_features.name[host_features.last_index], "s1_minmax22v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D;

	COOC_HIST_MIN_MAX(2, false, 2);//();
	
	
	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 0;
	strcpy(host_features.name[host_features.last_index], "s1_minmax22v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D;
	COOC_HIST_MIN_MAX(2, true, 2);
	

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;

	//return;
	
	host_features.index[++host_features.last_index] = 1;host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 1;
	strcpy(host_features.name[host_features.last_index], "s1_min34h"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U; residuals_indexes[residual_offset][2] = RESIDUALS_1st::R;
	COOC_HIST_MIN_MAX(3, false, 2);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 1;
	strcpy(host_features.name[host_features.last_index], "s1_min34h"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::R; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D; residuals_indexes[residual_offset][2] = RESIDUALS_1st::L;
	COOC_HIST_MIN_MAX(3, false, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 1;
	strcpy(host_features.name[host_features.last_index], "s1_min34h"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R; residuals_indexes[residual_offset][2] = RESIDUALS_1st::D;
	COOC_HIST_MIN_MAX(3, true, 2);


	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 1;
	strcpy(host_features.name[host_features.last_index], "s1_min34h"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::D; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U; residuals_indexes[residual_offset][2] = RESIDUALS_1st::L;
	COOC_HIST_MIN_MAX(3, true, 2);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;
	/*//L = 0 U = 1 R = 2 D = 3 LU = 4 RU = 5 RD = 6 LD = 7
	g.spam14h = reshape(ProjHistSpam(R, 'hor', q) + ProjHistSpam(U, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	g.spam14v = reshape(ProjHistSpam(R, 'ver', q) + ProjHistSpam(U, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	*/

	host_features.index[++host_features.last_index] = 1; host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 2;
	strcpy(host_features.name[host_features.last_index], "s1_spam14h");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_1st::R], false, false, 2); // , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 2;
	strcpy(host_features.name[host_features.last_index], "s1_spam14h");
	COOC_HIST_SPAM(  host_residuals[RESIDUALS_1st::U], true  , false, 2);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 1;
	host_features.submodel[host_features.last_index] = 2;
	strcpy(host_features.name[host_features.last_index], "s1_spam14v");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_1st::R], true, true, 2);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 2;
	strcpy(host_features.name[host_features.last_index], "s1_spam14v");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_1st::U], false, true, 2);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;


	/*//L = 0 U = 1 R = 2 D = 3 LU = 4 RU = 5 RD = 6 LD = 7
	g.min22v = reshape(ProjHistMinMax(RLq_min, 'ver', q) + ProjHistMinMax(UDq_min, 'hor', q), [], 1);
	g.max22v = reshape(ProjHistMinMax(RLq_max, 'ver', q) + ProjHistMinMax(UDq_max, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	*/
	
	host_features.index[++host_features.last_index] = 1;host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 3;
	strcpy(host_features.name[host_features.last_index], "s1_min22v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R;
	COOC_HIST_MIN_MAX(2, true, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 3;
	strcpy(host_features.name[host_features.last_index], "s1_min22v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D;
	COOC_HIST_MIN_MAX(2, false, 2);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;


	/*//L = 0 U = 1 R = 2 D = 3 LU = 4 RU = 5 RD = 6 LD = 7
	[RUq_min, RDq_min, LUq_min, LDq_min] = deal(min(R, U), min(R, D), min(L, U), min(L, D));
	[RUq_max, RDq_max, LUq_max, LDq_max] = deal(max(R, U), max(R, D), max(L, U), max(L, D));
	g.min24 = reshape(ProjHistMinMax(RUq_min, 'hor', q) + ProjHistMinMax(RDq_min, 'hor', q) + ProjHistMinMax(LUq_min, 'hor', q) + ProjHistMinMax(LDq_min, 'hor', q) + ...
	ProjHistMinMax(RUq_min, 'ver', q) + ProjHistMinMax(RDq_min, 'ver', q) + ProjHistMinMax(LUq_min, 'ver', q) + ProjHistMinMax(LDq_min, 'ver', q), [], 1);
	g.max24 = reshape(ProjHistMinMax(RUq_max, 'hor', q) + ProjHistMinMax(RDq_max, 'hor', q) + ProjHistMinMax(LUq_max, 'hor', q) + ProjHistMinMax(LDq_max, 'hor', q) + ...
	ProjHistMinMax(RUq_max, 'ver', q) + ProjHistMinMax(RDq_max, 'ver', q) + ProjHistMinMax(LUq_max, 'ver', q) + ProjHistMinMax(LDq_max, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/

	
	host_features.index[++host_features.last_index] = 1;host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 4;
	strcpy(host_features.name[host_features.last_index], "s1_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R;
	COOC_HIST_MIN_MAX(2, false, 2);

	host_features.index[++host_features.last_index] = 5;
	host_features.submodel[host_features.last_index] = 4;
	strcpy(host_features.name[host_features.last_index], "s1_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R;
	COOC_HIST_MIN_MAX(2, true, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 4;
	strcpy(host_features.name[host_features.last_index], "s1_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::R; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D;
	COOC_HIST_MIN_MAX(2, false, 2);

	host_features.index[++host_features.last_index] = 6;
	host_features.submodel[host_features.last_index] = 4;
	strcpy(host_features.name[host_features.last_index], "s1_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::R; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D;
	COOC_HIST_MIN_MAX(2, true, 2);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 4;
	strcpy(host_features.name[host_features.last_index], "s1_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U;
	COOC_HIST_MIN_MAX(2, false, 2);

	host_features.index[++host_features.last_index] = 7;
	host_features.submodel[host_features.last_index] = 4;
	strcpy(host_features.name[host_features.last_index], "s1_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U;
	COOC_HIST_MIN_MAX(2, true, 2);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 4;
	strcpy(host_features.name[host_features.last_index], "s1_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::D; residuals_indexes[residual_offset][1] = RESIDUALS_1st::L;
	COOC_HIST_MIN_MAX(2, false, 2);


	host_features.index[++host_features.last_index] = 8;
	host_features.submodel[host_features.last_index] = 4;
	strcpy(host_features.name[host_features.last_index], "s1_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D;
	COOC_HIST_MIN_MAX(2, true, 2);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;
	/*//L = 0 U = 1 R = 2 D = 3 LU = 4 RU = 5 RD = 6 LD = 7
	g.min34v = reshape(ProjHistMinMax(Uq_min, 'ver', q) + ProjHistMinMax(Dq_min, 'ver', q) + ProjHistMinMax(Rq_min, 'hor', q) + ProjHistMinMax(Lq_min, 'hor', q), [], 1);
	g.max34v = reshape(ProjHistMinMax(Uq_max, 'ver', q) + ProjHistMinMax(Dq_max, 'ver', q) + ProjHistMinMax(Rq_max, 'hor', q) + ProjHistMinMax(Lq_max, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	
	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 5;
	strcpy(host_features.name[host_features.last_index], "s1_min34v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U; residuals_indexes[residual_offset][2] = RESIDUALS_1st::R;
	COOC_HIST_MIN_MAX(3, true, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 5;
	strcpy(host_features.name[host_features.last_index], "s1_min34v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R; residuals_indexes[residual_offset][2] = RESIDUALS_1st::D;
	COOC_HIST_MIN_MAX(3, true, 2);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 5;
	strcpy(host_features.name[host_features.last_index], "s1_min34v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R; residuals_indexes[residual_offset][2] = RESIDUALS_1st::D;
	COOC_HIST_MIN_MAX(3, false, 2);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 5;
	strcpy(host_features.name[host_features.last_index], "s1_min34v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::D; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U; residuals_indexes[residual_offset][2] = RESIDUALS_1st::L;
	COOC_HIST_MIN_MAX(3, false, 2);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;
	/*L = 0 U = 1 R = 2 D = 3 LU = 4 RU = 5 RD = 6 LD = 7
	[R_min, R_max] = deal(min(RLq_min, UDq_min), max(RLq_max, UDq_max));
	g.min41 = reshape(ProjHistMinMax(R_min, 'hor', q) + ProjHistMinMax(R_min, 'ver', q), [], 1);
	g.max41 = reshape(ProjHistMinMax(R_max, 'hor', q) + ProjHistMinMax(R_max, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	//return;
	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 6;
	strcpy(host_features.name[host_features.last_index], "s1_min41"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U; residuals_indexes[residual_offset][2] = RESIDUALS_1st::R; residuals_indexes[residual_offset][3] = RESIDUALS_1st::D;
	COOC_HIST_MIN_MAX(4, false, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 6;
	strcpy(host_features.name[host_features.last_index], "s1_min41"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U; residuals_indexes[residual_offset][2] = RESIDUALS_1st::R; residuals_indexes[residual_offset][3] = RESIDUALS_1st::D;
	COOC_HIST_MIN_MAX(4, true, 2);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;
	//return;
	/*//L = 0 U = 1 R = 2 D = 3 LU = 4 RU = 5 RD = 6 LD = 7
	[RUq_min, RDq_min, LUq_min, LDq_min] = deal(min(RUq_min, RU), min(RDq_min, RD), min(LUq_min, LU), min(LDq_min, LD));
	[RUq_max, RDq_max, LUq_max, LDq_max] = deal(max(RUq_max, RU), max(RDq_max, RD), max(LUq_max, LU), max(LDq_max, LD));
	g.min34 = reshape(ProjHistMinMax(RUq_min, 'hor', q) + ProjHistMinMax(RDq_min, 'hor', q) + ProjHistMinMax(LUq_min, 'hor', q) + ProjHistMinMax(LDq_min, 'hor', q) + ...
	ProjHistMinMax(RUq_min, 'ver', q) + ProjHistMinMax(RDq_min, 'ver', q) + ProjHistMinMax(LUq_min, 'ver', q) + ProjHistMinMax(LDq_min, 'ver', q), [], 1);
	g.max34 = reshape(ProjHistMinMax(RUq_max, 'hor', q) + ProjHistMinMax(RDq_max, 'hor', q) + ProjHistMinMax(LUq_max, 'hor', q) + ProjHistMinMax(LDq_max, 'hor', q) + ...
	ProjHistMinMax(RUq_max, 'ver', q) + ProjHistMinMax(RDq_max, 'ver', q) + ProjHistMinMax(LUq_max, 'ver', q) + ProjHistMinMax(LDq_max, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 7;
	strcpy(host_features.name[host_features.last_index], "s1_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R; residuals_indexes[residual_offset][2] = RESIDUALS_1st::RU;
	COOC_HIST_MIN_MAX(3, false, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 7;
	strcpy(host_features.name[host_features.last_index], "s1_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::R; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D; residuals_indexes[residual_offset][2] = RESIDUALS_1st::RD;
	COOC_HIST_MIN_MAX(3, false, 2);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 7;
	strcpy(host_features.name[host_features.last_index], "s1_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U; residuals_indexes[residual_offset][2] = RESIDUALS_1st::LU;
	COOC_HIST_MIN_MAX(3, false, 2);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 7;
	strcpy(host_features.name[host_features.last_index], "s1_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D; residuals_indexes[residual_offset][2] = RESIDUALS_1st::LD;
	COOC_HIST_MIN_MAX(3, false, 2);

	host_features.index[++host_features.last_index] = 5;
	host_features.submodel[host_features.last_index] = 7;
	strcpy(host_features.name[host_features.last_index], "s1_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R; residuals_indexes[residual_offset][2] = RESIDUALS_1st::RU;
	COOC_HIST_MIN_MAX(3, true, 2);

	host_features.index[++host_features.last_index] = 6;
	host_features.submodel[host_features.last_index] = 7;
	strcpy(host_features.name[host_features.last_index], "s1_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::R; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D; residuals_indexes[residual_offset][2] = RESIDUALS_1st::RD;
	COOC_HIST_MIN_MAX(3, true, 2);

	host_features.index[++host_features.last_index] = 7;
	host_features.submodel[host_features.last_index] = 7;
	strcpy(host_features.name[host_features.last_index], "s1_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U; residuals_indexes[residual_offset][2] = RESIDUALS_1st::LU;
	COOC_HIST_MIN_MAX(3, true, 2);

	host_features.index[++host_features.last_index] = 8;
	host_features.submodel[host_features.last_index] = 7;
	strcpy(host_features.name[host_features.last_index], "s1_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D; residuals_indexes[residual_offset][2] = RESIDUALS_1st::LD;
	COOC_HIST_MIN_MAX(3, true, 2);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;
	//return;
	/*//L = 0 U = 1 R = 2 D = 3 LU = 4 RU = 5 RD = 6 LD = 7
	[RUq_min2, RDq_min2, LDq_min2, LUq_min2] = deal(min(RUq_min, LU), min(RDq_min, RU), min(LDq_min, RD), min(LUq_min, LD));
	[RUq_min3, RDq_min3, LDq_min3, LUq_min3] = deal(min(RUq_min, RD), min(RDq_min, LD), min(LDq_min, LU), min(LUq_min, RU));
	g.min48h = reshape(ProjHistMinMax(RUq_min2, 'hor', q) + ProjHistMinMax(LDq_min2, 'hor', q) + ProjHistMinMax(RDq_min3, 'hor', q) + ProjHistMinMax(LUq_min3, 'hor', q) + ...
	ProjHistMinMax(RDq_min2, 'ver', q) + ProjHistMinMax(LUq_min2, 'ver', q) + ProjHistMinMax(RUq_min3, 'ver', q) + ProjHistMinMax(LDq_min3, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	g.min48v = reshape(ProjHistMinMax(RDq_min2, 'hor', q) + ProjHistMinMax(LUq_min2, 'hor', q) + ProjHistMinMax(RUq_min3, 'hor', q) + ProjHistMinMax(LDq_min3, 'hor', q) + ...
	ProjHistMinMax(RUq_min2, 'ver', q) + ProjHistMinMax(LDq_min2, 'ver', q) + ProjHistMinMax(RDq_min3, 'ver', q) + ProjHistMinMax(LUq_min3, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex - 1;*/

	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::RU; residuals_indexes[residual_offset][3] = RESIDUALS_1st::LU;
	COOC_HIST_MIN_MAX(4, false, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::LD; residuals_indexes[residual_offset][3] = RESIDUALS_1st::RD;
	COOC_HIST_MIN_MAX(4, false, 2);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::R; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::RD; residuals_indexes[residual_offset][3] = RESIDUALS_1st::LD;
	COOC_HIST_MIN_MAX(4, false, 2);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::LU; residuals_indexes[residual_offset][3] = RESIDUALS_1st::RU;
	COOC_HIST_MIN_MAX(4, false, 2);

	host_features.index[++host_features.last_index] = 5;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::RD; residuals_indexes[residual_offset][3] = RESIDUALS_1st::RU;
	COOC_HIST_MIN_MAX(4, true, 2);

	host_features.index[++host_features.last_index] = 6;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::LU; residuals_indexes[residual_offset][3] = RESIDUALS_1st::LD;
	COOC_HIST_MIN_MAX(4, true, 2);

	host_features.index[++host_features.last_index] = 7;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::RU; residuals_indexes[residual_offset][3] = RESIDUALS_1st::RD;
	COOC_HIST_MIN_MAX(4, true, 2);

	host_features.index[++host_features.last_index] = 8;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::LD; residuals_indexes[residual_offset][3] = RESIDUALS_1st::LU;
	COOC_HIST_MIN_MAX(4, true, 2);

	
	//return;
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;

	/*//L = 0 U = 1 R = 2 D = 3 LU = 4 RU = 5 RD = 6 LD = 7
	[RUq_max2, RDq_max2, LDq_max2, LUq_max2] = deal(max(RUq_max, LU), max(RDq_max, RU), max(LDq_max, RD), max(LUq_max, LD));
	[RUq_max3, RDq_max3, LDq_max3, LUq_max3] = deal(max(RUq_max, RD), max(RDq_max, LD), max(LDq_max, LU), max(LUq_max, RU));
	g.max48h = reshape(ProjHistMinMax(RUq_max2, 'hor', q) + ProjHistMinMax(LDq_max2, 'hor', q) + ProjHistMinMax(RDq_max3, 'hor', q) + ProjHistMinMax(LUq_max3, 'hor', q) + ...
	ProjHistMinMax(RDq_max2, 'ver', q) + ProjHistMinMax(LUq_max2, 'ver', q) + ProjHistMinMax(RUq_max3, 'ver', q) + ProjHistMinMax(LDq_max3, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	g.max48v = reshape(ProjHistMinMax(RDq_max2, 'hor', q) + ProjHistMinMax(LUq_max2, 'hor', q) + ProjHistMinMax(RUq_max3, 'hor', q) + ProjHistMinMax(LDq_max3, 'hor', q) + ...
	ProjHistMinMax(RUq_max2, 'ver', q) + ProjHistMinMax(LDq_max2, 'ver', q) + ProjHistMinMax(RDq_max3, 'ver', q) + ProjHistMinMax(LUq_max3, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	*/

	host_features.index[++host_features.last_index] = 1; host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::D; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::RD; residuals_indexes[residual_offset][3] = RESIDUALS_1st::RU;
	COOC_HIST_MIN_MAX(4, false, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::LU; residuals_indexes[residual_offset][3] = RESIDUALS_1st::LD;
	COOC_HIST_MIN_MAX(4, false, 2);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::R; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::RU; residuals_indexes[residual_offset][3] = RESIDUALS_1st::RD;
	COOC_HIST_MIN_MAX(4, false, 2);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::LD; residuals_indexes[residual_offset][3] = RESIDUALS_1st::LU;
	COOC_HIST_MIN_MAX(4, false, 2);

	host_features.index[++host_features.last_index] = 5;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::RU; residuals_indexes[residual_offset][3] = RESIDUALS_1st::LU;
	COOC_HIST_MIN_MAX(4, true, 2);

	host_features.index[++host_features.last_index] = 6;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::LD; residuals_indexes[residual_offset][3] = RESIDUALS_1st::RD;
	COOC_HIST_MIN_MAX(4, true, 2);

	host_features.index[++host_features.last_index] = 7;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::D; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::RD; residuals_indexes[residual_offset][3] = RESIDUALS_1st::LD;
	COOC_HIST_MIN_MAX(4, true, 2);

	host_features.index[++host_features.last_index] = 8;
	host_features.submodel[host_features.last_index] = 8;
	strcpy(host_features.name[host_features.last_index], "s1_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::LU; residuals_indexes[residual_offset][3] = RESIDUALS_1st::RU;
	COOC_HIST_MIN_MAX(4, true, 2);
	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;
	/*	L = 0 U = 1 R = 2 D = 3 LU = 4 RU = 5 RD = 6 LD = 7
	%[RUq_max4,RDq_max4,LDq_max4,LUq_max4] = deal(max(R-U-RU-LU,RD),max(R-D-RD-RU,LD),max(L-D-LD-RD,LU),max(L-U-LU-LD,RU));
	%[RUq_max5,RDq_max5,LDq_max5,LUq_max5] = deal(max(R-U-RU-RD,LU),max(R-D-RD-LD,RU),max(L-D-LD-LU,RD),max(L-U-LU-RU,LD));
	g.min54 = reshape(ProjHistMinMax(RUq_min4, 'hor', q) + ProjHistMinMax(LDq_min4, 'hor', q) + ProjHistMinMax(RDq_min5, 'hor', q) + ProjHistMinMax(LUq_min5, 'hor', q) + ...
	ProjHistMinMax(RDq_min4, 'ver', q) + ProjHistMinMax(LUq_min4, 'ver', q) + ProjHistMinMax(RUq_min5, 'ver', q) + ProjHistMinMax(LDq_min5, 'ver', q), [], 1);
	[RUq_max4, RDq_max4, LDq_max4, LUq_max4] = deal(max(RUq_max2, RD), max(RDq_max2, LD), max(LDq_max2, LU), max(LUq_max2, RU));
	[RUq_max5, RDq_max5, LDq_max5, LUq_max5] = deal(max(RUq_max3, LU), max(RDq_max3, RU), max(LDq_max3, RD), max(LUq_max3, LD));
	g.max54 = reshape(ProjHistMinMax(RUq_max4, 'hor', q) + ProjHistMinMax(LDq_max4, 'hor', q) + ProjHistMinMax(RDq_max5, 'hor', q) + ProjHistMinMax(LUq_max5, 'hor', q) + ...
	ProjHistMinMax(RDq_max4, 'ver', q) + ProjHistMinMax(LUq_max4, 'ver', q) + ProjHistMinMax(RUq_max5, 'ver', q) + ProjHistMinMax(LDq_max5, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	*/
	//return;
	host_features.index[++host_features.last_index] = 1; host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 9;
	strcpy(host_features.name[host_features.last_index], "s1_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::RU; residuals_indexes[residual_offset][3] = RESIDUALS_1st::LU; residuals_indexes[residual_offset][4] = RESIDUALS_1st::RD;
	COOC_HIST_MIN_MAX(5, false, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 9;
	strcpy(host_features.name[host_features.last_index], "s1_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::LD; residuals_indexes[residual_offset][3] = RESIDUALS_1st::RD; residuals_indexes[residual_offset][4] = RESIDUALS_1st::LU;
	COOC_HIST_MIN_MAX(5, false, 2);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 9;
	strcpy(host_features.name[host_features.last_index], "s1_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::R; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::RD; residuals_indexes[residual_offset][3] = RESIDUALS_1st::RU; residuals_indexes[residual_offset][4] = RESIDUALS_1st::LD;
	COOC_HIST_MIN_MAX(5, false, 2);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 9;
	strcpy(host_features.name[host_features.last_index], "s1_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::LU; residuals_indexes[residual_offset][3] = RESIDUALS_1st::LD; residuals_indexes[residual_offset][4] = RESIDUALS_1st::RU;
	COOC_HIST_MIN_MAX(5, false, 2);

	host_features.index[++host_features.last_index] = 5;
	host_features.submodel[host_features.last_index] = 9;
	strcpy(host_features.name[host_features.last_index], "s1_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::D; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::RD; residuals_indexes[residual_offset][3] = RESIDUALS_1st::RU; residuals_indexes[residual_offset][4] = RESIDUALS_1st::LD;
	COOC_HIST_MIN_MAX(5, true, 2);

	host_features.index[++host_features.last_index] = 6;
	host_features.submodel[host_features.last_index] = 9;
	strcpy(host_features.name[host_features.last_index], "s1_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::U;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::LU; residuals_indexes[residual_offset][3] = RESIDUALS_1st::LD; residuals_indexes[residual_offset][4] = RESIDUALS_1st::RU;
	COOC_HIST_MIN_MAX(5, true, 2);

	host_features.index[++host_features.last_index] = 7;
	host_features.submodel[host_features.last_index] = 9;
	strcpy(host_features.name[host_features.last_index], "s1_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::U; residuals_indexes[residual_offset][1] = RESIDUALS_1st::R;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::RU; residuals_indexes[residual_offset][3] = RESIDUALS_1st::LU; residuals_indexes[residual_offset][4] = RESIDUALS_1st::RD;
	COOC_HIST_MIN_MAX(5, true, 2);

	host_features.index[++host_features.last_index] = 8;
	host_features.submodel[host_features.last_index] = 9;
	strcpy(host_features.name[host_features.last_index], "s1_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_1st::L; residuals_indexes[residual_offset][1] = RESIDUALS_1st::D;
	residuals_indexes[residual_offset][2] = RESIDUALS_1st::LD; residuals_indexes[residual_offset][3] = RESIDUALS_1st::RD; residuals_indexes[residual_offset][4] = RESIDUALS_1st::LU;
	COOC_HIST_MIN_MAX(5, true, 2);


	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;
	
	//float t1 = (float)clock();
	//for (int i = 0; i < HIST_COUNT; i++)
	//{
	//	cudaStatus = cudaFree(residuals[i]);
	//}
	//cudaStatus = cudaFree(dev_residuals);
	//float t2 = (clock() - t1) / CLOCKS_PER_SEC;
	//fprintf(stderr, "\nCPU Elapsed Time is %f Seconds\n", t2);
	
	return;

}
void make_models_2st(float **dev_residuals, int** residuals, float ** host_residuals, int *dev_MINMAXsymmCoord, int *dev_SPAMsymmCoord, cudaStream_t streams[], PSRM_Features &host_features, int src_width, int src_height)
{
	
	//return;
	int tile_width = 8, tile_height = 8;
	const int q = 2;
	

	


	/*H = 8		V = 9	D = 10	M = 11
	g.spam12h = reshape(ProjHistSpam(Dh, 'hor', q) + ProjHistSpam(Dv, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	g.spam12v = reshape(ProjHistSpam(Dh, 'ver', q) + ProjHistSpam(Dv, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	host_features.index[++host_features.last_index] = 1; host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 10;
	strcpy(host_features.name[host_features.last_index], "s2_spam12h");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_2st::dH], false, false, 2);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 10;
	strcpy(host_features.name[host_features.last_index], "s2_spam12h");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_2st::dV], true, false, 2);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 1;
	host_features.submodel[host_features.last_index] = 10;
	strcpy(host_features.name[host_features.last_index], "s2_spam12v");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_2st::dH], true, true, 2);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 10;
	strcpy(host_features.name[host_features.last_index], "s2_spam12v");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_2st::dV], false, true, 2);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;
	/*H = 8		V = 9	D = 10	M = 11
	[Dmin, Dmax] = deal(min(Dh, Dv), max(Dh, Dv));
	g.min21 = reshape(ProjHistMinMax(Dmin, 'hor', q) + ProjHistMinMax(Dmin, 'ver', q), [], 1);
	g.max21 = reshape(ProjHistMinMax(Dmax, 'hor', q) + ProjHistMinMax(Dmax, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 11;
	strcpy(host_features.name[host_features.last_index], "s2_min21"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dH; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dV;
	COOC_HIST_MIN_MAX(2, false, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 11;
	strcpy(host_features.name[host_features.last_index], "s2_min21"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dH; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dV;
	COOC_HIST_MIN_MAX(2, true, 2);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;



	/*H = 8		V = 9	D = 10	M = 11
	[Dmin2, Dmax2] = deal(min(min(Dh, Dv), min(Dd, Dm)), max(max(Dh, Dv), max(Dd, Dm)));
	g.min41 = reshape(ProjHistMinMax(Dmin2, 'hor', q) + ProjHistMinMax(Dmin2, 'ver', q), [], 1);
	g.max41 = reshape(ProjHistMinMax(Dmax2, 'hor', q) + ProjHistMinMax(Dmax2, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/

	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 12;
	strcpy(host_features.name[host_features.last_index], "s2_min41"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dH; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dV;
	residuals_indexes[residual_offset][2] = RESIDUALS_2st::dD; residuals_indexes[residual_offset][3] = RESIDUALS_2st::dM;
	COOC_HIST_MIN_MAX(4, false, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 12;
	strcpy(host_features.name[host_features.last_index], "s2_min41"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dH; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dV;
	residuals_indexes[residual_offset][2] = RESIDUALS_2st::dD; residuals_indexes[residual_offset][3] = RESIDUALS_2st::dM;
	COOC_HIST_MIN_MAX(4, true, 2);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;


	/*H = 8		V = 9	D = 10	M = 11
	[RUq_min, RDq_min] = deal(min(min(Dh, Dv), Dm), min(min(Dh, Dv), Dd));
	[RUq_max, RDq_max] = deal(max(max(max(Dh, Dv), Dm), max(max(max(Dh, Dv), Dd));
	g.min32 = reshape(ProjHistMinMax(RUq_min, 'hor', q) + ProjHistMinMax(RDq_min, 'hor', q) + ProjHistMinMax(RUq_min, 'ver', q) + ProjHistMinMax(RDq_min, 'ver', q), [], 1);
	g.max32 = reshape(ProjHistMinMax(RUq_max, 'hor', q) + ProjHistMinMax(RDq_max, 'hor', q) + ProjHistMinMax(RUq_max, 'ver', q) + ProjHistMinMax(RDq_max, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/

	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 13;
	strcpy(host_features.name[host_features.last_index], "s2_min32"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dH; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dV;
	residuals_indexes[residual_offset][2] = RESIDUALS_2st::dM;
	COOC_HIST_MIN_MAX(3, false, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 13;
	strcpy(host_features.name[host_features.last_index], "s2_min32"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dH; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dV;
	residuals_indexes[residual_offset][2] = RESIDUALS_2st::dD;
	COOC_HIST_MIN_MAX(3, false, 2);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 13;
	strcpy(host_features.name[host_features.last_index], "s2_min32"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dH; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dV;
	residuals_indexes[residual_offset][2] = RESIDUALS_2st::dM;
	COOC_HIST_MIN_MAX(3, true, 2);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 13;
	strcpy(host_features.name[host_features.last_index], "s2_min32"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dH; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dV;
	residuals_indexes[residual_offset][2] = RESIDUALS_2st::dD;
	COOC_HIST_MIN_MAX(3, true, 2);


	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;

	/*H = 8		V = 9	D = 10	M = 11
	[RUq_min2, RDq_min2, RUq_min3, LUq_min3] = deal(min(Dm, Dh), min(Dd, Dh), min(Dm, Dv), min(Dd, Dv));
	g.min24h = reshape(ProjHistMinMax(RUq_min2, 'hor', q) + ProjHistMinMax(RDq_min2, 'hor', q) + ProjHistMinMax(RUq_min3, 'ver', q) + ProjHistMinMax(LUq_min3, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	g.min24v = reshape(ProjHistMinMax(RUq_min2, 'ver', q) + ProjHistMinMax(RDq_min2, 'ver', q) + ProjHistMinMax(RUq_min3, 'hor', q) + ProjHistMinMax(LUq_min3, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex - 1;*/

	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 14;
	strcpy(host_features.name[host_features.last_index], "s2_min24h"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dM; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dH;
	COOC_HIST_MIN_MAX(2, false, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 14;
	strcpy(host_features.name[host_features.last_index], "s2_min24h"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dD; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dH;
	COOC_HIST_MIN_MAX(2, false, 2);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 14;
	strcpy(host_features.name[host_features.last_index], "s2_min24h"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dM; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dV;
	COOC_HIST_MIN_MAX(2, true, 2);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 14;
	strcpy(host_features.name[host_features.last_index], "s2_min24h"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dD; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dV;
	COOC_HIST_MIN_MAX(2, true, 2);
	/*H = 8		V = 9	D = 10	M = 11
	[RUq_max2, RDq_max2, RUq_max3, LUq_max3] = deal(max(Dm, Dh), max(Dd, Dh), max(Dm, Dv), max(Dd, Dv));
	g.max24h = reshape(ProjHistMinMax(RUq_max2, 'hor', q) + ProjHistMinMax(RDq_max2, 'hor', q) + ProjHistMinMax(RUq_max3, 'ver', q) + ProjHistMinMax(LUq_max3, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	g.max24v = reshape(ProjHistMinMax(RUq_max2, 'ver', q) + ProjHistMinMax(RDq_max2, 'ver', q) + ProjHistMinMax(RUq_max3, 'hor', q) + ProjHistMinMax(LUq_max3, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;

	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 14;
	strcpy(host_features.name[host_features.last_index], "s2_min24v"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dM; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dH;
	COOC_HIST_MIN_MAX(2, true, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 14;
	strcpy(host_features.name[host_features.last_index], "s2_min24v"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dD; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dH;
	COOC_HIST_MIN_MAX(2, true, 2);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 14;
	strcpy(host_features.name[host_features.last_index], "s2_min24v"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dM; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dV;
	COOC_HIST_MIN_MAX(2, false, 2);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 14;
	strcpy(host_features.name[host_features.last_index], "s2_min24v"); residuals_indexes[residual_offset][0] = RESIDUALS_2st::dD; residuals_indexes[residual_offset][1] = RESIDUALS_2st::dV;
	COOC_HIST_MIN_MAX(2, false, 2);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;


	//for (int i = 0; i < HIST_COUNT; i++)
	//{
	//	cudaStatus = cudaFree(residuals[i]);
	//}
	//cudaStatus = cudaFree(dev_residuals);
	return;
}
void make_models_3x3(float **dev_residuals, int** residuals, float ** host_residuals, int *dev_MINMAXsymmCoord, int *dev_SPAMsymmCoord, cudaStream_t streams[], PSRM_Features &host_features, int src_width, int src_height)
{
	//return;
	int tile_width = 8, tile_height = 8;
	
	
	/*float **dev_residuals;
	int* residuals[HIST_COUNT];
	
	for (int i = 0; i < HIST_COUNT; i++)
	{
		cudaStatus = cudaMalloc((void**)&residuals[i], 5 * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc for failed!");
		}
	}
	cudaStatus = cudaMalloc((void**)&dev_residuals, 30 * sizeof(float*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "line 789: cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(dev_residuals, host_residuals, 30 * sizeof(float*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "line 793: cudaMemcpy failed!");
	}*/

	const int q = 4;
	


	/*Dl = 20		Du = 21		Dr = 22		Db = 23		D = 24
	g.spam14v = reshape(ProjHistSpam(Du, 'ver', q) + ProjHistSpam(Db, 'ver', q) + ProjHistSpam(Dl, 'hor', q) + ProjHistSpam(Dr, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	g.spam14h = reshape(ProjHistSpam(Du, 'hor', q) + ProjHistSpam(Db, 'hor', q) + ProjHistSpam(Dl, 'ver', q) + ProjHistSpam(Dr, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	*/
	host_features.index[++host_features.last_index] = 1; host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 27;
	strcpy(host_features.name[host_features.last_index], "s3x3_spam14v");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_3x3::Du], true, true, 2);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 27;
	strcpy(host_features.name[host_features.last_index], "s3x3_spam14v");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_3x3::Db], true, true, 2);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);
	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 27;
	strcpy(host_features.name[host_features.last_index], "s3x3_spam14v");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_3x3::Dl], false, true, 2);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 27;
	strcpy(host_features.name[host_features.last_index], "s3x3_spam14v");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_3x3::Dr], false, true, 2);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 1;
	host_features.submodel[host_features.last_index] = 27;
	strcpy(host_features.name[host_features.last_index], "s3x3_spam14h");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_3x3::Du], false, false, 2);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 27;
	strcpy(host_features.name[host_features.last_index], "s3x3_spam14h");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_3x3::Db], false, false, 2);
	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 27;
	strcpy(host_features.name[host_features.last_index], "s3x3_spam14h");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_3x3::Dl], true,false,2);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);
	// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 27;
	strcpy(host_features.name[host_features.last_index], "s3x3_spam14h");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_3x3::Dr], true,false ,2);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;

	/*Dl = 20		Du = 21		Dr = 22		Db = 23		D = 24
	[Dmin1, Dmin2, Dmin3, Dmin4] = deal(min(Du, Dl), min(Db, Dr), min(Du, Dr), min(Db, Dl));
	g.min24 = reshape(ProjHistMinMax(Dmin1, 'ver', q) + ProjHistMinMax(Dmin2, 'ver', q) + ProjHistMinMax(Dmin3, 'ver', q) + ProjHistMinMax(Dmin4, 'ver', q) + ...
	ProjHistMinMax(Dmin1, 'hor', q) + ProjHistMinMax(Dmin2, 'hor', q) + ProjHistMinMax(Dmin3, 'hor', q) + ProjHistMinMax(Dmin4, 'hor', q), [], 1);
	[Dmax1, Dmax2, Dmax3, Dmax4] = deal(max(Du, Dl), max(Db, Dr), max(Du, Dr), max(Db, Dl));
	g.max24 = reshape(ProjHistMinMax(Dmax1, 'ver', q) + ProjHistMinMax(Dmax2, 'ver', q) + ProjHistMinMax(Dmax3, 'ver', q) + ProjHistMinMax(Dmax4, 'ver', q) + ...
	ProjHistMinMax(Dmax1, 'hor', q) + ProjHistMinMax(Dmax2, 'hor', q) + ProjHistMinMax(Dmax3, 'hor', q) + ProjHistMinMax(Dmax4, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	*/
	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 28;
	strcpy(host_features.name[host_features.last_index], "s3x3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3x3::Du; residuals_indexes[residual_offset][1] = RESIDUALS_3x3::Dl;
	COOC_HIST_MIN_MAX(2, false, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 28;
	strcpy(host_features.name[host_features.last_index], "s3x3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3x3::Db; residuals_indexes[residual_offset][1] = RESIDUALS_3x3::Dr;
	COOC_HIST_MIN_MAX(2, false, 2);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 28;
	strcpy(host_features.name[host_features.last_index], "s3x3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3x3::Du; residuals_indexes[residual_offset][1] = RESIDUALS_3x3::Dr;
	COOC_HIST_MIN_MAX(2, false, 2);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 28;
	strcpy(host_features.name[host_features.last_index], "s3x3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3x3::Db; residuals_indexes[residual_offset][1] = RESIDUALS_3x3::Dl;
	COOC_HIST_MIN_MAX(2, false, 2);

	host_features.index[++host_features.last_index] = 5;
	host_features.submodel[host_features.last_index] = 28;
	strcpy(host_features.name[host_features.last_index], "s3x3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3x3::Du; residuals_indexes[residual_offset][1] = RESIDUALS_3x3::Dl;
	COOC_HIST_MIN_MAX(2, true, 2);

	host_features.index[++host_features.last_index] = 6;
	host_features.submodel[host_features.last_index] = 28;
	strcpy(host_features.name[host_features.last_index], "s3x3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3x3::Db; residuals_indexes[residual_offset][1] = RESIDUALS_3x3::Dr;
	COOC_HIST_MIN_MAX(2, true, 2);

	host_features.index[++host_features.last_index] = 7;
	host_features.submodel[host_features.last_index] = 28;
	strcpy(host_features.name[host_features.last_index], "s3x3_min24");  residuals_indexes[residual_offset][0] = RESIDUALS_3x3::Du; residuals_indexes[residual_offset][1] = RESIDUALS_3x3::Dr;
	COOC_HIST_MIN_MAX(2, true, 2);

	host_features.index[++host_features.last_index] = 8;
	host_features.submodel[host_features.last_index] = 28;
	strcpy(host_features.name[host_features.last_index], "s3x3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3x3::Db; residuals_indexes[residual_offset][1] = RESIDUALS_3x3::Dl;
	COOC_HIST_MIN_MAX(2, true, 2);


	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;


	/*Dl = 20		Du = 21		Dr = 22		Db = 23		D = 24
	[UEq_min, REq_min] = deal(min(Du, Db), min(Dr, Dl));
	[UEq_max, REq_max] = deal(max(Du, Db), max(Dr, Dl));
	g.min22h = reshape(ProjHistMinMax(UEq_min, 'hor', q) + ProjHistMinMax(REq_min, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	g.max22h = reshape(ProjHistMinMax(UEq_max, 'hor', q) + ProjHistMinMax(REq_max, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	g.min22v = reshape(ProjHistMinMax(UEq_min, 'ver', q) + ProjHistMinMax(REq_min, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex - 1;
	g.max22v = reshape(ProjHistMinMax(UEq_max, 'ver', q) + ProjHistMinMax(REq_max, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 29;
	strcpy(host_features.name[host_features.last_index], "s3x3_min22v"); residuals_indexes[residual_offset][0] = RESIDUALS_3x3::Du; residuals_indexes[residual_offset][1] = RESIDUALS_3x3::Db;
	COOC_HIST_MIN_MAX(2, true, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 29;
	strcpy(host_features.name[host_features.last_index], "s3x3_min22v"); residuals_indexes[residual_offset][0] = RESIDUALS_3x3::Dr; residuals_indexes[residual_offset][1] = RESIDUALS_3x3::Dl;
	COOC_HIST_MIN_MAX(2, false, 2);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;



	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 30;
	strcpy(host_features.name[host_features.last_index], "s3x3_min22h"); residuals_indexes[residual_offset][0] = RESIDUALS_3x3::Du; residuals_indexes[residual_offset][1] = RESIDUALS_3x3::Db;
	COOC_HIST_MIN_MAX(2, false, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 30;
	strcpy(host_features.name[host_features.last_index], "s3x3_min22h"); residuals_indexes[residual_offset][0] = RESIDUALS_3x3::Dr; residuals_indexes[residual_offset][1] = RESIDUALS_3x3::Dl;
	COOC_HIST_MIN_MAX(2, true, 2);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;

	/*Dl = 20		Du = 21		Dr = 22		Db = 23		D = 24
	[Dmin5, Dmax5] = deal(min(Dmin1, Dmin2), max(Dmax1, Dmax2));
	g.min41 = reshape(ProjHistMinMax(Dmin5, 'ver', q) + ProjHistMinMax(Dmin5, 'hor', q), [], 1);
	g.max41 = reshape(ProjHistMinMax(Dmax5, 'ver', q) + ProjHistMinMax(Dmax5, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 31;
	strcpy(host_features.name[host_features.last_index], "s3x3_min41"); residuals_indexes[residual_offset][0] = RESIDUALS_3x3::Du; residuals_indexes[residual_offset][1] = RESIDUALS_3x3::Db;
	residuals_indexes[residual_offset][2] = RESIDUALS_3x3::Dr; residuals_indexes[residual_offset][3] = RESIDUALS_3x3::Dl;
	COOC_HIST_MIN_MAX(4, true, 2);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 31;
	strcpy(host_features.name[host_features.last_index], "s3x3_min41"); residuals_indexes[residual_offset][0] = RESIDUALS_3x3::Du; residuals_indexes[residual_offset][1] = RESIDUALS_3x3::Db;
	residuals_indexes[residual_offset][2] = RESIDUALS_3x3::Dr; residuals_indexes[residual_offset][3] = RESIDUALS_3x3::Dl;
	COOC_HIST_MIN_MAX(4, false, 2);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;
	

	return;
}
void make_models_3st(float **dev_residuals, int** residuals, float ** host_residuals, int *dev_MINMAXsymmCoord, int *dev_SPAMsymmCoord, cudaStream_t streams[], PSRM_Features &host_features, int src_width, int src_height)
{
	//return;
	int tile_width = 8, tile_height = 8;
	/*host_features.last_index = -1;
	host_features.submodel_index = -1;*/
	
	/*float **dev_residuals;
	int* residuals[HIST_COUNT];
	
	for (int i = 0; i < HIST_COUNT; i++)
	{
		cudaStatus = cudaMalloc((void**)&residuals[i], 5 * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc for failed!");
		}
	}
	cudaStatus = cudaMalloc((void**)&dev_residuals, 30 * sizeof(float*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "line 789: cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(dev_residuals, host_residuals, 30 * sizeof(float*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "line 793: cudaMemcpy failed!");
	}*/

	const int q = 3;
	/*L = 12	U = 13	R =	14	D = 15	LU = 16	RU = 17	RD = 18	LD = 19
	[RLq_min, UDq_min, RLq_max, UDq_max] = deal(min(R, L), min(U, D), max(R, L), max(U, D));
	g.min22h = reshape(ProjHistMinMax(RLq_min, 'hor', q) + ProjHistMinMax(UDq_min, 'ver', q), [], 1);
	g.max22h = reshape(ProjHistMinMax(RLq_max, 'hor', q) + ProjHistMinMax(UDq_max, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	/**/host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 16;
	strcpy(host_features.name[host_features.last_index], "s3_min22h"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::L_;
	COOC_HIST_MIN_MAX(2, false, 4);
	
	
	host_features.index[++host_features.last_index] = 2;
	//host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 16;
	strcpy(host_features.name[host_features.last_index], "s3_min22h"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st;
	COOC_HIST_MIN_MAX(2, true, 4);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;
//return;

	/*L = 12 U = 13 R = 14 D = 15 LU = 16 RU = 17 RD = 18 LD = 19
	[Uq_min, Rq_min, Dq_min, Lq_min] = deal(min(RLq_min, U), min(UDq_min, R), min(RLq_min, D), min(UDq_min, L));
	[Uq_max, Rq_max, Dq_max, Lq_max] = deal(max(RLq_max, U), max(UDq_max, R), max(RLq_max, D), max(UDq_max, L));
	g.min34h = reshape(ProjHistMinMax(Uq_min, 'hor', q) + ProjHistMinMax(Dq_min, 'hor', q) + ProjHistMinMax(Lq_min, 'ver', q) + ProjHistMinMax(Rq_min, 'ver', q), [], 1);
	g.max34h = reshape(ProjHistMinMax(Uq_max, 'hor', q) + ProjHistMinMax(Dq_max, 'hor', q) + ProjHistMinMax(Lq_max, 'ver', q) + ProjHistMinMax(Rq_max, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 17;
	strcpy(host_features.name[host_features.last_index], "s3_min34h"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][2] = RESIDUALS_3st::R_;
	COOC_HIST_MIN_MAX(3, false, 4);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 17;
	strcpy(host_features.name[host_features.last_index], "s3_min34h"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][2] = RESIDUALS_3st::D_3st;
	COOC_HIST_MIN_MAX(3, true, 4);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 17;
	strcpy(host_features.name[host_features.last_index], "s3_min34h"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st; residuals_indexes[residual_offset][2] = RESIDUALS_3st::L_;
	COOC_HIST_MIN_MAX(3, false, 4);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 17;
	strcpy(host_features.name[host_features.last_index], "s3_min34h"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::D_3st; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][2] = RESIDUALS_3st::L_;
	COOC_HIST_MIN_MAX(3, true, 4);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;
	/*//L = 0 U = 1 R = 2 D = 3 LU = 4 RU = 5 RD = 6 LD = 7
	g.spam14h = reshape(ProjHistSpam(R, 'hor', q) + ProjHistSpam(U, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	g.spam14v = reshape(ProjHistSpam(R, 'ver', q) + ProjHistSpam(U, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	*/
	host_features.index[++host_features.last_index] = 1; host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 18;
	strcpy(host_features.name[host_features.last_index], "s3_spam14h");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_3st::R_], false, false, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 18;
	strcpy(host_features.name[host_features.last_index], "s3_spam14h");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_3st::U_], true, false, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 1;
	host_features.submodel[host_features.last_index] = 18;
	strcpy(host_features.name[host_features.last_index], "s3_spam14v");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_3st::R_], true, true, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 18;
	strcpy(host_features.name[host_features.last_index], "s3_spam14v");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_3st::U_], false, true, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;
	/*L = 12 U = 13 R = 14 D = 15 LU = 16 RU = 17 RD = 18 LD = 19
	g.min22v = reshape(ProjHistMinMax(RLq_min, 'ver', q) + ProjHistMinMax(UDq_min, 'hor', q), [], 1);
	g.max22v = reshape(ProjHistMinMax(RLq_max, 'ver', q) + ProjHistMinMax(UDq_max, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	*/

	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 19;
	strcpy(host_features.name[host_features.last_index], "s3_min22v"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_;
	COOC_HIST_MIN_MAX(2, true, 4);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 19;
	strcpy(host_features.name[host_features.last_index], "s3_min22v"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st;
	COOC_HIST_MIN_MAX(2, false, 4);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;

	/*L = 12 U = 13 R = 14 D = 15 LU = 16 RU = 17 RD = 18 LD = 19
	[RUq_min, RDq_min, LUq_min, LDq_min] = deal(min(R, U), min(R, D), min(L, U), min(L, D));
	[RUq_max, RDq_max, LUq_max, LDq_max] = deal(max(R, U), max(R, D), max(L, U), max(L, D));
	g.min24 = reshape(ProjHistMinMax(RUq_min, 'hor', q) + ProjHistMinMax(RDq_min, 'hor', q) + ProjHistMinMax(LUq_min, 'hor', q) + ProjHistMinMax(LDq_min, 'hor', q) + ...
	ProjHistMinMax(RUq_min, 'ver', q) + ProjHistMinMax(RDq_min, 'ver', q) + ProjHistMinMax(LUq_min, 'ver', q) + ProjHistMinMax(LDq_min, 'ver', q), [], 1);
	g.max24 = reshape(ProjHistMinMax(RUq_max, 'hor', q) + ProjHistMinMax(RDq_max, 'hor', q) + ProjHistMinMax(LUq_max, 'hor', q) + ProjHistMinMax(LDq_max, 'hor', q) + ...
	ProjHistMinMax(RUq_max, 'ver', q) + ProjHistMinMax(RDq_max, 'ver', q) + ProjHistMinMax(LUq_max, 'ver', q) + ProjHistMinMax(LDq_max, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/

	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 20;
	strcpy(host_features.name[host_features.last_index], "s3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_;
	COOC_HIST_MIN_MAX(2, false, 4);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 20;
	strcpy(host_features.name[host_features.last_index], "s3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st;
	COOC_HIST_MIN_MAX(2, false, 4);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 20;
	strcpy(host_features.name[host_features.last_index], "s3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_;
	COOC_HIST_MIN_MAX(2, false, 4);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 20;
	strcpy(host_features.name[host_features.last_index], "s3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::D_3st; residuals_indexes[residual_offset][1] = RESIDUALS_3st::L_;
	COOC_HIST_MIN_MAX(2, false, 4);

	host_features.index[++host_features.last_index] = 5;
	host_features.submodel[host_features.last_index] = 20;
	strcpy(host_features.name[host_features.last_index], "s3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_;
	COOC_HIST_MIN_MAX(2, true, 4);

	host_features.index[++host_features.last_index] = 6;
	host_features.submodel[host_features.last_index] = 20;
	strcpy(host_features.name[host_features.last_index], "s3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st;
	COOC_HIST_MIN_MAX(2, true, 4);

	host_features.index[++host_features.last_index] = 7;
	host_features.submodel[host_features.last_index] = 20;
	strcpy(host_features.name[host_features.last_index], "s3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_;
	COOC_HIST_MIN_MAX(2, true, 4);

	host_features.index[++host_features.last_index] = 8;
	host_features.submodel[host_features.last_index] = 20;
	strcpy(host_features.name[host_features.last_index], "s3_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st;
	COOC_HIST_MIN_MAX(2, true, 4);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;

	/*L = 12 U = 13 R = 14 D = 15 LU = 16 RU = 17 RD = 18 LD = 19
	g.min34v = reshape(ProjHistMinMax(Uq_min, 'ver', q) + ProjHistMinMax(Dq_min, 'ver', q) + ProjHistMinMax(Rq_min, 'hor', q) + ProjHistMinMax(Lq_min, 'hor', q), [], 1);
	g.max34v = reshape(ProjHistMinMax(Uq_max, 'ver', q) + ProjHistMinMax(Dq_max, 'ver', q) + ProjHistMinMax(Rq_max, 'hor', q) + ProjHistMinMax(Lq_max, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 21;
	strcpy(host_features.name[host_features.last_index], "s3_min34v"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][2] = RESIDUALS_3st::R_;
	COOC_HIST_MIN_MAX(3, true, 4);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 21;
	strcpy(host_features.name[host_features.last_index], "s3_min34v"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][2] = RESIDUALS_3st::D_3st;
	COOC_HIST_MIN_MAX(3, true, 4);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 21;
	strcpy(host_features.name[host_features.last_index], "s3_min34v"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][2] = RESIDUALS_3st::D_3st;
	COOC_HIST_MIN_MAX(3, false, 4);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 21;
	strcpy(host_features.name[host_features.last_index], "s3_min34v"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::D_3st; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][2] = RESIDUALS_3st::L_;
	COOC_HIST_MIN_MAX(3, false, 4);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;

	/*L = 12 U = 13 R = 14 D = 15 LU = 16 RU = 17 RD = 18 LD = 19
	[R_min, R_max] = deal(min(RUq_min, LDq_min), max(RUq_max, LDq_max));
	g.min41 = reshape(ProjHistMinMax(R_min, 'hor', q) + ProjHistMinMax(R_min, 'ver', q), [], 1);
	g.max41 = reshape(ProjHistMinMax(R_max, 'hor', q) + ProjHistMinMax(R_max, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 22;
	strcpy(host_features.name[host_features.last_index], "s3_min41"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][2] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::D_3st;
	COOC_HIST_MIN_MAX(4, false, 4);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 22;
	strcpy(host_features.name[host_features.last_index], "s3_min41"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][2] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::D_3st;
	COOC_HIST_MIN_MAX(4, true, 4);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;

	/*L = 12 U = 13 R = 14 D = 15 LU = 16 RU = 17 RD = 18 LD = 19
	[RUq_min2, RDq_min2, LUq_min2, LDq_min2] = deal(min(RUq_min, RU), min(RDq_min, RD), min(LUq_min, LU), min(LDq_min, LD));
	[RUq_max2, RDq_max2, LUq_max2, LDq_max2] = deal(max(RUq_max, RU), max(RDq_max, RD), max(LUq_max, LU), max(LDq_max, LD));
	g.min34 = reshape(ProjHistMinMax(RUq_min2, 'hor', q) + ProjHistMinMax(RDq_min2, 'hor', q) + ProjHistMinMax(LUq_min2, 'hor', q) + ProjHistMinMax(LDq_min2, 'hor', q) + ...
	ProjHistMinMax(RUq_min2, 'ver', q) + ProjHistMinMax(RDq_min2, 'ver', q) + ProjHistMinMax(LUq_min2, 'ver', q) + ProjHistMinMax(LDq_min2, 'ver', q), [], 1);
	g.max34 = reshape(ProjHistMinMax(RUq_max2, 'hor', q) + ProjHistMinMax(RDq_max2, 'hor', q) + ProjHistMinMax(LUq_max2, 'hor', q) + ProjHistMinMax(LDq_max2, 'hor', q) + ...
	ProjHistMinMax(RUq_max2, 'ver', q) + ProjHistMinMax(RDq_max2, 'ver', q) + ProjHistMinMax(LUq_max2, 'ver', q) + ProjHistMinMax(LDq_max2, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/

	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 23;
	strcpy(host_features.name[host_features.last_index], "s3_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][2] = RESIDUALS_3st::RU_;
	COOC_HIST_MIN_MAX(3, false, 4);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 23;
	strcpy(host_features.name[host_features.last_index], "s3_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st; residuals_indexes[residual_offset][2] = RESIDUALS_3st::RD_;
	COOC_HIST_MIN_MAX(3, false, 4);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 23;
	strcpy(host_features.name[host_features.last_index], "s3_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][2] = RESIDUALS_3st::LU_;
	COOC_HIST_MIN_MAX(3, false, 4);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 23;
	strcpy(host_features.name[host_features.last_index], "s3_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st; residuals_indexes[residual_offset][2] = RESIDUALS_3st::LD_;
	COOC_HIST_MIN_MAX(3, false, 4);

	host_features.index[++host_features.last_index] = 5;
	host_features.submodel[host_features.last_index] = 23;
	strcpy(host_features.name[host_features.last_index], "s3_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][2] = RESIDUALS_3st::RU_;
	COOC_HIST_MIN_MAX(3, true, 4);

	host_features.index[++host_features.last_index] = 6;
	host_features.submodel[host_features.last_index] = 23;
	strcpy(host_features.name[host_features.last_index], "s3_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st; residuals_indexes[residual_offset][2] = RESIDUALS_3st::RD_;
	COOC_HIST_MIN_MAX(3, true, 4);

	host_features.index[++host_features.last_index] = 7;
	host_features.submodel[host_features.last_index] = 23;
	strcpy(host_features.name[host_features.last_index], "s3_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][2] = RESIDUALS_3st::LU_;
	COOC_HIST_MIN_MAX(3, true, 4);

	host_features.index[++host_features.last_index] = 8;
	host_features.submodel[host_features.last_index] = 23;
	strcpy(host_features.name[host_features.last_index], "s3_min34"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st; residuals_indexes[residual_offset][2] = RESIDUALS_3st::LD_;
	COOC_HIST_MIN_MAX(3, true, 4);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;


	/*L = 12 U = 13 R = 14 D = 15 LU = 16 RU = 17 RD = 18 LD = 19
	[RUq_min3, RDq_min3, LDq_min3, LUq_min3] = deal(min(RUq_min2, LU), min(RDq_min2, RU), min(LDq_min2, RD), min(LUq_min2, LD));
	[RUq_min4, RDq_min4, LDq_min4, LUq_min4] = deal(min(RUq_min2, RD), min(RDq_min2, LD), min(LDq_min2, LU), min(LUq_min2, RU));
	g.min48h = reshape(ProjHistMinMax(RUq_min3, 'hor', q) + ProjHistMinMax(LDq_min3, 'hor', q) + ProjHistMinMax(RDq_min4, 'hor', q) + ProjHistMinMax(LUq_min4, 'hor', q) + ...
	ProjHistMinMax(RDq_min3, 'ver', q) + ProjHistMinMax(LUq_min3, 'ver', q) + ProjHistMinMax(RUq_min4, 'ver', q) + ProjHistMinMax(LDq_min4, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	g.min48v = reshape(ProjHistMinMax(RUq_min3, 'ver', q) + ProjHistMinMax(LDq_min3, 'ver', q) + ProjHistMinMax(RDq_min4, 'ver', q) + ProjHistMinMax(LUq_min4, 'ver', q) + ...
	ProjHistMinMax(RDq_min3, 'hor', q) + ProjHistMinMax(LUq_min3, 'hor', q) + ProjHistMinMax(RUq_min4, 'hor', q) + ProjHistMinMax(LDq_min4, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex - 1;
	*/
	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::RU_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::LU_;
	COOC_HIST_MIN_MAX(4, false, 4);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::LD_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::RD_;
	COOC_HIST_MIN_MAX(4, false, 4);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::RD_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::LD_;
	COOC_HIST_MIN_MAX(4, false, 4);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::LU_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::RU_;
	COOC_HIST_MIN_MAX(4, false, 4);

	host_features.index[++host_features.last_index] = 5;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::RD_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::RU_;
	COOC_HIST_MIN_MAX(4, true, 4);

	host_features.index[++host_features.last_index] = 6;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::LU_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::LD_;
	COOC_HIST_MIN_MAX(4, true, 4);

	host_features.index[++host_features.last_index] = 7;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::RU_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::RD_;
	COOC_HIST_MIN_MAX(4, true, 4);

	host_features.index[++host_features.last_index] = 8;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48h"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::LD_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::LU_;
	COOC_HIST_MIN_MAX(4, true, 4);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;



	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::D_3st; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::RD_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::RU_;
	COOC_HIST_MIN_MAX(4, false, 4);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::LU_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::LD_;
	COOC_HIST_MIN_MAX(4, false, 4);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::RU_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::RD_;
	COOC_HIST_MIN_MAX(4, false, 4);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::LD_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::LU_;
	COOC_HIST_MIN_MAX(4, false, 4);

	host_features.index[++host_features.last_index] = 5;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::RU_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::LU_;
	COOC_HIST_MIN_MAX(4, true, 4);

	host_features.index[++host_features.last_index] = 6;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::LD_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::RD_;
	COOC_HIST_MIN_MAX(4, true, 4);

	host_features.index[++host_features.last_index] = 7;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::D_3st; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::RD_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::LD_;
	COOC_HIST_MIN_MAX(4, true, 4);

	host_features.index[++host_features.last_index] = 8;
	host_features.submodel[host_features.last_index] = 24;
	strcpy(host_features.name[host_features.last_index], "s3_min48v"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::LU_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::RU_;
	COOC_HIST_MIN_MAX(4, true, 4);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;

	/*L = 12 U = 13 R = 14 D = 15 LU = 16 RU = 17 RD = 18 LD = 19
	[RUq_min5, RDq_min5, LDq_min5, LUq_min5] = deal(min(RUq_min3, RD), min(RDq_min3, LD), min(LDq_min3, LU), min(LUq_min3, RU));
	[RUq_max5, RDq_max5, LDq_max5, LUq_max5] = deal(max(RUq_max3, RD), max(RDq_max3, LD), max(LDq_max3, LU), max(LUq_max3, RU));
	g.min54 = reshape(ProjHistMinMax(RUq_min5, 'hor', q) + ProjHistMinMax(LDq_min5, 'hor', q) + ProjHistMinMax(RDq_min5, 'hor', q) + ProjHistMinMax(LUq_min5, 'hor', q) + ...
	ProjHistMinMax(RDq_min5, 'ver', q) + ProjHistMinMax(LUq_min5, 'ver', q) + ProjHistMinMax(RUq_min5, 'ver', q) + ProjHistMinMax(LDq_min5, 'ver', q), [], 1);
	g.max54 = reshape(ProjHistMinMax(RUq_max5, 'hor', q) + ProjHistMinMax(LDq_max5, 'hor', q) + ProjHistMinMax(RDq_max5, 'hor', q) + ProjHistMinMax(LUq_max5, 'hor', q) + ...
	ProjHistMinMax(RDq_max5, 'ver', q) + ProjHistMinMax(LUq_max5, 'ver', q) + ProjHistMinMax(RUq_max5, 'ver', q) + ProjHistMinMax(LDq_max5, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	*/
	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 25;
	strcpy(host_features.name[host_features.last_index], "s3_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::RU_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::LU_; residuals_indexes[residual_offset][4] = RESIDUALS_3st::RD_;
	COOC_HIST_MIN_MAX(5, false, 4);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 25;
	strcpy(host_features.name[host_features.last_index], "s3_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::LD_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::RD_; residuals_indexes[residual_offset][4] = RESIDUALS_3st::LU_;
	COOC_HIST_MIN_MAX(5, false, 4);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 25;
	strcpy(host_features.name[host_features.last_index], "s3_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::R_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::RD_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::RU_; residuals_indexes[residual_offset][4] = RESIDUALS_3st::LD_;
	COOC_HIST_MIN_MAX(5, false, 4);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 25;
	strcpy(host_features.name[host_features.last_index], "s3_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::LU_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::LD_; residuals_indexes[residual_offset][4] = RESIDUALS_3st::RU_;
	COOC_HIST_MIN_MAX(5, false, 4);

	host_features.index[++host_features.last_index] = 5;
	host_features.submodel[host_features.last_index] = 25;
	strcpy(host_features.name[host_features.last_index], "s3_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::D_3st; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::RD_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::RU_; residuals_indexes[residual_offset][4] = RESIDUALS_3st::LD_;
	COOC_HIST_MIN_MAX(5, true, 4);

	host_features.index[++host_features.last_index] = 6;
	host_features.submodel[host_features.last_index] = 25;
	strcpy(host_features.name[host_features.last_index], "s3_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::U_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::LU_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::LD_; residuals_indexes[residual_offset][4] = RESIDUALS_3st::RU_;
	COOC_HIST_MIN_MAX(5, true, 4);

	host_features.index[++host_features.last_index] = 7;
	host_features.submodel[host_features.last_index] = 25;
	strcpy(host_features.name[host_features.last_index], "s3_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::U_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::R_;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::RU_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::LU_; residuals_indexes[residual_offset][4] = RESIDUALS_3st::RD_;
	COOC_HIST_MIN_MAX(5, true, 4);

	host_features.index[++host_features.last_index] = 8;
	host_features.submodel[host_features.last_index] = 25;
	strcpy(host_features.name[host_features.last_index], "s3_min54"); residuals_indexes[residual_offset][0] = RESIDUALS_3st::L_; residuals_indexes[residual_offset][1] = RESIDUALS_3st::D_3st;
	residuals_indexes[residual_offset][2] = RESIDUALS_3st::LD_; residuals_indexes[residual_offset][3] = RESIDUALS_3st::RD_; residuals_indexes[residual_offset][4] = RESIDUALS_3st::LU_;
	COOC_HIST_MIN_MAX(5, true, 4);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;


	/*for (int i = 0; i < HIST_COUNT; i++)
	{
		cudaStatus = cudaFree(residuals[i]);
	}
	cudaStatus = cudaFree(dev_residuals);*/
	return;
}
void make_models_5x5(float **dev_residuals, int** residuals, float ** host_residuals, int *dev_MINMAXsymmCoord, int *dev_SPAMsymmCoord, cudaStream_t streams[], PSRM_Features &host_features, int src_width, int src_height)
{
	//return;
	int tile_width = 8, tile_height = 8;
	
	
	/*float **dev_residuals;
	int* residuals[HIST_COUNT];
	
	for (int i = 0; i < HIST_COUNT; i++)
	{
		cudaStatus = cudaMalloc((void**)&residuals[i], 5 * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc for failed!");
		}
	}
	cudaStatus = cudaMalloc((void**)&dev_residuals, 30 * sizeof(float*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "line 789: cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(dev_residuals, host_residuals, 30 * sizeof(float*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "line 793: cudaMemcpy failed!");
	}*/

	int q = 12;
	



	/*Dl = 20		Du = 21		Dr = 22		Db = 23		D = 24
	g.spam14v = reshape(ProjHistSpam(Du, 'ver', q) + ProjHistSpam(Db, 'ver', q) + ProjHistSpam(Dl, 'hor', q) + ProjHistSpam(Dr, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	g.spam14h = reshape(ProjHistSpam(Du, 'hor', q) + ProjHistSpam(Db, 'hor', q) + ProjHistSpam(Dl, 'ver', q) + ProjHistSpam(Dr, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	*/

	host_features.index[++host_features.last_index] = 1; host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 33;
	strcpy(host_features.name[host_features.last_index], "s5x5_spam14v");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_5x5::Du_], true, true, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);
	
	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 33;
	strcpy(host_features.name[host_features.last_index], "s5x5_spam14v");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_5x5::Db_], true, true, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);
	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 33;
	strcpy(host_features.name[host_features.last_index], "s5x5_spam14v");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_5x5::Dl_], false, true, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 33;
	strcpy(host_features.name[host_features.last_index], "s5x5_spam14v");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_5x5::Dr_], false, true, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);
	host_features.index[++host_features.last_index] = 1;
	host_features.submodel[host_features.last_index] = 33;
	strcpy(host_features.name[host_features.last_index], "s5x5_spam14h");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_5x5::Du_], false, false, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 33;
	strcpy(host_features.name[host_features.last_index], "s5x5_spam14h");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_5x5::Db_], false, false, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);
	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 33;
	strcpy(host_features.name[host_features.last_index], "s5x5_spam14h");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_5x5::Dl_], true, false, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 33;
	strcpy(host_features.name[host_features.last_index], "s5x5_spam14h");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_5x5::Dr_], true, false, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;


	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 34;
	strcpy(host_features.name[host_features.last_index], "s5x5_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_5x5::Du_; residuals_indexes[residual_offset][1] = RESIDUALS_5x5::Dl_;
	COOC_HIST_MIN_MAX(2, false, 4);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 34;
	strcpy(host_features.name[host_features.last_index], "s5x5_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_5x5::Db_; residuals_indexes[residual_offset][1] = RESIDUALS_5x5::Dr_;
	COOC_HIST_MIN_MAX(2, false, 4);

	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 34;
	strcpy(host_features.name[host_features.last_index], "s5x5_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_5x5::Du_; residuals_indexes[residual_offset][1] = RESIDUALS_5x5::Dr_;
	COOC_HIST_MIN_MAX(2, false, 4);

	host_features.index[++host_features.last_index] = 4;
	host_features.submodel[host_features.last_index] = 34;
	strcpy(host_features.name[host_features.last_index], "s5x5_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_5x5::Db_; residuals_indexes[residual_offset][1] = RESIDUALS_5x5::Dl_;
	COOC_HIST_MIN_MAX(2, false, 4);

	host_features.index[++host_features.last_index] = 5;
	host_features.submodel[host_features.last_index] = 34;
	strcpy(host_features.name[host_features.last_index], "s5x5_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_5x5::Du_; residuals_indexes[residual_offset][1] = RESIDUALS_5x5::Dl_;
	COOC_HIST_MIN_MAX(2, true, 4);

	host_features.index[++host_features.last_index] = 6;
	host_features.submodel[host_features.last_index] = 34;
	strcpy(host_features.name[host_features.last_index], "s5x5_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_5x5::Db_; residuals_indexes[residual_offset][1] = RESIDUALS_5x5::Dr_;
	COOC_HIST_MIN_MAX(2, true, 4);

	host_features.index[++host_features.last_index] = 7;
	host_features.submodel[host_features.last_index] = 34;
	strcpy(host_features.name[host_features.last_index], "s5x5_min24");  residuals_indexes[residual_offset][0] = RESIDUALS_5x5::Du_; residuals_indexes[residual_offset][1] = RESIDUALS_5x5::Dr_;
	COOC_HIST_MIN_MAX(2, true, 4);

	host_features.index[++host_features.last_index] = 8;
	host_features.submodel[host_features.last_index] = 34;
	strcpy(host_features.name[host_features.last_index], "s5x5_min24"); residuals_indexes[residual_offset][0] = RESIDUALS_5x5::Db_; residuals_indexes[residual_offset][1] = RESIDUALS_5x5::Dl_;
	COOC_HIST_MIN_MAX(2, true, 4);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;




	/*Dl = 20		Du = 21		Dr = 22		Db = 23		D = 24
	[UEq_min, REq_min] = deal(min(Du, Db), min(Dr, Dl));
	g.min22h = reshape(ProjHistMinMax(UEq_min, 'hor', q) + ProjHistMinMax(REq_min, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	g.min22v = reshape(ProjHistMinMax(UEq_min, 'ver', q) + ProjHistMinMax(REq_min, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex - 1;
	[UEq_max, REq_max] = deal(max(Du, Db), max(Dr, Dl));
	g.max22h = reshape(ProjHistMinMax(UEq_max, 'hor', q) + ProjHistMinMax(REq_max, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;
	g.max22v = reshape(ProjHistMinMax(UEq_max, 'ver', q) + ProjHistMinMax(REq_max, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 35;
	strcpy(host_features.name[host_features.last_index], "s5x5_min22v"); residuals_indexes[residual_offset][0] = RESIDUALS_5x5::Du_; residuals_indexes[residual_offset][1] = RESIDUALS_5x5::Db_;
	COOC_HIST_MIN_MAX(2, true, 4);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 35;
	strcpy(host_features.name[host_features.last_index], "s5x5_min22v"); residuals_indexes[residual_offset][0] = RESIDUALS_5x5::Dr_; residuals_indexes[residual_offset][1] = RESIDUALS_5x5::Dl_;
	COOC_HIST_MIN_MAX(2, false, 4);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;

	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 36;
	strcpy(host_features.name[host_features.last_index], "s5x5_min22h"); residuals_indexes[residual_offset][0] = RESIDUALS_5x5::Du_; residuals_indexes[residual_offset][1] = RESIDUALS_5x5::Db_;
	COOC_HIST_MIN_MAX(2, false, 4);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 36;
	strcpy(host_features.name[host_features.last_index], "s5x5_min22h"); residuals_indexes[residual_offset][0] = RESIDUALS_5x5::Dr_; residuals_indexes[residual_offset][1] = RESIDUALS_5x5::Dl_;
	COOC_HIST_MIN_MAX(2, true, 4);

	
	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;




	/*Dl = 20		Du = 21		Dr = 22		Db = 23		D = 24
	[Dmin5, Dmax5] = deal(min(Dmin1, Dmin2), max(Dmax1, Dmax2));
	g.min41 = reshape(ProjHistMinMax(Dmin5, 'ver', q) + ProjHistMinMax(Dmin5, 'hor', q), [], 1);
	g.max41 = reshape(ProjHistMinMax(Dmax5, 'ver', q) + ProjHistMinMax(Dmax5, 'hor', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 37;
	strcpy(host_features.name[host_features.last_index], "s5x5_min41"); residuals_indexes[residual_offset][0] = RESIDUALS_5x5::Du_; residuals_indexes[residual_offset][1] = RESIDUALS_5x5::Db_;
	residuals_indexes[residual_offset][2] = RESIDUALS_5x5::Dr_; residuals_indexes[residual_offset][3] = RESIDUALS_5x5::Dl_;
	COOC_HIST_MIN_MAX(4, true, 4);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 37;
	strcpy(host_features.name[host_features.last_index], "s5x5_min41"); residuals_indexes[residual_offset][0] = RESIDUALS_5x5::Du_; residuals_indexes[residual_offset][1] = RESIDUALS_5x5::Db_;
	residuals_indexes[residual_offset][2] = RESIDUALS_5x5::Dr_; residuals_indexes[residual_offset][3] = RESIDUALS_5x5::Dl_;
	COOC_HIST_MIN_MAX(4, false, 4);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;
	/*Dl = 25		Du = 26		Dr = 27		Db = 28		D = 29
	D = Residual(X, 3, 'KV');
	g.spam11 = reshape(ProjHistSpam(D, 'hor', q) + ProjHistSpam(D, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/


	host_features.index[++host_features.last_index] = 1;
	host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 32;
	strcpy(host_features.name[host_features.last_index], "s5x5_spam11");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_5x5::D_5st], false, true, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 32;
	strcpy(host_features.name[host_features.last_index], "s5x5_spam11");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_5x5::D_5st], true, true, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], q, src_width, src_height, tile_width, tile_height);
	/*Dl = 20		Du = 21		Dr = 22		Db = 23		D = 24
	D = Residual(X, 2, 'KB');
	g.spam11 = reshape(ProjHistSpam(D, 'hor', q) + ProjHistSpam(D, 'ver', q), [], 1); settings.seedIndex = settings.seedIndex + 1;*/
	q = 4;
	host_features.index[++host_features.last_index] = 1;
	host_features.submodel[host_features.last_index] = 26;
	strcpy(host_features.name[host_features.last_index], "s3x3_spam11");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_3x3::D_4st], false, false, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], 4, src_width, src_height, tile_width, tile_height);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 26;
	strcpy(host_features.name[host_features.last_index], "s3x3_spam11");
	COOC_HIST_SPAM(host_residuals[RESIDUALS_3x3::D_4st], true, false, 4);// , dev_kernels, dev_kernels_t, host_features.hists[host_features.last_index], 4, src_width, src_height, tile_width, tile_height);

	host_features.sub_model_count[host_features.submodel_index] = host_features.last_index - host_features.sub_model_index[host_features.submodel_index] + 1;
	

	return;
}


__global__ void cooc_hist_min_max(float** dev_residuals, int*residuals_index, int residual_count, int *dev_MINMAXsymmCoord, int *dev_SPAMsymmCoord, bool trans, int* out_hist, int q, int src_width, int src_height, int tile_width, int tile_height, int border)
{
	__shared__ int hist0[SPAM_SYM_COUNT * 2];
	const float multi[] = { 1, 5, 25, 125 };
	register float  ptr_min[12][12];
	register float  ptr_max[12][12];
	register float  ptr_r[12][12];
	register float  ptr_l[12][12];
	float res1;
	float min_max;
	int i, j, w, z, m, n;
	int tile_top_left, tile_top_left0, row, col;
	int feaNumber = 0;
	int dimToIncrease = 0;
	int o = 0;
	float*residuals[5];

	for (i = 0; i < residual_count; i++)
	{
		residuals[i] = dev_residuals[residuals_index[i]];
		//printf("\n%d\t %d, %d\t%d\n", i, threadIdx.x, threadIdx.y, residuals_index[i]);
	}

	__syncthreads();
	if (!threadIdx.x && !threadIdx.y)
	{
		memset(hist0, 0, SPAM_SYM_COUNT * 2 * sizeof(int));
	}

	__syncthreads();
	tile_top_left =
		(blockIdx.y * tile_height * blockDim.y + threadIdx.y * tile_height) * (src_width)+
		blockIdx.x * (blockDim.x * tile_width) +
		threadIdx.x * tile_width;
	tile_top_left0 = tile_top_left;
	row = tile_top_left / src_width + tile_height - 1;
	col = tile_top_left % src_width + tile_width - 1;
	int endx = src_width - border - 1;
	int endy = src_height - border - 1;
	if ((col > (endx) - ORDER + 1) && !trans)
	{
		tile_width =  (endx)-tile_top_left % src_width - ORDER + 2;
		if (tile_width < 0)
			return;
	}
	if ((row > (endy)) && !trans)
	{
		tile_height = (endy)-tile_top_left / src_width + 1;
	}
	if ((row > (endy)- ORDER + 1) && trans)
	{
		tile_height = (endy)-tile_top_left / src_width - ORDER + 2;
		if (tile_height < 0)
			return;
	}
	if ((col > (endx)) && trans)
	{
		tile_width = (endx)-tile_top_left % src_width + 1;
	}

	//max
	{
		min_max = -1000000.f;
		for (i = 0; i < tile_height + KERNEL_RIGHT_BOTTOM_PADD; i++)//+3 for 4*4 kernel
		{
			for (j = 0; j < tile_width + KERNEL_RIGHT_BOTTOM_PADD; j++)//+3 for 4*4 kernel
			{
				min_max = -1000000.0f;
				for (m = 0; m < residual_count; m++)
				{
					//res1 = residuals[m][tile_top_left0 + j];
					ptr_r[i][j] = residuals[0][tile_top_left0 + j];
					ptr_l[i][j] = residuals[1][tile_top_left0 + j];
					res1 = (residuals[m][tile_top_left0 + j] / q);
					if (res1 - floorf(res1) > 0.5) res1 = ceilf(res1);
					else res1 = floorf(res1);
					res1 = res1 < -T ? -T : (res1 > T ? T : res1);
					if (res1 > min_max)
					{
						min_max = res1;
					}
				}
				//res1 = roundf(min_max / q);
				ptr_max[i][j] = min_max;
			}
			tile_top_left0 += src_width;
		}
	}
	//min
	{
		min_max = 10000.0f;
		tile_top_left0 = tile_top_left;
		//printf("\t%d\n", tile_top_left0);
		for (i = 0; i < tile_height + KERNEL_RIGHT_BOTTOM_PADD; i++)//+3 for 4*4 kernel
		{
			//printf("\n");
			for (j = 0; j < tile_width + KERNEL_RIGHT_BOTTOM_PADD; j++)//+3 for 4*4 kernel
			{
				min_max = 10000.0f;
				for (m = 0; m < residual_count; m++)
				{
					//res1 = residuals[m][tile_top_left0 + j];
					res1 = (residuals[m][tile_top_left0 + j] / q);
					if (res1 - floorf(res1) > 0.5) res1 =ceilf(res1);
					else res1 = floorf(res1);
					res1 = res1 < -T ? -T : (res1 > T ? T : res1);
					//printf(" %f", res1);
					if (res1 < min_max)
					{
						min_max = res1;
					}
				}
				//res1 = roundf(min_max/ q) ;
				//ptr_min[i][j] = res1 < -T ? -T : res1 > T ? T : res1;
				ptr_min[i][j] = min_max;
			}
			tile_top_left0 += src_width;
		}

	}
	/*if (blockIdx.x == 0 && blockIdx.y == 7 && threadIdx.x == 0 && threadIdx.y == 7)
	{
		printf("\ntile_height = %d\n", tile_height);
		printf("\n\n");
		for (i = 0; i < tile_height + 3; i++)
		{
			printf("\n");
			for (j = 0; j < tile_width + 3; j++)
			{
				printf(" %f", ptr_l[i][j]);
			}
		}
		printf("\n\n");
		for (i = 0; i < tile_height + 3; i++)
		{
			printf("\n");
			for (j = 0; j < tile_width + 3; j++)
			{
				printf(" %f", ptr_r[i][j]);
			}
		}
		printf("\n\n");
		printf("\n\n");
		for (i = 0; i < tile_height + 3; i++)
		{
			printf("\n");
			for (j = 0; j < tile_width + 3; j++)
			{
				printf(" %f", ptr_min[i][j]);
			}
		}
		printf("\n\n");
		for (i = 0; i < tile_height + 3; i++)
		{
			printf("\n");
			for (j = 0; j < tile_width + 3; j++)
			{
				printf(" %f", ptr_max[i][j]);
			}
		}
		printf("\n\n");
	}*/
	if (!trans)
	{
		for (i = 0; i < tile_height; i++)
		{
			for (j = 0; j < tile_width; j++)
			{
				feaNumber = 0;
				for (o = 0; o < ORDER; o++)
					feaNumber += (ptr_min[i][j + o] + T) * multi[o];

				dimToIncrease = dev_MINMAXsymmCoord[feaNumber];
				atomicAdd(&hist0[dimToIncrease], 1);

				// MAX
				feaNumber = 0;
				for (o = 0; o < ORDER; o++)
					feaNumber += (ptr_max[i][j + o] + T) * multi[o];

				feaNumber += FULL_DIM;
				dimToIncrease = dev_MINMAXsymmCoord[feaNumber];
				atomicAdd(&hist0[dimToIncrease], 1);
			}
		}

	}
	else
	{
		for (i = 0; i < tile_height; i++)
		{
			for (j = 0; j < tile_width; j++)
			{
				feaNumber = 0;
				for (o = 0; o < ORDER; o++)
					feaNumber += (ptr_min[i + o][j] + T)* multi[o];

				dimToIncrease = dev_MINMAXsymmCoord[feaNumber];
				atomicAdd(&hist0[dimToIncrease], 1);

				// MAX
				feaNumber = 0;
				for (o = 0; o < ORDER; o++)
					feaNumber += (ptr_max[i + o][j] + T)* multi[o];

				feaNumber += FULL_DIM;
				dimToIncrease = dev_MINMAXsymmCoord[feaNumber];
				atomicAdd(&hist0[dimToIncrease], 1);
			}
		}

	}

	__syncthreads();
	if (!threadIdx.x && !threadIdx.y)
	{
		for (i = 0; i < SPAM_SYM_COUNT * 2; i++)
		{
			atomicAdd(&out_hist[i], hist0[i]);
		}

	}
	return;

}

__global__ void cooc_hist_spam(float* first_residual, int *dev_MINMAXsymmCoord, int *dev_SPAMsymmCoord, bool trans, bool hist_position, int* out_hist, int q, int src_width, int src_height, int tile_width, int tile_height, int border)
{

	//return;
	__shared__ int hist0[SPAM_SYM_COUNT];
	const float multi[] = { 1, 5, 25, 125 };
	int feaNumber = 0;
	int dimToIncrease = 0;
	int o = 0;
	register float  ptr_spam[12][12];
	register float  ptr_l[12][12];

	int i, j, w, z, m, n;
	int tile_top_left, tile_top_left0, row, col;
	int hist_pos = (hist_position == false) ? 0 : SPAM_SYM_COUNT;
	float res1 = 0;

	

	__syncthreads();
	if (!threadIdx.x && !threadIdx.y)
	{
		memset(hist0, 0, SPAM_SYM_COUNT * sizeof(int));
	}

	__syncthreads();
	tile_top_left =
		(blockIdx.y * tile_height * blockDim.y + threadIdx.y * tile_height) * (src_width)+
		blockIdx.x * (blockDim.x * tile_width) +
		threadIdx.x * tile_width;
	tile_top_left0 = tile_top_left;
	row = tile_top_left / src_width + tile_height - 1;
	col = tile_top_left % src_width + tile_width - 1;
	if ((col >= (src_width - border) - ORDER) && !trans)
	{
		tile_width = (src_width - border) - tile_top_left % src_width + 1 - ORDER;
		if (tile_width < 0)
			return;
	}
	if ((row >= (src_height - border) ) && !trans)
	{
		tile_height = (src_height - border) - tile_top_left / src_width ;
	}
	if ((row >= (src_height - border) - ORDER) && trans)
	{
		tile_height = (src_height - border) - tile_top_left / src_width + 1 - ORDER;
		if (tile_height < 0)
			return;
	}
	if ((col >= (src_width - border)) && trans)
	{
		tile_width = (src_width - border) - tile_top_left % src_width ;
	}

	for (int i = 0; i < tile_height + KERNEL_RIGHT_BOTTOM_PADD; i++)//+3 for 4*4 kernel
	{
		for (int j = 0; j < tile_width + KERNEL_RIGHT_BOTTOM_PADD; j++)//+3 for 4*4 kernel
		{
			res1 = (first_residual[tile_top_left0 + j] / q);
			ptr_l[i][j] = first_residual[tile_top_left0 + j];
			if (res1 - floorf(res1) > 0.5) res1 = ceilf(res1);
			else res1 = floorf(res1);
			res1 = res1 < -T ? -T : (res1 > T ? T : res1);
			ptr_spam[i][j] = res1 ;
		}
		tile_top_left0 += src_width;
	}
	/*if (!blockIdx.x && !blockIdx.y && !threadIdx.x && !threadIdx.y)
	{
		printf("\ntile_height = %d\n", tile_height);
		printf("\n\n");
		for (i = 0; i < tile_height + 3; i++)
		{
			printf("\n");
			for (j = 0; j < tile_width + 3; j++)
			{
				printf(" %f", ptr_spam[i][j]);
			}
		}
		printf("\n\n");
		printf("\n\n");
		for (i = 0; i < tile_height + 3; i++)
		{
			printf("\n");
			for (j = 0; j < tile_width + 3; j++)
			{
				printf(" %f", ptr_l[i][j]);
			}
		}
		printf("\n\n");

	}*/
	//return;
	if (!trans)
	{
		for (i = 0; i < tile_height; i++)
		{
			for (j = 0; j < tile_width; j++)
			{
				feaNumber = 0;
				for (o = 0; o < ORDER; o++)
					feaNumber += (ptr_spam[i][j + o] + T)* multi[o];

				dimToIncrease = dev_SPAMsymmCoord[feaNumber];
				//hist0[dimToIncrease]++;
				/*if (dimToIncrease > 168)
				{
					printf("###############     %d , %d", blockIdx.x, blockIdx.y);
				}*/
				atomicAdd(&hist0[dimToIncrease], 1);

			}
		}
	}
	else
	{
		for (i = 0; i < tile_height; i++)
		{
			for (j = 0; j < tile_width; j++)
			{
				feaNumber = 0;
				for (o = 0; o < ORDER; o++)
					feaNumber += (ptr_spam[i + o][j] + T)* multi[o];

				dimToIncrease = dev_SPAMsymmCoord[feaNumber];
				//hist0[dimToIncrease]++;
				atomicAdd(&hist0[dimToIncrease], 1);

			}
		}
	}
	__syncthreads();
	if (!threadIdx.x && !threadIdx.y)
	{
		for (i = 0; i < SPAM_SYM_COUNT; i++)
		{
			atomicAdd(&out_hist[i + hist_pos], hist0[i]);
		}

	}
	return;

	
}