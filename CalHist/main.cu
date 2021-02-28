
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <string>
#include <random>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <opencv.hpp>

#include "residualMaker.cuh"
#include "168modelMaker.cuh"
using namespace cv;
using namespace std;

void free_hists(PSRM_Features &host_features)
{
	for (size_t i = 0; i < HIST_COUNT; i++)
	{
		cudaFree(host_features.hists[i]);
	}
}
int save_features(string file_path, int class_id, int** hists)
{
	int ret_val = 0;
	file_path += "_Features.fea";
	FILE* fp_result = fopen(file_path.c_str(), "wb");
	if (fp_result)
	{
		std::fprintf(fp_result, "%d\n", class_id);
		float sum0 = 0;
		for (size_t i = 0; i < COUNT_OF_SUBMODELS; i++)
		{
			sum0 = 0;
			for (size_t j = 0; j < SPAM_SYM_COUNT * 2; j++)
			{
				sum0 += hists[i][j];
			}
			sum0 /= 4.0f;
			for (size_t j = 0; j < SPAM_SYM_COUNT * 2; j++)
			{
				if (sum0 > 0)
					std::fprintf(fp_result, "\t%f", hists[i][j] / sum0);
				else
					std::fprintf(fp_result, "\t%f", 0);
			}
			std::fprintf(fp_result, "\n");
		}
		fclose(fp_result);
	}
	else
	{
		ret_val = -1;
	}
	return ret_val;
}
int MINMAXsymmCoord[ 2 * FULL_DIM];
int SPAMsymmCoord[FULL_DIM];
void ComputeSymmCoords()
{
	bool symmSign = true,  symmReverse = true,  symmMinMax = true;
	// Preparation of inSymCoord matrix for co-occurrence and symmetrization
	int B = 2 * T + 1;

	int alreadyUsed;

	// MINMAX
	alreadyUsed = 0;
	//MINMAXsymmCoord = new int[2 * FULL_DIM]; // [0, FULL_DIM-1] = min; [FULL_DIM, 2*FULL_DIM-1] = max
	for (int i = 0; i<2 * FULL_DIM; i++) MINMAXsymmCoord[i] = -1;

	for (int numIter = 0; numIter < FULL_DIM; numIter++)
	{
		if (MINMAXsymmCoord[numIter] == -1)
		{
			int coordReverse = 0;
			int num = numIter;
			for (int i = 0; i<ORDER; i++)
			{
				coordReverse += (num % B) * ((int)std::pow((float)B, ORDER - i - 1));
				num = num / B;
			}
			// To the same bin: min(X), max(-X), min(Xreverse), max(-Xreverse)
			if (MINMAXsymmCoord[numIter] == -1)
			{
				MINMAXsymmCoord[numIter] = alreadyUsed; // min(X)
				if (symmMinMax) MINMAXsymmCoord[2 * FULL_DIM - numIter - 1] = alreadyUsed; // max(-X)
				if (symmReverse) MINMAXsymmCoord[coordReverse] = alreadyUsed; // min(Xreverse)
				if ((symmMinMax) && (symmReverse)) MINMAXsymmCoord[2 * FULL_DIM - coordReverse - 1] = alreadyUsed; // max(-Xreverse)
				alreadyUsed++;
			}
		}
	}
	for (int numIter = 0; numIter < FULL_DIM; numIter++)
	{
		if (MINMAXsymmCoord[FULL_DIM + numIter] == -1)
		{
			int coordReverse = 0;
			int num = numIter;
			for (int i = 0; i<ORDER; i++)
			{
				coordReverse += (num % B) * ((int)std::pow((float)B, ORDER - i - 1));
				num = num / B;
			}
			// To the same bin: max(X), min(-X), max(Xreverse), min(-Xreverse)
			if (MINMAXsymmCoord[FULL_DIM + numIter] == -1)
			{
				MINMAXsymmCoord[FULL_DIM + numIter] = alreadyUsed; // max(X)
				if (symmMinMax) MINMAXsymmCoord[FULL_DIM - numIter - 1] = alreadyUsed; // min(-X)
				if (symmReverse) MINMAXsymmCoord[FULL_DIM + coordReverse] = alreadyUsed; // max(Xreverse)
				if ((symmMinMax) && (symmReverse)) MINMAXsymmCoord[FULL_DIM - coordReverse - 1] = alreadyUsed; // min(-Xreverse)
				alreadyUsed++;
			}
		}
	}

	// SPAM
	alreadyUsed = 0;
	//SPAMsymmCoord = new int[FULL_DIM];
	for (int i = 0; i<FULL_DIM; i++) SPAMsymmCoord[i] = -1;
	for (int numIter = 0; numIter < FULL_DIM; numIter++)
	{
		if (SPAMsymmCoord[numIter] == -1)
		{
			int coordReverse = 0;
			int num = numIter;
			for (int i = 0; i<ORDER; i++)
			{
				coordReverse += (num % B) * ((int)std::pow((float)B, ORDER - i - 1));
				num = num / B;
			}
			// To the same bin: X, -X, Xreverse, -Xreverse
			SPAMsymmCoord[numIter] = alreadyUsed; // X
			if (symmSign) SPAMsymmCoord[FULL_DIM - numIter - 1] = alreadyUsed; // -X
			if (symmReverse) SPAMsymmCoord[coordReverse] = alreadyUsed; // Xreverse
			if ((symmSign) && (symmReverse)) SPAMsymmCoord[FULL_DIM - coordReverse - 1] = alreadyUsed; // -Xreverse
			alreadyUsed++;
		}
	}
	// In order to have the same order of the features as the matlab SRM - shift +1
	for (int i = 0; i<FULL_DIM; i++)
	{
		if (SPAMsymmCoord[i] == alreadyUsed - 1) SPAMsymmCoord[i] = 0;
		else SPAMsymmCoord[i]++;
	}
}

int main(int argc, char*argv[])
{
	if (argc < 2)
	{
		printf("Please, Enter the List File Path as first argument");
		getchar();
		return -1;
	}
	printf("file list is %s\n", argv[1]);
	FILE* fp_list = fopen(argv[1], "r");
	FILE *fp_existance = 0;
	if (!fp_list)
	{
		printf("the your Entered List File Path is not exist");
		getchar();
		return -2;
	}
	cudaError_t cudaStatus;
	ComputeSymmCoords();
	int *dev_MINMAXsymmCoord;
	int *dev_SPAMsymmCoord;
	cudaStatus = cudaMalloc(&dev_MINMAXsymmCoord, FULL_DIM * 2 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
	}
	cudaStatus = cudaMemcpy(dev_MINMAXsymmCoord, MINMAXsymmCoord, FULL_DIM * 2 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
	}
	cudaStatus = cudaMalloc(&dev_SPAMsymmCoord, FULL_DIM * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
	}
	cudaStatus = cudaMemcpy(dev_SPAMsymmCoord, SPAMsymmCoord, FULL_DIM * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
	}

	PSRM_Features host_features = {};
	int *hists[COUNT_OF_SUBMODELS];
	
	for (size_t i = 0; i < COUNT_OF_SUBMODELS; i++)
	{
		hists[i] = new int[SPAM_SYM_COUNT * 2];
	}
	for (size_t i = 0; i < HIST_COUNT; i++)
	{
		cudaStatus = cudaMalloc(&(host_features.hists[i]), SPAM_SYM_COUNT * 2 * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, "\nline %d: cudaMalloc failed!", __LINE__);
			return 1;
		}
	}
	
	int class_id = 0;
	int dim1 = 0, dim2 = 0;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	char str_file_path[_MAX_PATH];
	float* dev_src = NULL, *host_dev_residuals[KERNELS_COUNT] = {};
	float* host_src[KERNELS_COUNT] = {};
	cudaStream_t streams[STREAM_COUNT + 5];
	cudaStatus = cudaSetDevice(0);
	cudaEvent_t start[STREAM_COUNT], stop[STREAM_COUNT];
	float **dev_residuals;
	int* residuals[HIST_COUNT];
	const int MAX_COLS = 1024 + (1024 / 7.0f + KERNEL_RIGHT_BOTTOM_PADD) * KERNEL_RIGHT_BOTTOM_PADD;
	const int MAX_ROWS = 1024 + (1024 / 7.0f + KERNEL_RIGHT_BOTTOM_PADD) * KERNEL_RIGHT_BOTTOM_PADD;
	const int TILE_HEIGHT = 8;
	const int TILE_WEIGHT = 8;
	uint3 blocks_res = { 128, 128, 1 }, threads_res = { 2, 2, 1 };

	for (size_t i = 0; i < STREAM_COUNT + 5; i++)
	{
		cudaStreamCreate(&streams[i]);
		if (i >= STREAM_COUNT)continue;
		cudaEventCreate(&start[i]);
		cudaEventCreate(&stop[i]);
	}
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "\nline %d: cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", __LINE__);
	}
	

	cudaStatus = cudaMalloc((void**)&dev_src, MAX_COLS * MAX_ROWS * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "\nline %d: cudaMalloc failed!", __LINE__);
		return 1;
	}
	for (size_t i = 0; i < KERNELS_COUNT; i++)
	{
		cudaStatus = cudaHostAlloc((void**)&host_src[i], MAX_COLS * MAX_ROWS * sizeof(float), cudaHostAllocMapped | cudaHostAllocWriteCombined);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			return 1;
		}

		cudaStatus = cudaHostGetDevicePointer(&host_dev_residuals[i], host_src[i], 0);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			return 1;
		}
	}
	for (int i = 0; i < HIST_COUNT; i++)
	{
		cudaStatus = cudaMalloc((void**)&residuals[i], 5 * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, "\nline %d: cudaMalloc for failed!", __LINE__);
		}

	}
	cudaStatus = cudaMalloc((void**)&dev_residuals, KERNELS_COUNT * sizeof(float*));
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "\nline %d: cudaMalloc failed!", __LINE__);
	}
	cudaStatus = cudaMemcpy(dev_residuals, host_dev_residuals, KERNELS_COUNT * sizeof(float*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "\nline %d: cudaMemcpy failed!", __LINE__);
	}
	//std::fprintf(stderr, "\nline %d: cudaMemcpy failed!", __LINE__);
	while (!feof(fp_list))
	{
		host_features.last_index = 0;
		host_features.submodel_index = 0;
		if (!fscanf(fp_list, "%d\t%s\n", &class_id, str_file_path))
		{
			printf("\nProcessing is finished");
			break;
		}
		printf("\nclassID = %d, path = %s", class_id, str_file_path);
		fp_existance = fopen((string(str_file_path) + string("_Features.fea")).c_str(), "rb");
		if (fp_existance)
		{
			fclose(fp_existance);
			printf("\tProcessed in the Past");
			//continue;
		}
		Mat img = imread(str_file_path, CV_LOAD_IMAGE_GRAYSCALE);
		if (!img.data || !img.cols || !img.rows)
		{
			printf("\nNULL image, %s", str_file_path);
			continue;
		}
		if (img.cols > 1018 && img.rows > 1018)
			img = img(Rect(0, 0, 1018, 1018));
		else if (img.cols > 1018)
			img = img(Rect(0, 0, 1018, img.rows));
		else if (img.rows > 1018)
			img = img(Rect(0, 0, img.cols, 1018));
		//copyMakeBorder(img, img, 2, 2, 2, 2, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
		dim1 = (int)ceil(sqrt(img.cols / 8.0f));
		blocks_res.x =  dim1;
		threads_res.x = (unsigned int)ceil(img.cols / 8.0f / dim1);
		dim2 = (int)ceil(sqrt(img.rows / 8.0f));
		blocks_res.y = dim2;
		threads_res.y =  (unsigned int)ceil(img.rows / 8.0f / dim2);
		blocks = blocks_res;
		threads = threads_res;
		//imshow("1", img); cvWaitKey();
		img.convertTo(img, CV_32FC1);
		cudaStatus = cudaMemcpy(dev_src, img.data, img.rows * img.cols * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			continue;
		}
		cudaStatus = cudaDeviceSynchronize();
		// do some work on the GPU

		float t1 = (float)clock();
		int i = 0;
		int kernel_index = 1;
		cudaEventRecord(start[i], streams[STREAM_COUNT + 0]);

		for (; i < 8; i++)
		{
			make_res_1st << <blocks_res, threads_res, 0, streams[STREAM_COUNT + 0] >> >(dev_src, host_dev_residuals[i], img.cols, img.rows, kernel_index++, TILE_WEIGHT, TILE_HEIGHT);

		}
		cudaStatus = cudaStreamSynchronize(streams[STREAM_COUNT + 0]);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			//return 1;
		}
		/*
		Mat newImg = Mat(img.rows + 3 * (blocks.y + 1), img.cols + 3 * (blocks.x+1), CV_32F);
		cudaStatus = cudaMemcpy(newImg.data, host_dev_residuals[0], (img.cols + 3 * (blocks.x + 1)) * (img.rows + 3 * (blocks.y + 1)) * sizeof(float), cudaMemcpyDeviceToHost);
		*/
		/*Mat newImg = Mat(img.rows + 3 * (img.rows / 8.0 + 1), img.cols + 3 * (img.cols / 8.0 + 1), CV_32F);
		cudaStatus = cudaMemcpy(newImg.data, host_dev_residuals[0], (newImg.cols * newImg.rows) * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			continue;
		}
		//newImg.convertTo(newImg, CV_8UC1);
		imwrite("e:\\0--1024.bmp", newImg);*/
		/*imshow("2", img); cvWaitKey();
		*/
		cudaEventRecord(stop[0], streams[STREAM_COUNT + 0]); cudaEventSynchronize(stop[0]);
		make_models_1st(dev_residuals, residuals, host_dev_residuals, dev_MINMAXsymmCoord, dev_SPAMsymmCoord, streams, host_features, img.cols, img.rows);
		printf("\n make_models_1st is done\n");
		kernel_index = 1; cudaEventRecord(start[1], streams[STREAM_COUNT + 1]);
		for (; i < 12; i++)
		{
			make_res_2st << <blocks_res, threads_res, 0, streams[STREAM_COUNT + 1] >> >(dev_src, host_dev_residuals[i], img.cols, img.rows, kernel_index++, TILE_WEIGHT, TILE_HEIGHT);

		}
		cudaStatus = cudaStreamSynchronize(streams[STREAM_COUNT + 1]);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			//return 1;
		}
		cudaEventRecord(stop[1], streams[STREAM_COUNT + 1]); cudaEventSynchronize(stop[1]);
		make_models_2st(dev_residuals, residuals, host_dev_residuals, dev_MINMAXsymmCoord, dev_SPAMsymmCoord, streams, host_features, img.cols, img.rows);
		printf("\n make_models_2st is done\n");
		kernel_index = 1; cudaEventRecord(start[2], streams[STREAM_COUNT + 2]);
		for (; i < 20; i++)
		{
			make_res_3st << <blocks_res, threads_res, 0, streams[STREAM_COUNT + 2] >> >(dev_src, host_dev_residuals[i], img.cols, img.rows, kernel_index++, TILE_WEIGHT, TILE_HEIGHT);

		}
		cudaStatus = cudaStreamSynchronize(streams[STREAM_COUNT + 2]);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			//return 1;
		}
		cudaEventRecord(stop[2], streams[STREAM_COUNT + 2]); cudaEventSynchronize(stop[2]);
		make_models_3st(dev_residuals, residuals, host_dev_residuals, dev_MINMAXsymmCoord, dev_SPAMsymmCoord, streams, host_features, img.cols, img.rows);
		printf("\n make_models_3st is done\n");
		kernel_index = 1; cudaEventRecord(start[3], streams[STREAM_COUNT + 3]);
		for (; i < 25; i++)
		{
			make_res_3x3 << <blocks_res, threads_res, 0, streams[STREAM_COUNT + 3] >> >(dev_src, host_dev_residuals[i], img.cols, img.rows, kernel_index++, TILE_WEIGHT, TILE_HEIGHT);

		}
		cudaStatus = cudaStreamSynchronize(streams[STREAM_COUNT + 3]);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			//return 1;
		}
		cudaEventRecord(stop[3], streams[STREAM_COUNT + 3]); cudaEventSynchronize(stop[3]);
		make_models_3x3(dev_residuals, residuals, host_dev_residuals, dev_MINMAXsymmCoord, dev_SPAMsymmCoord, streams, host_features, img.cols, img.rows);
		printf("\n make_models_3x3 is done\n");
		kernel_index = 1; cudaEventRecord(start[4], streams[STREAM_COUNT + 4]);
		for (; i < KERNELS_COUNT; i++)
		{
			make_res_5x5 << <blocks_res, threads_res, 0, streams[STREAM_COUNT + 4] >> >(dev_src, host_dev_residuals[i], img.cols, img.rows, kernel_index++, TILE_WEIGHT, TILE_HEIGHT);
		}
		cudaStatus = cudaStreamSynchronize(streams[STREAM_COUNT + 4]);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			//return 1;
		}
		cudaEventRecord(stop[4], streams[STREAM_COUNT + 4]); cudaEventSynchronize(stop[4]);
		make_models_5x5(dev_residuals, residuals, host_dev_residuals, dev_MINMAXsymmCoord, dev_SPAMsymmCoord, streams, host_features, img.cols, img.rows);
		printf("\n make_models_5x5 is done\n");
		for (size_t i = 0; i < STREAM_COUNT; i++)
		{
			cudaStreamSynchronize(streams[i]);
		}
		/*
		*/








		compute_submodels(host_features);














		cudaDeviceSynchronize();
		float t2 = (clock() - t1) / CLOCKS_PER_SEC;
		printf("\n %d\n", host_features.submodel_index);
		
		for (size_t i = 0; i < COUNT_OF_SUBMODELS; i++)
		{
			printf("\n%d", host_features.sub_model_index[i]);
			cudaStatus = cudaMemcpy(hists[i], host_features.hists[host_features.sub_model_index[i]], SPAM_SYM_COUNT * 2 * sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				std::fprintf(stderr, "\nline %d: %s\n", __LINE__, cudaGetErrorString(cudaStatus) );
				break;
			}
		}
		if (cudaStatus != cudaSuccess)
		{

			printf("\nfeature extracting is not successfully");
			continue;
		}
		float sum0 = 0;
		for (size_t i = 0; i < 1; i++)
		{
			printf("\n\n\n\n");
			sum0 = 0;
			for (size_t j = 0; j < SPAM_SYM_COUNT * 2; j++)
			{
				sum0 += hists[i][j];
				printf(" %d", hists[i][j]);
			}
			printf("\n\n\n\n");
			sum0 /= 4.0f;
			for (size_t j = 0; j < SPAM_SYM_COUNT * 2; j++)
			{
				if (sum0 > 0)
				printf(" %f", hists[i][j] / sum0);
			}

		}
		/*for (size_t i = 38; i < 39; i++)
		{
			printf("\n\n\n\n");
			for (size_t j = 0; j < SPAM_SYM_COUNT * 2; j++)
			{
				printf(" %d", hists[i][j]);
			}

		}
		printf("\n\n", sum);*/
		
		/*for (size_t i = 0; i < STREAM_COUNT; i++)
		{

		cudaEventElapsedTime(&elapsedTime,
		start[i], stop[i]);
		std::fprintf(stderr, "\nline %d: \nGPU Elapsed Time is %f Second", elapsedTime/1000, t2);
		}*/
		std::fprintf(stderr, "\nCPU Elapsed Time is %f Seconds\n", t2);
		if (save_features(str_file_path, class_id, hists))
		{
			printf("Saving in file is not successfully \n");
		}
	}
	cudaFree(dev_src);
	cudaFree(dev_MINMAXsymmCoord);
	cudaFree(dev_SPAMsymmCoord);
	for (int i = 0; i < HIST_COUNT; i++)
	{
		cudaStatus = cudaFree(residuals[i]);
	}
	cudaStatus = cudaFree(dev_residuals);
	//cudaFree(dev_residuals);
	for (size_t i = 0; i < HIST_COUNT; i++)
	{
		cudaFree(host_features.hists[i]);
	}
	for (size_t i = 0; i < COUNT_OF_SUBMODELS; i++)
	{
		delete hists[i];
	}
	for (size_t i = 0; i < KERNELS_COUNT; i++)
	{
		cudaFreeHost(host_src[i]);
	}
	for (size_t i = 0; i < STREAM_COUNT + 5; i++)
	{
		cudaStreamDestroy(streams[i]);
	}
	fclose(fp_list);
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(unsigned int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
	unsigned int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "\nline %d: cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", __LINE__);
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::fprintf(stderr, "\nline %d: cudaMalloc failed!", __LINE__);
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::fprintf(stderr, "\nline %d: cudaMalloc failed!", __LINE__);
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::fprintf(stderr, "\nline %d: cudaMalloc failed!", __LINE__);
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::fprintf(stderr, "\nline %d: cudaMemcpy failed!", __LINE__);
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::fprintf(stderr, "\nline %d: cudaMemcpy failed!", __LINE__);
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //PreFeature<<<1, 2>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::fprintf(stderr, "\nline %d: addKernel launch failed: %s\n", __LINE__, cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "\nline %d: cudaDeviceSynchronize returned error code %d after launching addKernel!\n", __LINE__, cudaStatus);
        goto Error;
    }
	
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::fprintf(stderr, "\nline %d: cudaMemcpy failed!", __LINE__);
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
