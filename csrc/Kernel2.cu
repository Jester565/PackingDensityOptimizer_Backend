#include "Kernel2.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>
#include <iostream>
#include <time.h>

#define BLOCK_SIZE2 32
#define BLOCK_SIZE_MAX 2048


__device__ void createLine(TangentPoint* tp, float pos, bool horz, bool max) {
	tp->y = pos;
	tp->x = horz;
	tp->r = max;
}

__global__ void initRun(TangentPoint* devTangentPoints, CircType* devCircleTypes, CircType* devConstCircleTypes, int* devNumCircleTypes, float w, float h, int numCircTypes) {

	devNumCircleTypes[0] = numCircTypes;
	for (int i = 0; i < numCircTypes; i++) {
		devCircleTypes[i] = devConstCircleTypes[i];
	}

	//left
	createLine(&devTangentPoints[0], 0, false, false);
	//top
	createLine(&devTangentPoints[1], 0, true, false);
	//right
	createLine(&devTangentPoints[2], w, false, true);
	//bottom
	createLine(&devTangentPoints[3], h, true, true);
}

__device__ void getMaxBenefit(const TangentPoint* devTangentPoints, int tpointsSize, TangentPointResult& target, const int i, const int j, const bool side) {
	const TangentPoint& p1 = devTangentPoints[i];
	const TangentPoint& p2 = devTangentPoints[j];
	//If p1.r and p2.r is greater than 0, both are circles
	if (i > 3 && j > 3) {
		float l = sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
		if (l - p1.r - p2.r > target.r * 2) {
			target.b = -FLT_MAX;
			return;
		}
		float d = (pow(l, 2) + (p1.r - p2.r) * (2 * target.r + p1.r + p2.r)) / (2 * l);
		float h = sqrt(pow(target.r + p1.r, 2) - pow(d, 2));
		float deg = atan2f(p2.y - p1.y, p2.x - p1.x);
		float xd = cosf(deg) * d + p1.x;
		float yd = sinf(deg) * d + p1.y;
		if (side) {
			target.x = xd + h * cos(deg - M_PI / 2.0f);
			target.y = yd + h * sin(deg - M_PI / 2.0f);
		}
		else {
			target.x = xd + h * cos(deg + M_PI / 2.0f);
			target.y = yd + h * sin(deg + M_PI / 2.0f);
		}
	}
	//One is a line and one is a circle
	else if ((i > 3 && j <= 3) || (j > 3 && i <= 3)) {
		const TangentPoint* line = &p1;
		const TangentPoint* circle = &p2;
		if (j <= 3) {
			line = &p2;
			circle = &p1;
		}
		bool horizontal = (line->x != 0);
		bool max = (line->r != 0);
		if (horizontal) {
			float yDif = abs(circle->y - line->y) - target.r;
			if (yDif > circle->r + target.r || yDif < 0) {
				target.b = -FLT_MAX;
				return;
			}
			float xDif = sqrt(pow(circle->r + target.r, 2) - pow(yDif, 2));
			float y = line->y + target.r;
			if (max) {
				y = line->y - target.r;
			}
			target.y = y;
			if (side) {
				target.x = circle->x + xDif;
			}
			else {
				target.x = circle->x - xDif;
			}
		}
		else {
			float xDif = abs(circle->x - line->y) - target.r;
			if (xDif > circle->r + target.r || xDif < 0) {
				target.b = -FLT_MAX;
				return;
			}
			float yDif = sqrt(pow(circle->r + target.r, 2) - pow(xDif, 2));
			float x = line->y + target.r;
			if (max) {
				x = line->y - target.r;
			}
			target.x = x;
			if (side) {
				target.y = circle->y + yDif;
			}
			else {
				target.y = circle->y - yDif;
			}
		}
	}
	//Both are lines
	else {
		//Get line horizontal and max values
		bool horizontal1 = (p1.x != 0);
		bool horizontal2 = (p2.x != 0);
		bool max1 = (p1.r != 0);
		bool max2 = (p2.r != 0);
		//If lines are in same direction, return
		if (horizontal1 == horizontal2) {
			target.b = -FLT_MAX;
			return;
		}
		float x = p1.y;
		float y = p2.y;
		float dx = (!max1) * 2 - 1;
		float dy = (!max2) * 2 - 1;
		if (horizontal1) {
			x = p2.y;
			y = p1.y;
			dx = (!max2) * 2 - 1;
			dy = (!max1) * 2 - 1;
		}
		target.x = x + (dx * target.r);
		target.y = y + (dy * target.r);
	}
	float minDist = FLT_MAX;
	for (int k = 0; k < 4; k++) {
		if (k != i && k != j) {
			const TangentPoint& line = devTangentPoints[k];
			bool horizontal = (line.x != 0);
			bool max = (line.r != 0);
			float dist = FLT_MAX;
			if (horizontal) {
				dist = line.y - target.y;
			}
			else {
				dist = line.y - target.x;
			}
			if (max) {
				if (dist < target.r) {
					target.b = -FLT_MAX;
					return;
				}
			}
			else {
				if (dist > -target.r) {
					target.b = -FLT_MAX;
					return;
				}
				dist *= -1;
			}
			//Only take distance from lines into account for the first two circles
			if (tpointsSize <= 5) {
				if (dist < minDist) {
					minDist = dist;
				}
			}
		}
	}
	for (int k = 4; k < tpointsSize; k++) {
		if (k != i && k != j) {
			const TangentPoint& circle = devTangentPoints[k];
			float dist = sqrt(pow(target.x - circle.x, 2) + pow(target.y - circle.y, 2));
			if (dist < target.r + circle.r) {
				target.b = -FLT_MAX;
				return;
			}
			if (dist < minDist) {
				minDist = dist;
			}
		}
	}
	target.b = 1 - minDist / target.r;
}

__device__ int devGetNumThreads(int numTangentPoints, int numCircleTypes) {
	return (numTangentPoints - 1) * (numTangentPoints)* numCircleTypes;
}

__global__ void genTangentPointResults(const TangentPoint* devTangentPoints, const CircType* devCircleTypes, TangentPointResult* devResults, const int* devNumCircleTypes, int numTangentPoints, int numCircleTypes) {
	__shared__ TangentPointResult sharedResults[BLOCK_SIZE2];
	TangentPointResult& target = sharedResults[threadIdx.x];
	int numThreads = devGetNumThreads(numTangentPoints, numCircleTypes);
	int threadI = (blockDim.x * blockIdx.x + threadIdx.x);
	if (threadI < numThreads) {
		//printf("ThreadI: %i\n", threadI);
		int tsSize = (numThreads / numCircleTypes);
		//printf("TSize: %i\n", tSize);
		int circleI = (threadI / tsSize);
		if (circleI < *devNumCircleTypes) {
			bool side = (threadI % tsSize) % 2;
			int tIdx = (threadI % tsSize) / 2;
			int tSize = tsSize / 2;
			//printf("TIdx: %i , numTangentPoints: %i \n", tIdx, numTangentPoints);
			int i = (1 - sqrt((float)(1 + 8 * (tSize - tIdx)))) / 2 + numTangentPoints - 1;
			int iIdx = tSize - ((numTangentPoints - i)*(numTangentPoints - i - 1)) / 2;
			int j = numTangentPoints - 1 - (tIdx - iIdx);
			if (i > 0 && j == 0) {
				j = numTangentPoints - 1;
			}
			//printf("i: %i , j: %i \n", i, j);
			target.r = devCircleTypes[circleI].r;
			getMaxBenefit(devTangentPoints, numTangentPoints, target, i, j, side);
		}
		else {
			//printf("Skipped %i", threadI);
			target.b = -FLT_MAX;
		}
	}
	else {
		//printf("Skipped %i", threadI);
		target.b = -FLT_MAX;
	}
	__syncthreads();
	int k = BLOCK_SIZE2 / 2;
	while (k != 0) {
		if (threadIdx.x < k) {
			if (sharedResults[threadIdx.x].b < sharedResults[threadIdx.x + k].b) {
				sharedResults[threadIdx.x] = sharedResults[threadIdx.x + k];
			}
		}
		__syncthreads();
		//divide by 2
		k /= 2;
	}
	if (threadIdx.x == 0) {
		devResults[blockIdx.x] = sharedResults[0];
	}
}

__global__ void updateTangentPoints(TangentPointResult* devResults, TangentPoint* devTangentPoints, CircType* devCircleTypes, int* devNumCircleTypes, int tangentPointSize, int numResultBlocks) {
	__shared__ TangentPointResult sharedResults[1024];
	int stride = blockDim.x;
	int idx = threadIdx.x;

	TangentPointResult maxTpr;
	maxTpr.b = -FLT_MAX;
	while (idx < numResultBlocks) {
		if (devResults[idx].b > maxTpr.b) {
			maxTpr = devResults[idx];
		}
		idx += stride;
	}
	sharedResults[threadIdx.x] = maxTpr;
	//printf("SR: %f () %f\n", sharedResults[threadIdx.x].b, sharedResults[threadIdx.x].r);
	int k = blockDim.x / 2;
	__syncthreads();
	//printf("TI: %f () %f\n", sharedResults[threadIdx.x].b, sharedResults[threadIdx.x].r);
	while (k != 0) {
		if (threadIdx.x < k) {
			if (sharedResults[threadIdx.x].b < sharedResults[threadIdx.x + k].b) {
				//printf("Replaced %f with %f : %i\n", sharedResults[threadIdx.x].b, sharedResults[threadIdx.x + k].b, threadIdx.x + k);
				sharedResults[threadIdx.x] = sharedResults[threadIdx.x + k];
			}
		}
		__syncthreads();
		k /= 2;
	}
	if (threadIdx.x == 0) {
		maxTpr = sharedResults[0];
		//printf("MaxTpr: %f : %f\n", maxTpr.b, maxTpr.r);
		if (maxTpr.b == -FLT_MAX) {
			*devNumCircleTypes = -1;
			return;
		}
		//printf("This R: %f\n", maxTpr.r);
		devTangentPoints[tangentPointSize].x = maxTpr.x;
		devTangentPoints[tangentPointSize].y = maxTpr.y;
		devTangentPoints[tangentPointSize].r = maxTpr.r;
		//printf("Num CircTypes: %i\n", *devNumCircleTypes);
		for (int i = 0; i < *devNumCircleTypes; i++) {
			//printf("R: %f\n", devCircleTypes[i].r);
			if (devCircleTypes[i].r == maxTpr.r) {
				devCircleTypes[i].count--;
				if (devCircleTypes[i].count == 0) {
					for (int j = i + 1; j < *devNumCircleTypes; j++) {
						devCircleTypes[i - 1] = devCircleTypes[i];
					}
					(*devNumCircleTypes)--;
				}
				return;
			}
		}
		printf("UPDATE TANGENT POINTS SHOULD NEVER GET HERE!");
	}
}

int getNumThreads(int numTangentPoints, int numCircleTypes) {
	return (numTangentPoints - 1) * (numTangentPoints)* numCircleTypes;
}

Kernel2Data* initK2(const std::map<float, int>& circleTypes) {
	Kernel2Data* k2Data = new Kernel2Data();
	k2Data->maxTangentPoints = 4;
	k2Data->numConstCircleTypes = circleTypes.size();
	k2Data->numCircleTypes = k2Data->numConstCircleTypes;
	CircType* cTypeArr = new CircType[k2Data->numCircleTypes];
	int i = 0;
	for (auto it = circleTypes.begin(); it != circleTypes.end(); it++) {
		k2Data->maxTangentPoints += it->second;
		cTypeArr[i].r = it->first;
		cTypeArr[i].count = it->second;
		i++;
	}
	int maxResultBlocks = (int)ceilf((float)getNumThreads(k2Data->maxTangentPoints, k2Data->numCircleTypes) / BLOCK_SIZE2);
	cudaMalloc((void**)&k2Data->devTangentPoints, k2Data->maxTangentPoints * sizeof(TangentPoint));
	cudaMalloc((void**)&k2Data->devResults, maxResultBlocks * sizeof(TangentPointResult));
	cudaMalloc((void**)&k2Data->devNumCircleTypes, sizeof(int));
	cudaMalloc((void**)&k2Data->devCircleTypes, k2Data->numCircleTypes * sizeof(CircType));
	cudaMalloc((void**)&k2Data->devConstCircleTypes, k2Data->numCircleTypes * sizeof(CircType));
	cudaMemcpy(k2Data->devConstCircleTypes, cTypeArr, k2Data->numCircleTypes * sizeof(CircType), cudaMemcpyHostToDevice);

	k2Data->tangentPoints = new TangentPoint[k2Data->maxTangentPoints];
	delete[] cTypeArr;
	return k2Data;
}

RunResult runK2(Kernel2Data* k2Data, float w, float h) {
	k2Data->numCircleTypes = k2Data->numConstCircleTypes;
	initRun << <1, 1 >> >(k2Data->devTangentPoints, k2Data->devCircleTypes, k2Data->devConstCircleTypes, k2Data->devNumCircleTypes, w, h, k2Data->numCircleTypes);
	//cudaDeviceSynchronize();

	clock_t begin = clock();
	int i = 4;

	for (; i < k2Data->maxTangentPoints; i++) {
		int numBlocks = ceilf((float)getNumThreads(i, k2Data->numCircleTypes) / BLOCK_SIZE2);
		//int numBlocks = ceilf((float)getNumThreads(i) / BLOCK_SIZE2);
		//printf("NumBlocks: %i\n", numBlocks);
		genTangentPointResults << <numBlocks, BLOCK_SIZE2 >> >(k2Data->devTangentPoints, k2Data->devCircleTypes, k2Data->devResults, k2Data->devNumCircleTypes, i, k2Data->numCircleTypes);
		int numThreads = 0;
		for (int i = 0; i <= 10; i++) {
			numThreads = pow(2, i);
			if (numThreads >= numBlocks) {
				break;
			}
		}
		//cudaDeviceSynchronize();
		//printf("NumThreads: %i\n", numThreads);
		//cudaDeviceSynchronize();
		//printf("NumCirc: %i\n", numCircleTypes);
		updateTangentPoints << <1, numThreads >> >(k2Data->devResults, k2Data->devTangentPoints, k2Data->devCircleTypes, k2Data->devNumCircleTypes, i, numBlocks);
		//cudaDeviceSynchronize();
		//cudaDeviceSynchronize();
		//printf("circTypes: %i", numCircleTypes);
		/*
		if (i % 10) {
		cudaMemcpy(&numCircleTypes, devNumCircleTypes, sizeof(int), cudaMemcpyDeviceToHost);
		}
		if (numCircleTypes <= 0) {
		if (numCircleTypes == 0) {
		i++;
		}
		break;
		}
		*/
	}
	cudaDeviceSynchronize();
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "ElapsedSecs: " << elapsed_secs << std::endl;
	cudaMemcpy(&k2Data->numCircleTypes, k2Data->devNumCircleTypes, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(k2Data->tangentPoints, k2Data->devTangentPoints, i * sizeof(TangentPoint), cudaMemcpyDeviceToHost);
	RunResult result;
	result.tangentPoints = k2Data->tangentPoints;
	result.size = i;
	result.circlesFit = (k2Data->numCircleTypes == 0);
	return result;
}

void freeK2(Kernel2Data* k2Data) {
	cudaFree(k2Data->devTangentPoints);
	cudaFree(k2Data->devResults);
	cudaFree(k2Data->devCircleTypes);
	cudaFree(k2Data->devConstCircleTypes);
	delete k2Data;
}