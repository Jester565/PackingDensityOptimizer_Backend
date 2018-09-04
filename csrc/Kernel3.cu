#include "Kernel3.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#define _USE_MATH_DEFINES
#include <float.h>
#include <iostream>
#include <time.h>
#include <stdio.h>
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "math_constants.h"
#include "math_functions.h"
#include <math.h>
#include "VecHelper.h"

namespace three {
	#define NUM_SIDES 180
	#define NUM_SLOTS 181
	#define BLOCK_SIZE2 32
	#define BLOCK_SIZE_MAX 1024
	#define UNDEFINED_FLT -1000000.0f
	#define X_AXIS 0
	#define Y_AXIS 1
	#define Z_AXIS 2
	#define NUM_CORNERS 8
	#define MAX_DIST 20.0f

	struct FloatIntPair {
		float first;
		int second;
	};


	#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false)
	{
		if (code != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		}
	}

	__device__ void wallSpherePivotHelper(TangentPoint& target, float targetR, float wallCord, bool wallMax, float sphereCord, float sphereR) {
		float dif = abs(sphereCord - wallCord) - targetR;
		float h = sqrtf(powf(sphereR + targetR, 2) - powf(dif, 2));
		target.pos.x = wallCord + (!wallMax * 2 - 1) * targetR;
		target.pos.y = h;
	}

	__device__ void wallSpherePivotHelper2(float targetR, float wallCord, bool wallMax, float sphereCord, float sphereR, bool& inRange) {
		float dif = abs(sphereCord - wallCord) - targetR;
		//not sure about dif < 0
		if (dif > sphereR + targetR) {
			inRange = false;
		}
	}

	__device__ void getPivotInfo(const Tangible* devTangibles, int tpointsSize, const int i, const int j, float targetR, bool& inRange) {
		const Tangible& t1 = devTangibles[i];
		const Tangible& t2 = devTangibles[j];
		//j will always be a sphere and will, therefore, have an index greater than 3 hence the remove check
		if (i >= 6) {
			float l = sqrt(pow(t2.pos.x - t1.pos.x, 2) + pow(t2.pos.y - t1.pos.y, 2) + pow(t2.pos.z - t1.pos.z, 2));
			//Circles must be close enough so target can touch both of them
			if (l - t1.r - t2.r > targetR * 2) {
				inRange = false;
				return;
			}
		}
		//One element is a sphere and one is a wall
		else {
			const Tangible* wall = &t1;
			const Tangible* sphere = &t2;
			if (wall->pos.x == X_AXIS) {
				wallSpherePivotHelper2(targetR, wall->pos.y, (wall->r != 0), sphere->pos.x, sphere->r, inRange);
			}
			else if (wall->pos.x == Y_AXIS) {
				wallSpherePivotHelper2(targetR, wall->pos.y, (wall->r != 0), sphere->pos.y, sphere->r, inRange);
			}
			else if (wall->pos.x == Z_AXIS) {
				wallSpherePivotHelper2(targetR, wall->pos.y, (wall->r != 0), sphere->pos.z, sphere->r, inRange);
			}
		}
	}

	__device__ void getTangentPointFromPivotInfo(const Tangible* devTangibles, int tpointsSize, TangentPoint& target, float targetR, TangentPoint& pivotInfo, const int i, const int j, float circTheta) {
		const Tangible& t1 = devTangibles[i];
		const Tangible& t2 = devTangibles[j];
		//j will always be a sphere and will, therefore, have an index greater than 3 hence the remove check
		if (i >= 6) {
			float l = sqrt(pow(t2.pos.x - t1.pos.x, 2) + pow(t2.pos.y - t1.pos.y, 2) + pow(t2.pos.z - t1.pos.z, 2));
			//Circles must be close enough so target can touch both of them
			if (l - t1.r - t2.r > targetR * 2) {
				target.b = UNDEFINED_FLT;
				return;
			}
			float d = (pow(l, 2) + (t1.r - t2.r) * (2 * targetR + t1.r + t2.r)) / (2 * l);
			float h = sqrt(pow(targetR + t1.r, 2) - pow(d, 2));
			float3 unitVector = make_float3((t2.pos.x - t1.pos.x) / l, (t2.pos.y - t1.pos.y) / l, (t2.pos.z - t1.pos.z) / l);
			float3 betweenPoint = make_float3(unitVector.x * d + t1.pos.x, unitVector.y * d + t1.pos.y, unitVector.z * d + t1.pos.z);
			float3 perpVec1;
			if (unitVector.x != 0 && unitVector.y != 0 && unitVector.z != 0) {
				perpVec1 = make_float3(1, 1, (-unitVector.x - unitVector.y) / unitVector.z);
			}
			else {
				perpVec1 = make_float3((unitVector.x == 0), (unitVector.y == 0), (unitVector.z == 0));
			}
			perpVec1 = getNormalizedVec(perpVec1);
			float3 perpVec2 = getCrossProduct(perpVec1, unitVector);
			target.pos.x = betweenPoint.x + h * cos(circTheta) * perpVec1.x + h * sin(circTheta) * perpVec2.x;
			target.pos.y = betweenPoint.y + h * cos(circTheta) * perpVec1.y + h * sin(circTheta) * perpVec2.y;
			target.pos.z = betweenPoint.z + h * cos(circTheta) * perpVec1.z + h * sin(circTheta) * perpVec2.z;
		}
		//One element is a sphere and one is a wall
		else {
			const Tangible* wall = &t1;
			const Tangible* sphere = &t2;
			if (wall->pos.x == X_AXIS) {
				wallSpherePivotHelper(target, targetR, wall->pos.y, (wall->r != 0), sphere->pos.x, sphere->r);
				//TODO: Remove conditional
				if (target.b == UNDEFINED_FLT) {
					return;
				}
				target.pos.z = sphere->pos.z + target.pos.y * sinf(circTheta);
				target.pos.y = sphere->pos.y + target.pos.y * cosf(circTheta);
			}
			else if (wall->pos.x == Y_AXIS) {
				wallSpherePivotHelper(target, targetR, wall->pos.y, (wall->r != 0), sphere->pos.y, sphere->r);
				if (target.b == UNDEFINED_FLT) {
					return;
				}
				target.pos.z = sphere->pos.z + target.pos.y * sinf(circTheta);
				float tempX = sphere->pos.x + target.pos.y * cosf(circTheta);
				target.pos.y = target.pos.x;
				target.pos.x = tempX;
			}
			else if (wall->pos.x == Z_AXIS) {
				wallSpherePivotHelper(target, targetR, wall->pos.y, (wall->r != 0), sphere->pos.z, sphere->r);
				if (target.b == UNDEFINED_FLT) {
					return;
				}
				target.pos.z = target.pos.x;
				target.pos.x = sphere->pos.x + target.pos.y * cosf(circTheta);
				target.pos.y = sphere->pos.y + target.pos.y * sinf(circTheta);
			}
		}
		//No longer consider situation with two walls due to change in algorithm
	}

	__device__ int devGetNumThreads(int numTangentPoints, int numCircleTypes) {
		return ((numTangentPoints - 1) * (numTangentPoints)/2) * NUM_SLOTS * numCircleTypes;
	}

	__device__ void createSide(Tangible& tangible, float pos, int axisCode, bool max) {
		tangible.r = max;
		tangible.pos.x = axisCode;
		tangible.pos.y = pos;
	}

	//Initializes tangibles with the first lines
	__global__ void initRun(Tangible* devTangibles, CircType* devCircleTypes, CircType* devConstCircleTypes, int* devNumCircleTypes, int numConstCircleTypes, float3 cubeDim, TangentPointFull* devMaxResult, int* devNumResults) {
		devNumCircleTypes[0] = numConstCircleTypes;
		for (int i = 0; i < numConstCircleTypes; i++) {
			devCircleTypes[i] = devConstCircleTypes[i];
		}
		createSide(devTangibles[0], 0, X_AXIS, false);
		createSide(devTangibles[1], cubeDim.x, X_AXIS, true);
		createSide(devTangibles[2], 0, Y_AXIS, false);
		createSide(devTangibles[3], cubeDim.y, Y_AXIS, true);
		createSide(devTangibles[4], 0, Z_AXIS, false);
		createSide(devTangibles[5], cubeDim.z, Z_AXIS, true);

		devMaxResult->r = devCircleTypes[0].r;
		devMaxResult->pos.x = devMaxResult->r;
		devMaxResult->pos.y = devMaxResult->r;
		devMaxResult->pos.z = devMaxResult->r;
		devTangibles[6].r = devMaxResult->r;
		devTangibles[6].pos.x = devMaxResult->pos.x;
		devTangibles[6].pos.y = devMaxResult->pos.y;
		devTangibles[6].pos.z = devMaxResult->pos.z;
		*devNumResults = 0;
	}

	__global__ void getDist(const Tangible* devTangibles, const int tangiblesSize, const int i, const int j, TangentPoint& target, float targetR, int* devMutex) {
		extern __shared__ float dists[];
		__shared__ bool collided;
		if (threadIdx.x == 0) {
			collided = false;
		}
		__syncthreads();
		dists[threadIdx.x] = 0;
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < tangiblesSize && idx != i && idx != j) {
			if (idx < 6) {
				const Tangible& line = devTangibles[idx];
				bool max = (line.r != 0);
				float dist = FLT_MAX;
				if (line.pos.x == X_AXIS) {
					dist = line.pos.y - target.pos.x;
				}
				else if (line.pos.x == Y_AXIS) {
					dist = line.pos.y - target.pos.y;
				}
				else if (line.pos.x == Z_AXIS) {
					dist = line.pos.y - target.pos.z;
				}
				if (max) {
					if (dist < targetR) {
						collided = true;
					}
				}
				else if (dist > -targetR) {
					collided = true;
				}
				dists[threadIdx.x] = (targetR/4.0) / dist;
			}
			else {
				const Tangible& sphere = devTangibles[idx];
				//distance between tangentPoint and circle
				float dist = sqrt(pow(target.pos.x - sphere.pos.x, 2) + pow(target.pos.y - sphere.pos.y, 2) + pow(target.pos.z - sphere.pos.z, 2));
				//check if tangent point collides with circle
				if (dist < targetR + sphere.r) {
					collided = true;
				}
				dists[threadIdx.x] = targetR / dist;
			}
		}
		__syncthreads();
		if (!collided) {
			int k = blockDim.x / 2;
			while (k != 0) {
				if (threadIdx.x < k) {
					dists[threadIdx.x] += dists[threadIdx.x + k];
				}
				__syncthreads();
				//divide by 2
				k /= 2;
			}
			if (threadIdx.x == 0) {
				/*
				while (atomicCAS(devMutex, 0, 1) != 0);
				if (benefit > target.b && target.b != UNDEFINED_FLT) {
					target.b = benefit;
					//printf("BENEFIT: %f", benefit);
				}
				atomicExch(devMutex, 0);
				*/
				if (i < 6 && i > 0) {
					dists[0] -= 1;
				}
				target.b = dists[0];
			}
		}
		else if (threadIdx.x == 0) {
			//while (atomicCAS(devMutex, 0, 1) != 0);
			target.b = UNDEFINED_FLT;
			//atomicExch(devMutex, 0);
		}
	}

	__global__ void subPlaceCircle(int pivotIdx, Tangible* devTangibles, int tangiblesSize, int i, int j, TangentPoint* devResults, int* devMutex, float targetR) {
		TangentPoint& pivotInfo = devResults[pivotIdx];
		int threadI = blockDim.x * blockIdx.x + threadIdx.x;
		if (threadI >= NUM_SIDES) {
			return;
		}
		int targetIdx = pivotIdx + threadI + 1;
		TangentPoint& target = devResults[targetIdx];
		target.b = -FLT_MAX / 2;
		float circTheta = (((float)threadI) / NUM_SIDES) * 2 * M_PI;
		getTangentPointFromPivotInfo(devTangibles, tangiblesSize, target, targetR, pivotInfo, i, j, circTheta);
		//getMaxBenefit(devTangibles, numTangentPoints, target, i, j, side);
		int distNumBlocks = ceilf(((float)tangiblesSize) / BLOCK_SIZE_MAX);
		unsigned int distNumThreads = min(tangiblesSize, BLOCK_SIZE_MAX);
		//Computes the next highest power of two
		distNumThreads--;
		distNumThreads |= distNumThreads >> 1;
		distNumThreads |= distNumThreads >> 2;
		distNumThreads |= distNumThreads >> 4;
		distNumThreads |= distNumThreads >> 8;
		distNumThreads |= distNumThreads >> 16;
		distNumThreads++;
		//TODO: Add dynamic blocks back in
		//TODO: Don't use devMutex (maybe a float in the pivotInfo)
		target.b = -FLT_MAX / 2;
		getDist <<<distNumBlocks, distNumThreads, distNumThreads * sizeof(float)>>> (devTangibles, tangiblesSize, i, j, target, targetR, devMutex);
	}

	//Runs for every tangentPoint * numCircleTypes * 2
	__global__ void placeCircle(Tangible* devTangibles, int tangiblesSize, int totalTangiblesSize, CircType* devCircleTypes, int numConstCircleTypes, int* devNumCircleTypes, int* devNumCircleTypesTemp, TangentPoint* devResults, TangentPointFull* devMaxResult, TangentPointFull* devMaxResultTemp, int* devLastBlockId, int* devMutex, int* devResultsSize) {
		//Do not run if we are out of circles or failed to place a circle
		if (*devNumCircleTypes < 0) {
			devNumCircleTypesTemp[0] = devNumCircleTypes[0];
			return;
		}
		//Total number of threads processing
		int numThreads = tangiblesSize * numConstCircleTypes;
		//The index of this thread
		int threadI = (blockDim.x * blockIdx.x + threadIdx.x);
		//If this threadI exceeds the size of the data, it has nothing to process
		if (threadI >= numThreads) {
			return;
		}
		int i = threadI / numConstCircleTypes;
		int j = tangiblesSize;
		int circleI = threadI % numConstCircleTypes;

		float targetR = devCircleTypes[circleI].r;
		bool inRange = true;
		getPivotInfo(devTangibles, tangiblesSize, i, j, targetR, inRange);
		if (inRange) {
			int pivotIdx = atomicAdd(devResultsSize, NUM_SLOTS);
			TangentPoint& pivotInfo = devResults[pivotIdx];
			pivotInfo.pos.x = i;
			pivotInfo.pos.y = j;
			pivotInfo.pos.z = circleI;
			pivotInfo.b = 1;
			int spcNumBlocks = ceilf(((float)NUM_SIDES) / BLOCK_SIZE2);
			int spcNumThreads = min(NUM_SIDES, BLOCK_SIZE2);
			subPlaceCircle<<<spcNumBlocks, spcNumThreads>>> (pivotIdx, devTangibles, tangiblesSize, i, j, devResults, devMutex, targetR);
		}

		//Ensures only one thread runs this
		if (blockIdx.x == 0 && threadIdx.x == 0) {
			*devMaxResultTemp = *devMaxResult;
			devMaxResult->b = UNDEFINED_FLT;
			*devLastBlockId = -1;
			*devNumCircleTypesTemp = *devNumCircleTypes;
			for (int i = 0; i < numConstCircleTypes; i++) {
				if (devCircleTypes[i].r == devMaxResult->r) {
					devCircleTypes[i].count--;
					return;
				}
			}
			//printf("SHOULD NEVER REACH HERE!");
		}
	}

	//CANNOT HANDLE MULTIPLE THREAD BLOCKS, NUM_SIDES <= 1024
	__global__ void updateDists(int pivotIdx, Tangible* devTangibles, TangentPoint* devResults, TangentPointFull* devMaxResultTemp, bool updateDists, float targetR) {
		extern __shared__ FloatIntPair benefits[];
		TangentPoint& pivotInfo = devResults[pivotIdx];
		int threadI = blockDim.x * blockIdx.x + threadIdx.x;
		benefits[threadIdx.x].first = UNDEFINED_FLT;
		benefits[threadIdx.x].second = threadI;
		if (threadI >= NUM_SIDES) {
			return;
		}
		int targetIdx = pivotIdx + threadI + 1;
		TangentPoint& target = devResults[targetIdx];
		if (updateDists) {
			if (target.b != UNDEFINED_FLT) {
				float dist = sqrt(pow(target.pos.x - devMaxResultTemp->pos.x, 2) + pow(target.pos.y - devMaxResultTemp->pos.y, 2) + pow(target.pos.z - devMaxResultTemp->pos.z, 2));
				if (dist >= targetR + devMaxResultTemp->r) {
					target.b += targetR / dist;
					benefits[threadIdx.x].first = target.b;
				}
				else {
					target.b = UNDEFINED_FLT;
				}
			}
		}
		else {
			benefits[threadIdx.x].first = target.b;
		}
		__syncthreads();
		int k = blockDim.x / 2;
		while (k != 0) {
			if (threadIdx.x < k) {
				if (benefits[threadIdx.x + k].first > benefits[threadIdx.x].first) {
					benefits[threadIdx.x] = benefits[threadIdx.x + k];
				}
			}
			__syncthreads();
			//divide by 2
			k /= 2;
		}
		if (threadIdx.x == 0) {
			//printf("MaxBenefit: %i  -  %f\n", benefits[0].second + pivotIdx + 1, benefits[0].first);
			pivotInfo.b = benefits[0].first;
			pivotInfo.pos.x = benefits[0].second;
		}
	}

	//runs on everyTangentPoint including the recently generated ones
	__global__ void setMaxResult(Tangible* devTangibles, int tangiblesSize, int totalTangiblesSize, CircType* devCircleTypes, int* devNumCircleTypes, int* devNumCircleTypesTemp, int numConstCircleTypes, TangentPoint* devResults, TangentPointFull* devMaxResult, TangentPointFull* devMaxResultTemp, int* devMutex, int* devLastBlockId, int* devResultsSize) {
		if (*devNumCircleTypesTemp < 0) {
			return;
		}
		__shared__ TangentPointFull sharedResults[BLOCK_SIZE_MAX];
		sharedResults[threadIdx.x].b = UNDEFINED_FLT;
		int numThreads = (*devResultsSize)/NUM_SLOTS;
		int threadI = (blockDim.x * blockIdx.x + threadIdx.x);

		if (threadI < numThreads) {
			//should be i?
			int pivotIdx = threadI * NUM_SLOTS;
			TangentPoint& pivotInfo = devResults[pivotIdx];
			int i = (int)pivotInfo.pos.x;
			int j = (int)pivotInfo.pos.y;
			//printf("Is %i, %i, %i\n", i, j, rIdx);
			CircType& circType = devCircleTypes[(int)pivotInfo.pos.z];
			if (circType.count > 0) {
				if (pivotInfo.b != UNDEFINED_FLT) {
					//printf("Updating dists for pivotIdx: %i\n", pivotIdx);
					//Not necessary unless using multiple blocks
					//pivotInfo.b = -FLT_MAX;
					float targetR = circType.r;
					int spcNumThreads = NUM_SIDES;
					spcNumThreads--;
					spcNumThreads |= spcNumThreads >> 1;
					spcNumThreads |= spcNumThreads >> 2;
					spcNumThreads |= spcNumThreads >> 4;
					spcNumThreads |= spcNumThreads >> 8;
					spcNumThreads |= spcNumThreads >> 16;
					spcNumThreads++;
					updateDists<<<1, spcNumThreads, spcNumThreads * sizeof(FloatIntPair)>>>(pivotIdx, devTangibles, devResults, devMaxResultTemp, (j < tangiblesSize - 1), targetR);
					cudaDeviceSynchronize();
					if (pivotInfo.b != UNDEFINED_FLT) {
						TangentPoint& maxBenefitTarget = devResults[(int)pivotInfo.pos.x + pivotIdx + 1];  //hopefully float is accurate
						//printf("MaxBenefitTarget: %f  -  %i\n", maxBenefitTarget.b, (int)pivotInfo.pos.x + pivotIdx + 1);
						sharedResults[threadIdx.x].pos.x = maxBenefitTarget.pos.x;
						sharedResults[threadIdx.x].pos.y = maxBenefitTarget.pos.y;
						sharedResults[threadIdx.x].pos.z = maxBenefitTarget.pos.z;
						sharedResults[threadIdx.x].r = targetR;
						sharedResults[threadIdx.x].b = maxBenefitTarget.b;
					}
				}
			}
			else {
				pivotInfo.b = UNDEFINED_FLT;
			}
			//printf("(%i, %i, %i): %f\n", i, j, rIdx, sharedResults[threadIdx.x].b);
		}
		__syncthreads();
		int k = blockDim.x / 2;
		while (k != 0) {
			if (threadIdx.x < k) {
				if (sharedResults[threadIdx.x + k].b > sharedResults[threadIdx.x].b) {
					//printf("NewMax: replaced %f with %f\n", sharedResults[threadIdx.x].b, sharedResults[threadIdx.x + k].b);
					sharedResults[threadIdx.x] = sharedResults[threadIdx.x + k];
				}
			}
			__syncthreads();
			//divide by 2
			k /= 2;
		}
		if (threadIdx.x == 0) {
			while (atomicCAS(devMutex, 0, 1) != 0);
			if (devMaxResult->b < sharedResults[0].b || (devMaxResult->b == sharedResults[0].b && blockIdx.x > *devLastBlockId)) {
				devMaxResult->pos.x = sharedResults[0].pos.x;
				devMaxResult->pos.y = sharedResults[0].pos.y;
				devMaxResult->pos.z = sharedResults[0].pos.z;
				devMaxResult->r = sharedResults[0].r;
				devMaxResult->b = sharedResults[0].b;
				*devLastBlockId = blockIdx.x;
				//printf("RESULT: (%f, %f, %f)  -  %f\n", devMaxResult->pos.x, devMaxResult->pos.y, devMaxResult->pos.z, devMaxResult->r);
				devTangibles[tangiblesSize].r = devMaxResult->r;
				devTangibles[tangiblesSize].pos.x = devMaxResult->pos.x;
				devTangibles[tangiblesSize].pos.y = devMaxResult->pos.y;
				devTangibles[tangiblesSize].pos.z = devMaxResult->pos.z;
			}
			if (devMaxResult->b == UNDEFINED_FLT) {
				*devNumCircleTypes = -1;
			}
			else {
				*devNumCircleTypes = 1;
			}
			atomicExch(devMutex, 0);
		}
	}

	int getNumThreads(int numTangentPoints, int numCircleTypes) {
		return (((numTangentPoints - 1) * (numTangentPoints))/2) * numCircleTypes * NUM_SLOTS;
	}

	Kernel2Data* initK3(const std::map<float, int>& circleTypes) {
		auto k2Data = new Kernel2Data();
		k2Data->maxTangibles = 6;
		k2Data->numConstCircleTypes = circleTypes.size();
		k2Data->numCircleTypes = k2Data->numConstCircleTypes;
		CircType* cTypeArr = new CircType[k2Data->numCircleTypes];
		int i = 0;
		for (auto it = circleTypes.rbegin(); it != circleTypes.rend(); it++) {
			k2Data->maxTangibles += it->second;
			cTypeArr[i].r = it->first;
			cTypeArr[i].count = it->second;
			i++;
		}
		float3 cornerVecs[NUM_CORNERS];
		cornerVecs[0] = make_float3(0, 0, 0);
		cornerVecs[1] = make_float3(1, 0, 0);
		cornerVecs[2] = make_float3(0, 1, 0);
		cornerVecs[3] = make_float3(1, 1, 0);
		cornerVecs[4] = make_float3(1, 0, 1);
		cornerVecs[5] = make_float3(0, 1, 1);
		cornerVecs[6] = make_float3(1, 1, 1);
		cornerVecs[7] = make_float3(0, 0, 1);
		cudaMalloc((void**)&k2Data->devTangibles, k2Data->maxTangibles * sizeof(Tangible));
		//TODO: Minimize thread size
		cudaMalloc((void**)&k2Data->devResults, getNumThreads(k2Data->maxTangibles, k2Data->numCircleTypes) * sizeof(TangentPoint));
		cudaMalloc((void**)&k2Data->devMaxResult, sizeof(TangentPointFull));
		cudaMalloc((void**)&k2Data->devMaxResultTemp, sizeof(TangentPointFull));
		cudaMalloc((void**)&k2Data->devNumCircleTypes, sizeof(int));
		cudaMalloc((void**)&k2Data->devNumCircleTypesTemp, sizeof(int));
		cudaMalloc((void**)&k2Data->devCircleTypes, k2Data->numCircleTypes * sizeof(CircType));
		cudaMalloc((void**)&k2Data->devConstCircleTypes, k2Data->numCircleTypes * sizeof(CircType));
		cudaMemcpy(k2Data->devConstCircleTypes, cTypeArr, k2Data->numCircleTypes * sizeof(CircType), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&k2Data->devCornerVecs, NUM_CORNERS * sizeof(float3));
		cudaMemcpy(k2Data->devCornerVecs, cornerVecs, NUM_CORNERS * sizeof(float3), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&k2Data->devMutex, sizeof(int));
		cudaMemset(k2Data->devMutex, 0, sizeof(int));
		cudaMalloc((void**)&k2Data->devLastBlockId, sizeof(int));
		cudaMallocManaged((void**)&k2Data->devResultsSize, sizeof(int));

		k2Data->tangibles = new Tangible[k2Data->maxTangibles];

		delete[] cTypeArr;
		return k2Data;
	}

	RunResult runK3(Kernel2Data* k2Data, float3 cubeDim) {
		//printf("CUBE DIM: %f, %f, %f\n", cubeDim.x, cubeDim.y, cubeDim.z);
		k2Data->numCircleTypes = k2Data->numConstCircleTypes;
		initRun <<<1, 1>>>(k2Data->devTangibles, k2Data->devCircleTypes, k2Data->devConstCircleTypes, k2Data->devNumCircleTypes, k2Data->numConstCircleTypes, cubeDim, k2Data->devMaxResult, k2Data->devResultsSize);
		clock_t begin = clock();
		int i = 6;
		
		for (; i < k2Data->maxTangibles; i++) {
			int numThreads = i * k2Data->numCircleTypes;
			int numBlocks = ceilf((float)numThreads / BLOCK_SIZE2);
			if (numBlocks > 1) {
				numThreads = BLOCK_SIZE2;
			}
			placeCircle<<<numBlocks, numThreads>>>(k2Data->devTangibles, i, k2Data->maxTangibles, k2Data->devCircleTypes, k2Data->numConstCircleTypes, k2Data->devNumCircleTypes, k2Data->devNumCircleTypesTemp, k2Data->devResults, k2Data->devMaxResult, k2Data->devMaxResultTemp, k2Data->devLastBlockId, k2Data->devMutex, k2Data->devResultsSize);
			if (i != k2Data->maxTangibles - 1) {
				cudaDeviceSynchronize();
				numThreads = (*(k2Data->devResultsSize))/NUM_SLOTS;
				numBlocks = ceilf((float)numThreads / BLOCK_SIZE_MAX);
				setMaxResult <<<numBlocks, BLOCK_SIZE_MAX>>> (k2Data->devTangibles, i + 1, k2Data->maxTangibles, k2Data->devCircleTypes, k2Data->devNumCircleTypes, k2Data->devNumCircleTypesTemp, k2Data->numConstCircleTypes, k2Data->devResults, k2Data->devMaxResult, k2Data->devMaxResultTemp, k2Data->devMutex, k2Data->devLastBlockId, k2Data->devResultsSize);
			}
		}
		cudaDeviceSynchronize();
		clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		std::cout << "ElapsedSecs: " << elapsed_secs << std::endl;
		cudaMemcpy(&k2Data->numCircleTypes, k2Data->devNumCircleTypes, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(k2Data->tangibles, k2Data->devTangibles, i * sizeof(Tangible), cudaMemcpyDeviceToHost);
		RunResult result;
		result.tangibles = k2Data->tangibles;
		result.size = i;
		result.circlesFit = (k2Data->numCircleTypes >= 0);
		return result;
	}

	void freeK3(Kernel2Data* k2Data) {
		cudaFree(k2Data->devTangibles);
		cudaFree(k2Data->devResults);
		cudaFree(k2Data->devCircleTypes);
		cudaFree(k2Data->devConstCircleTypes);
		cudaFree(k2Data->devCornerVecs);
		cudaFree(k2Data->devMaxResult);
		cudaFree(k2Data->devMaxResultTemp);
		cudaFree(k2Data->devMutex);
		cudaFree(k2Data->devLastBlockId);
		delete k2Data;
	}
}