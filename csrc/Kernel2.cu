#include "Kernel2.h"
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

#define NUM_SIDES 2
#define BLOCK_SIZE2 32
#define BLOCK_SIZE_MAX 1024 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ void getMaxBenefit(const TangentPoint* devTangibles, int tpointsSize, TangentPointResult& target, const int i, const int j, const bool side) {
	const TangentPoint& p1 = devTangibles[i];
	const TangentPoint& p2 = devTangibles[j];
	//Since i and j are greater than 3 both tangibles are circles
	if (i > 3 && j > 3) {
		float l = sqrtf(powf(p2.x - p1.x, 2) + powf(p2.y - p1.y, 2));
		//Circles must be close enough so target can touch both of them
		if (l - p1.r - p2.r > target.r * 2) {
			target.b = -FLT_MAX;
			return;
		}

		//Calculates distance from p2 to the point on the line made between p1 and p2 tangent to the tangentPoint
		float d = (pow(l, 2) + (p1.r - p2.r) * (2 * target.r + p1.r + p2.r)) / (2 * l);
		//Height of tangentPoint above or below line made between p1 and p2
		float h = sqrt(pow(target.r + p1.r, 2) - pow(d, 2));
		float deg = atan2f(p2.y - p1.y, p2.x - p1.x);
		//Calculate x and y coordinates of tangent points
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
	//One element is a circle and one is a line
	else if ((i > 3 && j <= 3) || (j > 3 && i <= 3)) {
		//Figure out which tangible is the line and which the circle
		const TangentPoint* line = &p1;
		const TangentPoint* circle = &p2;
		if (j <= 3) {
			line = &p2;
			circle = &p1;
		}
		//Is the line horizontal?
		bool horizontal = (line->x != 0);
		//Is the line upper or lower bounding?
		bool max = (line->r != 0);
		if (horizontal) {
			//Distance between circle and line
			float yDif = abs(circle->y - line->y) - target.r;
			//Won't work if the circle is too far from the line
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
}

__device__ void createLine(TangentPoint* tp, float pos, bool horz, bool max) {
	tp->y = pos;
	tp->x = horz;
	tp->r = max;
}

__device__ int devGetNumThreads(int numTangentPoints, int numCircleTypes) {
	return (numTangentPoints - 1) * (numTangentPoints)* numCircleTypes;
}

//Initializes tangibles with the first lines
__global__ void initRun(TangentPoint* devTangentPoints, CircType* devCircleTypes, CircType* devConstCircleTypes, int* devNumCircleTypes, float w, float h, int numCircTypes) {
	//printf("Init run started");
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
	//printf("Init run done!");
}

__global__ void getDist(const TangentPoint* devTangibles, const int tangibleSize, const int i, const int j, TangentPointResult& target, int* devMutex) {
	__shared__ float dists[BLOCK_SIZE_MAX];
	__shared__ bool collided;
	if (threadIdx.x == 0) {
		collided = false;
	}
	__syncthreads();
	dists[threadIdx.x] = FLT_MAX;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < tangibleSize && idx != i && idx != j) {
		if (idx < 4) {
			const TangentPoint& line = devTangibles[idx];
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
					collided = true;
				}
			}
			else {
				if (dist > -target.r) {
					collided = true;
				}
				dist *= -1;
			}
			if (tangibleSize < 5) {
				dists[threadIdx.x] = dist;
			}
		}
		else {
			const TangentPoint& circle = devTangibles[idx];
			//distance between tangentPoint and circle
			float dist = sqrt(pow(target.x - circle.x, 2) + pow(target.y - circle.y, 2));
			//check if tangent point collides with circle
			if (dist < target.r + circle.r) {
				collided = true;
			}
			dists[threadIdx.x] = dist;
		}
	}
	__syncthreads();
	if (!collided) {
		int k = blockDim.x / 2;
		while (k != 0) {
			if (threadIdx.x < k) {
				if (dists[threadIdx.x + k] < dists[threadIdx.x]) {
					// && (threadIdx.x + k) < tangibleSize
					dists[threadIdx.x] = dists[threadIdx.x + k];
					//printf("K: %i\n", k);
				}
			}
			__syncthreads();
			//divide by 2
			k /= 2;
		}
		if (threadIdx.x == 0) {
			/*
			float benefit = 1 - dists[0] / target.r;
			while (atomicCAS(devMutex, 0, 1) != 0);
			if (benefit > target.b && target.b != -FLT_MAX) {
			target.b = benefit;
			//printf("BENEFIT: %f", benefit);
			}
			atomicExch(devMutex, 0);
			*/
			target.b = 1 - dists[0] / target.r;
		}
	}
	else if (threadIdx.x == 0) {
		/*
		while (atomicCAS(devMutex, 0, 1) != 0);
		target.b = -FLT_MAX;
		atomicExch(devMutex, 0);
		*/
		target.b = -FLT_MAX;
	}
}
/*
__global__ void getDist(const Tangible* devTangibles, const int tangibleSize, const int distsSize, const int i, const int j, TangentPoint& target, int* devMutex) {
float minDist = FLT_MAX;
for (int k = 0; k < 4; k++) {
if (k != i && k != j) {
const Tangible& line = devTangibles[k];
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
//Don't take line to line into account after placing the first circle
if (tangibleSize < 5) {
if (dist < minDist) {
minDist = dist;
}
}
}
}
for (int k = 4; k < tangibleSize; k++) {
if (k != i && k != j) {
const Tangible& circle = devTangibles[k];
//distance between tangentPoint and circle
float dist = sqrt(pow(target.x - circle.x, 2) + pow(target.y - circle.y, 2));
//check if tangent point collides with circle
if (dist < target.r + circle.r) {
target.b = -FLT_MAX;
return;
}
if (dist < minDist) {
minDist = dist;
}
}
}

//minDist is guaranteed to be set
target.b = 1 - minDist / target.r;
}
*/

//Initialize the first tangentPoints at the corners
__global__ void initResults(TangentPoint* devTangibles, const CircType* devCircleTypes, TangentPointResult* devResults, TangentPointResult* devMaxResult, int numTangentPoints, int numConstCircleTypes, int totalNumTangentPoints, int* devMutex) {
	int numThreads = devGetNumThreads(numTangentPoints, numConstCircleTypes);
	int threadI = (blockDim.x * blockIdx.x + threadIdx.x);
	if (threadI >= numThreads) {
		return;
	}
	int tIdx = threadI / (numConstCircleTypes * 2);
	int circleI = (threadI / 2) % numConstCircleTypes;
	bool side = threadI % 2;
	int tSize = numThreads / (2 * numConstCircleTypes);
	//printf("TIdx: %i , numTangentPoints: %i \n", tIdx, numTangentPoints);
	int i = (1 - sqrt((double)(1 + 8 * (tSize - tIdx)))) / 2 + numTangentPoints - 1;
	int iIdx = tSize - ((numTangentPoints - i)*(numTangentPoints - i - 1)) / 2;
	int j = (tIdx - iIdx) + i + 1;
	//printf("i: %i, j: %i\n", i, j);
	//Modified to account for extra columns
	int rIdx = threadI + i * (totalNumTangentPoints - numTangentPoints) * (2 * numConstCircleTypes);
	//printf("rIdx: %i\n", rIdx);
	TangentPointResult& target = devResults[rIdx];
	//printf("i: %i , j: %i \n", i, j);
	target.r = devCircleTypes[circleI].r;
	target.b = 0;
	getMaxBenefit(devTangibles, numTangentPoints, target, i, j, side);
	if (target.b != -FLT_MAX) {
		target.b = -FLT_MAX / 2.0f;
		getDist << <1, BLOCK_SIZE_MAX >> > (devTangibles, numTangentPoints, i, j, target, devMutex);
	}
	if (threadIdx.x == 0) {
		cudaDeviceSynchronize();
		devMaxResult->x = target.x;
		devMaxResult->y = target.y;
		devMaxResult->r = target.r;
		devMaxResult->b = target.b;
		devTangibles[numTangentPoints].r = devMaxResult->r;
		devTangibles[numTangentPoints].x = devMaxResult->x;
		devTangibles[numTangentPoints].y = devMaxResult->y;
	}
}

//Runs for every tangentPoint * numCircleTypes * 2
__global__ void placeCircle(TangentPoint* devTangibles, CircType* devCircleTypes, TangentPointResult* devResults, TangentPointResult* devMaxResult, TangentPointResult* devMaxResultTemp, int* devNumCircleTypes, int numConstCircleTypes, int numTangentPoints, int totalNumTangentPoints, int* devLastBlockId, int* devNumCircleTypesTemp, int* devMutex) {
	//Do not run if we are out of circles or failed to place a circle
	if (*devNumCircleTypes < 0) {
		devNumCircleTypesTemp[0] = devNumCircleTypes[0];
		return;
	}
	//Total number of threads processing
	int numThreads = (numTangentPoints) * 2 * numConstCircleTypes;
	//Size of the results array
	int numResults = devGetNumThreads(totalNumTangentPoints, numConstCircleTypes);
	//The index of this thread
	int threadI = (blockDim.x * blockIdx.x + threadIdx.x);
	//If this threadI exceeds the size of the data, it has nothing to process
	if (threadI >= numThreads) {
		return;
	}
	//Index of the first tangible in pair
	int i = threadI / (numConstCircleTypes * 2);
	//Index of the circleType
	int circleI = (threadI / 2) % numConstCircleTypes;
	//Get the side of the circle
	bool side = threadI % 2;
	//Index of the second tangible in pair
	int j = numTangentPoints;
	//Index of first element in the results containing tangible i
	int iIdx = numResults / (2 * numConstCircleTypes) - ((totalNumTangentPoints - i)*(totalNumTangentPoints - i - 1)) / 2;
	//Accounts for index offset
	int off = threadI % (numConstCircleTypes * 2);
	//Index of the result
	int rIdx = (iIdx + (j - i - 1)) * numConstCircleTypes * 2 + off;
	TangentPointResult& target = devResults[rIdx];
	target.r = devCircleTypes[circleI].r;
	target.b = 0;
	getMaxBenefit(devTangibles, numTangentPoints, target, i, j, side);
	if (target.b != -FLT_MAX) {
		int distNumBlocks = ceilf(((float)numTangentPoints) / BLOCK_SIZE_MAX);
		unsigned int distNumThreads = min(numTangentPoints, BLOCK_SIZE_MAX);
		target.b = -FLT_MAX / 2;
		/*
		//Computes the next highest power of two
		distNumThreads--;
		distNumThreads |= numThreads >> 1;
		distNumThreads |= distNumThreads >> 2;
		distNumThreads |= distNumThreads >> 4;
		distNumThreads |= distNumThreads >> 8;
		distNumThreads |= distNumThreads >> 16;
		distNumThreads++;
		if (distNumBlocks > 1) {
		printf("MORE NUM BLOCKS");
		}
		*/
		getDist << <1, BLOCK_SIZE_MAX >> > (devTangibles, numTangentPoints, i, j, target, devMutex);
	}
	//printf("TargetR: %f, %f, %f\n", target.x, target.y, target.b);
	//Ensures only one thread runs this
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		//printf("-\n");
		//printf("CYCLE END\n");
		*devMaxResultTemp = *devMaxResult;
		//printf("TDNCT: %i : %i\n", *tempDevNumCircleTypes, *devNumCircleTypes);
		devMaxResult->b = -FLT_MAX;
		*devLastBlockId = -1;
		*devNumCircleTypesTemp = *devNumCircleTypes;
		for (int i = 0; i < numConstCircleTypes; i++) {
			if (devCircleTypes[i].r == devMaxResult->r) {
				devCircleTypes[i].count--;
				return;
			}
		}
		printf("SHOULD NEVER REACH HERE!");
	}
}


//runs on everyTangentPoint including the recently generated ones
__global__ void setMaxResult(TangentPoint* devTangibles, CircType* devCircleTypes, TangentPointResult* devResults, TangentPointResult* devMaxResult, TangentPointResult* devMaxResultTemp, int* devMutex, int* devNumCircleTypes, int numConstCircleTypes, int numTangentPoints, int totalNumTangentPoints, int* devLastBlockId, int* devNumCircleTypesTemp) {
	if (*devNumCircleTypesTemp < 0) {
		return;
	}
	/*
	if (threadIdx.x == 0 && blockIdx.x == 0) {
	printf("*\n");
	}
	*/
	//printf("MaxResultCalled\n");
	__shared__ TangentPointResult sharedResults[BLOCK_SIZE_MAX];
	sharedResults[threadIdx.x].b = -FLT_MAX;
	int numThreads = devGetNumThreads(numTangentPoints, numConstCircleTypes);
	int threadI = (blockDim.x * blockIdx.x + threadIdx.x);

	if (threadI < numThreads) {
		//should be i?
		int tIdx = threadI / (2 * numConstCircleTypes);
		int circleI = (threadI / 2) % numConstCircleTypes;
		bool side = threadI % 2;
		int tSize = numThreads / (2 * numConstCircleTypes);
		//printf("TIdx: %i , numTangentPoints: %i \n", tIdx, numTangentPoints);
		int i = (1.0 - sqrt((double)(1 + 8 * (tSize - tIdx)))) / 2.0 + numTangentPoints - 1;
		int iIdx = tSize - ((numTangentPoints - i)*(numTangentPoints - i - 1)) / 2;
		int j = (tIdx - iIdx) + i + 1;
		int rIdx = threadI + i * (totalNumTangentPoints - numTangentPoints) * (2 * numConstCircleTypes);
		TangentPointResult& target = devResults[rIdx];
		//printf("Is %i, %i, %i\n", i, j, rIdx);
		if (devCircleTypes[circleI].count > 0) {
			if (j < numTangentPoints - 1 && target.b != -FLT_MAX) {
				//Use tempMaxResult so the operation at the end of this function does not overwrite existing value
				float dist = sqrt(pow(target.x - devMaxResultTemp->x, 2) + pow(target.y - devMaxResultTemp->y, 2));
				//printf("Checking Dist for: (%i, %i), (%f, %f, %f): %f  ...  (%f, %f, %f)\n", i, j, target.x, target.y, target.r, dist, devTempMaxResult->x, devTempMaxResult->y, devTempMaxResult->r);
				//printf("X: %f, Y: %f\n", devTempMaxResult->x, devTempMaxResult->y, devTempMaxResult->r);
				if (dist < target.r + devMaxResultTemp->r) {
					target.b = -FLT_MAX;
				}
				else {
					float benefit = 1.0f - dist / target.r;
					if (benefit > target.b) {
						target.b = benefit;
					}
				}
			}
			sharedResults[threadIdx.x].x = target.x;
			sharedResults[threadIdx.x].y = target.y;
			sharedResults[threadIdx.x].r = target.r;
			sharedResults[threadIdx.x].b = target.b;
		}
		else {
			target.b = -FLT_MAX;
		}
		//printf("(%i, %i, %i): %f\n", i, j, rIdx, sharedResults[threadIdx.x].b);
	}
	__syncthreads();
	int k = BLOCK_SIZE_MAX / 2;
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
		//printf("ThreadMax: %f, (%f, %f)\n", sharedResults[0].b, sharedResults[0].x, sharedResults[0].y);
		while (atomicCAS(devMutex, 0, 1) != 0);
		//printf("BENEFIT: %f vs %f\n", devMaxResult->b, sharedResults[0].b);
		if (devMaxResult->b < sharedResults[0].b || (devMaxResult->b == sharedResults[0].b && blockIdx.x > *devLastBlockId)) {
			//printf("Max: replaced %f with %f\n", devMaxResult->b, sharedResults[0].b);
			devMaxResult->x = sharedResults[0].x;
			devMaxResult->y = sharedResults[0].y;
			devMaxResult->r = sharedResults[0].r;
			devMaxResult->b = sharedResults[0].b;
			*devLastBlockId = blockIdx.x;
			devTangibles[numTangentPoints].r = devMaxResult->r;
			devTangibles[numTangentPoints].x = devMaxResult->x;
			devTangibles[numTangentPoints].y = devMaxResult->y;
			//printf("GRA: %f, %i / %i\n", sharedResults[0].b, blockIdx.x, gridDim.x);
		}
		//printf("BENEFIT: %f\n", sharedResults[0].b);
		if (devMaxResult->b == -FLT_MAX) {
			//printf("Set to -1\n");
			*devNumCircleTypes = -1;
		}
		else {
			//printf("Set to 1\n");
			*devNumCircleTypes = 1;
		}
		atomicExch(devMutex, 0);
	}
}

int getNumThreads(int numTangentPoints, int numCircleTypes) {
	return (numTangentPoints - 1) * (numTangentPoints)* numCircleTypes;
}

Kernel2Data* initK2(const std::map<float, int>& circleTypes) {
	Kernel2Data* k2Data = new Kernel2Data();
	k2Data->maxTangentPoints = 4;
	int numConstCircleTypes = circleTypes.size();
	k2Data->numCircleTypes = circleTypes.size();
	//k2Data->devNumCircleTypes = numConstCircleTypes;
	//k2Data->devNumCircleTypesTemp = numConstCircleTypes;
	CircType* cTypeArr = new CircType[numConstCircleTypes];
	int i = 0;
	for (auto it = circleTypes.begin(); it != circleTypes.end(); it++) {
		k2Data->maxTangentPoints += it->second;
		cTypeArr[i].r = it->first;
		cTypeArr[i].count = it->second;
		i++;
	}
	//printf("DevR Size: %i\n", getNumThreads(maxTangentPoints));
	cudaMalloc((void**)&k2Data->devTangentPoints, k2Data->maxTangentPoints * sizeof(TangentPoint));
	cudaMalloc((void**)&k2Data->devResults, getNumThreads(k2Data->maxTangentPoints, k2Data->numCircleTypes) * sizeof(TangentPointResult));
	cudaMalloc((void**)&k2Data->devNumCircleTypes, sizeof(int));
	cudaMalloc((void**)&k2Data->devMaxResult, sizeof(TangentPointResult));
	cudaMalloc((void**)&k2Data->devMaxResultTemp, sizeof(TangentPointResult));
	cudaMalloc((void**)&k2Data->devNumCircleTypes, sizeof(int));
	cudaMalloc((void**)&k2Data->devNumCircleTypesTemp, sizeof(int));
	cudaMalloc((void**)&k2Data->devCircleTypes, k2Data->numCircleTypes * sizeof(CircType));
	cudaMalloc((void**)&k2Data->devConstCircleTypes, k2Data->numCircleTypes * sizeof(CircType));
	cudaMemcpy(k2Data->devConstCircleTypes, cTypeArr, k2Data->numCircleTypes * sizeof(CircType), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&k2Data->devMutex, sizeof(int));
	cudaMemset(k2Data->devMutex, 0, sizeof(int));
	cudaMalloc((void**)&k2Data->devLastBlockId, sizeof(int));

	k2Data->tangentPoints = new TangentPoint[k2Data->maxTangentPoints];

	delete[] cTypeArr;
	return k2Data;
}

RunResult runK2(Kernel2Data* k2Data, float w, float h) {
	//printf("CYCLE\n");
	initRun << <1, 1 >> >(k2Data->devTangentPoints, k2Data->devCircleTypes, k2Data->devConstCircleTypes, k2Data->devNumCircleTypes, w, h, k2Data->numCircleTypes);
	int numInitialResults = getNumThreads(4, k2Data->numCircleTypes);
	initResults << <1, numInitialResults >> >(k2Data->devTangentPoints, k2Data->devCircleTypes, k2Data->devResults, k2Data->devMaxResult, 4, k2Data->numCircleTypes, k2Data->maxTangentPoints, k2Data->devMutex);
	cudaDeviceSynchronize();

	clock_t begin = clock();
	int i = 4;

	for (; i < k2Data->maxTangentPoints; i++) {
		int numThreads = i * k2Data->numCircleTypes * 2;
		int numBlocks = ceilf((float)numThreads / BLOCK_SIZE2);
		if (numBlocks > 1) {
			numThreads = BLOCK_SIZE2;
		}
		cudaDeviceSynchronize();
		placeCircle << <numBlocks, numThreads >> >(k2Data->devTangentPoints, k2Data->devCircleTypes, k2Data->devResults, k2Data->devMaxResult, k2Data->devMaxResultTemp, k2Data->devNumCircleTypes, k2Data->numCircleTypes, i, k2Data->maxTangentPoints, k2Data->devLastBlockId, k2Data->devNumCircleTypesTemp, k2Data->devMutex);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		//printf("NumThreads: %i\n", numThreads);
		//cudaDeviceSynchronize();
		//printf("NumCirc: %i\n", numCircleTypes);
		if (i != k2Data->maxTangentPoints - 1) {
			numThreads = getNumThreads(i + 1, k2Data->numCircleTypes);
			numBlocks = ceilf((float)numThreads / BLOCK_SIZE_MAX);
			cudaDeviceSynchronize();
			setMaxResult << <numBlocks, BLOCK_SIZE_MAX >> > (k2Data->devTangentPoints, k2Data->devCircleTypes, k2Data->devResults, k2Data->devMaxResult, k2Data->devMaxResultTemp, k2Data->devMutex, k2Data->devNumCircleTypes, k2Data->numCircleTypes, i + 1, k2Data->maxTangentPoints, k2Data->devLastBlockId, k2Data->devNumCircleTypesTemp);
		}
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
	int numCircTypes = 0;
	RunResult result;
	cudaMemcpy(&numCircTypes, k2Data->devNumCircleTypes, sizeof(int), cudaMemcpyDeviceToHost);
	result.tangentPoints = nullptr;
	result.circlesFit = (numCircTypes >= 0);
	if (result.circlesFit) {
		//TangentPoint* tangibles = new TangentPoint[i];
		cudaMemcpy(k2Data->tangentPoints, k2Data->devTangentPoints, i * sizeof(TangentPoint), cudaMemcpyDeviceToHost);
		result.tangentPoints = k2Data->tangentPoints;
	}
	result.size = i;
	return result;
}

void freeK2(Kernel2Data* k2Data) {
	cudaFree(k2Data->devTangentPoints);
	cudaFree(k2Data->devResults);
	cudaFree(k2Data->devNumCircleTypes);
	cudaFree(k2Data->devNumCircleTypesTemp);
	cudaFree(k2Data->devMaxResultTemp);
	cudaFree(k2Data->devMaxResult);
	cudaFree(k2Data->devMutex);
	cudaFree(k2Data->devLastBlockId);

	delete[] k2Data->tangentPoints;
}