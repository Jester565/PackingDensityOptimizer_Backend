#pragma once
#include <map>
#include <vector_types.h>

namespace three {
	struct Tangible {
		float3 pos;
		float r;
	};

	struct RunResult {
		Tangible* tangibles;
		int size;
		bool circlesFit;
	};

	struct TangentPoint {
		float3 pos;
		float b;
	};

	struct TangentPointFull {
		float3 pos;
		float b;
		float r;
	};

	struct CircType {
		float r;
		int count;
	};


	struct Kernel3Data {
		Tangible* tangibles;
		Tangible* devTangibles;
		TangentPoint* devResults;
		CircType* devCircleTypes;
		CircType* devConstCircleTypes;
		int* devNumCircleTypes;
		int* devNumCircleTypesTemp;
		float3* devCornerVecs;
		TangentPointFull* devMaxResult;
		TangentPointFull* devMaxResultTemp;
		int maxTangibles;
		int numCircleTypes;
		int numConstCircleTypes;
		int* devMutex;
		int* devLastBlockId;
		int* devResultsSize;
	};

	Kernel2Data* initK3(const std::map<float, int>& circleTypes);

	RunResult runK3(Kernel2Data* k2Data, float3 cubeDim);

	void freeK3(Kernel2Data* k2Data);
}