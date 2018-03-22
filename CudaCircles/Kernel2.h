#pragma once
#include <map>
#include <cuda_runtime.h>
#include "CircleType.h"
#include <memory>


struct TangentPoint {
	float x;
	float y;
	float r;
};

struct TangentPointResult {
	float x;
	float y;
	float r;
	float b;
};

struct CircType {
	float r;
	int count;
};

struct Kernel2Data {
	TangentPoint* devTangentPoints;
	TangentPointResult* devResults;
	CircType* devCircleTypes;
	CircType* devConstCircleTypes;
	int* devNumCircleTypes;
	TangentPoint* tangentPoints;
	int maxTangentPoints;
	int numCircleTypes;
	int numConstCircleTypes;
};

struct RunResult {
	TangentPoint* tangentPoints;
	int size;
	bool circlesFit;
};

Kernel2Data* initK2(const std::map<float, int>& circleTypes);

RunResult runK2(Kernel2Data* k2Data, float w, float h);

void freeK2(Kernel2Data* k2Data);