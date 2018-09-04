#pragma once
#include <map>

namespace two {
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
    int maxTangentPoints;
    TangentPointResult* devResults;
    CircType* devCircleTypes;
    CircType* devConstCircleTypes;
    int* devNumCircleTypes;
    int* devNumCircleTypesTemp;
    TangentPoint* tangentPoints;
    int* devMutex;
    int* devLastBlockId;
    TangentPointResult* devMaxResult;
    TangentPointResult* devMaxResultTemp;
    int numCircleTypes;
  };

  struct RunResult {
    TangentPoint* tangentPoints;
    int size;
    bool circlesFit;
  };

  Kernel2Data* initK2(const std::map<float, int>& circleTypes);

  RunResult runK2(Kernel2Data* k2Data, float w, float h);

  void freeK2(Kernel2Data*);
}