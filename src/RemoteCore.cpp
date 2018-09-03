#include "RemoteCore.h"
#include <aws/sqs/model/ReceiveMessageRequest.h>
#include <aws/sqs/model/SendMessageRequest.h>
#include <aws/sqs/model/DeleteMessageRequest.h>
#include <aws/dynamodb/model/AttributeDefinition.h>
#include <aws/dynamodb/model/PutItemRequest.h>
#include <aws/dynamodb/model/PutItemResult.h>
#include "json.hpp"
#include "Kernel2.h"
#define _USE_MATH_DEFINES
#include <math.h>

struct Circle {
  Circle(float x, float y, float r)
    :x(x), y(y), r(r)
  {
  }
  float x;
  float y;
  float r;
};

struct Solution {
  float w;
  float h;
  float density;
  std::vector<Circle> circles;
};

std::string RemoteCore::TABLE_NAME = "Circle";
std::string RemoteCore::QUEUE_URL = "https://sqs.us-west-2.amazonaws.com/387396130957/circle.fifo";

RemoteCore::RemoteCore()
{
  Aws::Client::ClientConfiguration clientCfg;
  clientCfg.requestTimeoutMs = 600001;
  clientCfg.region = Aws::Region::US_WEST_2;
  sqsClient = new Aws::SQS::SQSClient(clientCfg);
  dbClient = new Aws::DynamoDB::DynamoDBClient(clientCfg);
  noMsgCount = 0;
}

void RemoteCore::run() {
  while (noMsgCount < NO_MSG_LIMIT) {
    if (receiveMsg()) {
      noMsgCount = 0;
    }
    else {
      noMsgCount++;
    }
  }
}

void RemoteCore::putInDBCB(const Aws::DynamoDB::DynamoDBClient *, const Aws::DynamoDB::Model::PutItemRequest &, const Aws::DynamoDB::Model::PutItemOutcome & outcome, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&)
{
  std::cout << "PUT IN DB CALLED" << std::endl;
  if (!outcome.IsSuccess()) {
    std::cout << outcome.GetError().GetMessage() << std::endl;
  }
}

bool RemoteCore::putInDb(const std::string& msgID, const std::map<float, int>& circCounts, const std::map<float, std::string>& extras, float precision, float packingDensity, const std::string& resp, int msgNum)
{
  std::string countID = "";
  std::string extraID = "";
  for (auto it = circCounts.begin(); it != circCounts.end(); it++) {
    countID += std::to_string(it->first) + ":" + std::to_string(it->second) + ";";
    auto extraIt = extras.find(it->first);
    if (extraIt != extras.end()) {
      extraID += std::to_string(extraIt->first) + ":" + extraIt->second + ";";
    }
  }
  Aws::DynamoDB::Model::PutItemRequest pir;
  pir.SetTableName(TABLE_NAME.c_str());

  Aws::DynamoDB::Model::AttributeValue cid;
  cid.SetS(countID.c_str());
  pir.AddItem(Aws::String("countID"), cid);
  if (extraID.size() > 0) {
    Aws::DynamoDB::Model::AttributeValue eid;
    eid.SetS(extraID.c_str());
    pir.AddItem(Aws::String("extraID"), eid);
  }
  Aws::DynamoDB::Model::AttributeValue mid;
  mid.SetS(msgID.c_str());
  pir.AddItem(Aws::String("msgID"), mid);
  Aws::DynamoDB::Model::AttributeValue prec;
  prec.SetN(std::to_string(precision).c_str());
  pir.AddItem(Aws::String("precision"), prec);
  Aws::DynamoDB::Model::AttributeValue pd;
  pd.SetN(std::to_string(packingDensity).c_str());
  pir.AddItem(Aws::String("packingDensity"), pd);
  Aws::DynamoDB::Model::AttributeValue r;
  r.SetS(resp.c_str());
  pir.AddItem(Aws::String("resp"), r);
  Aws::DynamoDB::Model::AttributeValue mn;
  mn.SetN(std::to_string(msgNum).c_str());
  pir.AddItem(Aws::String("msgNum"), mn);
  dbClient->PutItemAsync(pir, std::bind(&RemoteCore::putInDBCB, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
}

bool RemoteCore::receiveMsg() {
  Aws::SQS::Model::ReceiveMessageRequest req;
  req.SetQueueUrl(QUEUE_URL.c_str());
  req.SetMaxNumberOfMessages(1);
  req.SetWaitTimeSeconds(20);

  auto result = sqsClient->ReceiveMessage(req);
  if (result.IsSuccess()) {
    if (result.GetResult().GetMessages().size() > 0) {
      std::cout << "Message aquired" << std::endl;
      handleMsg(result.GetResult().GetMessages().at(0));
      return true;
    }
  }
  else {
    std::cerr << "Failed to get sqs msg " << result.GetError().GetMessage() << std::endl;
    return false;
  }
}

void RemoteCore::handleMsg(const Aws::SQS::Model::Message& msg) {
  auto bodyJson = nlohmann::json::parse(msg.GetBody().c_str());
  auto circTypesJson = bodyJson["circTypes"];

  std::map<float, int> circTypes;
  std::map<float, std::string> circExtras;
  for (auto it = circTypesJson.begin(); it != circTypesJson.end(); it++) {
    float radius = (*it)["radius"];
    int count = (*it)["count"];
    auto extraIt = (*it).find("extras");
    circTypes.insert(std::make_pair(radius, count));
    if (extraIt != (*it).end()) {
      std::string extra = (*it)["extras"];
      circExtras.insert(std::make_pair(radius, extra));
    }
  }
  float precision = bodyJson["precision"];
  float hRatio = 1;
  if (bodyJson.find("hRatio") != bodyJson.end()) {
    hRatio = bodyJson["hRatio"];
  }
  int msgNum = 0;
  if (bodyJson.find("msgNum") != bodyJson.end()) {
    msgNum = bodyJson["msgNum"];
  }
  std::string msgId = bodyJson["msgId"];
  std::shared_ptr<Solution> solution = nullptr;
  if (bodyJson["type"] == "matchArea") {
    solution = getSolution(circTypes, precision, hRatio);
  }
  else if (bodyJson["type"] == "matchCount") {
    float w = bodyJson["w"];
    float h = bodyJson["h"];
    solution = getMatchCountSolution(circTypes, 0.0001, w, h);
  }
  auto respBodyJson = nlohmann::json::object();
  respBodyJson["w"] = solution->w;
  respBodyJson["h"] = solution->h;
  respBodyJson["density"] = solution->density;
  auto circleArrJson = nlohmann::json::array();
  for (auto it = solution->circles.begin(); it != solution->circles.end(); it++) {
    auto circleJson = nlohmann::json::object();
    circleJson["x"] = it->x;
    circleJson["y"] = it->y;
    circleJson["r"] = it->r;
    circleArrJson.push_back(circleJson);
  }
  respBodyJson["circleArr"] = circleArrJson;
  respBodyJson["msgId"] = msgId;

  std::string respQueueUrl = bodyJson["queueUrl"];
  Aws::SQS::Model::SendMessageRequest sendReq;
  sendReq.SetQueueUrl(respQueueUrl.c_str());
  std::string respBody = respBodyJson.dump();
  sendReq.SetMessageBody(respBody.c_str());
  putInDb(msgId, circTypes, circExtras, precision, solution->density, respBody, msgNum);

  auto sendResult = sqsClient->SendMessage(sendReq);
  if (sendResult.IsSuccess()) {
    std::cout << "Sent to queue " << respQueueUrl << std::endl;
    Aws::SQS::Model::DeleteMessageRequest deleteReq;
    deleteReq.SetQueueUrl(QUEUE_URL.c_str());
    deleteReq.SetReceiptHandle(msg.GetReceiptHandle());
    auto deleteResult = sqsClient->DeleteMessage(deleteReq);
    if (deleteResult.IsSuccess()) {
      std::cout << "Deleted message successfully" << std::endl;
    }
  }
  else {
    std::cout << "ERROR: " << sendResult.GetError().GetMessage() << std::endl;
  }
}


Kernel2Data* RemoteCore::matchCountSolution(const std::map<float, int>& circPercents, float w, float h, float countScale, float deltaCountScale, float precision) {
  int direction = 0;
  while (true) {
    std::map<float, int> circCounts;
    for (auto it = circPercents.begin(); it != circPercents.end(); it++) {
      circCounts.insert(std::make_pair(it->first, it->second * countScale));
    }
    Kernel2Data* k2Data = initK2(circCounts);
    RunResult result = runK2(k2Data, w, h);

    if (deltaCountScale >= 0.0001 || !result.circlesFit) {
      if (result.circlesFit) {
	if (direction >= 0) {
	  countScale *= (1 + deltaCountScale);
	  direction = 1;
	}
	else {
	  deltaCountScale /= 2.0f;
	  countScale *= (1 + deltaCountScale);
	  direction = 1;
	}
      }
      else {
	if (direction <= 0) {
	  countScale *= (1 - deltaCountScale);
	  direction = -1;
	}
	else {
	  deltaCountScale /= 2.0f;
	  countScale *= (1 - deltaCountScale);
	  direction = -1;
	}
      }
    }
    else {
      return k2Data;
    }
    freeK2(k2Data);
  }
}

void RemoteCore::narrowToSolution(Kernel2Data* k2Data, float w, float deltaW, float hRatio, float precision) {
  int direction = 0;
  while (true) {
    std::cout << "W: " << w << "H: " << w * hRatio << std::endl;
    RunResult result = runK2(k2Data, w, w * hRatio);
    if (result.circlesFit) {
      for (int i = 0; i < result.size; i++) {
	//std::cout << "(" << result.tangentPoints[i].x << ", " << result.tangentPoints[i].y << "): " << result.tangentPoints[i].r << std::endl;
      }
    }
    if (deltaW >= precision || !result.circlesFit) {
      if (result.circlesFit) {
	if (direction <= 0) {
	  w *= (1 - deltaW);
	  direction = -1;
	}
	else {
	  deltaW /= 2.0f;
	  w *= (1 - deltaW);
	  direction = -1;
	}
      }
      else {
	if (direction >= 0) {
	  w *= (1 + deltaW);
	  direction = 1;
	}
	else {
	  deltaW /= 2.0f;
	  w *= (1 + deltaW);
	  direction = 1;
	}
      }
    }
    else {
      return;
    }
  }
}


std::shared_ptr<Solution> RemoteCore::getMatchCountSolution(const std::map<float, int>& circPercents, float precision, float w, float h) {
  float countScale = 1.0f;
  float deltaCountScale = 0.1f;
  Kernel2Data* k2Data = matchCountSolution(circPercents, w, h, countScale, deltaCountScale, precision);
  auto solution = std::make_shared<Solution>();

  float minX = 0;
  float maxX = 0;
  float minY = 0;
  float maxY = 0;
  for (int i = 0; i < 4; i++) {
    TangentPoint& tPoint = k2Data->tangentPoints[i];
    if (tPoint.x) {
      if (tPoint.r) {
	maxX = tPoint.y;
      }
      else {
	minX = tPoint.y;
      }
    }
    else {
      if (tPoint.r) {
	maxY = tPoint.y;
      }
      else {
	minY = tPoint.y;
      }
    }
  }
  solution->w = maxX - minX;
  solution->h = maxY - minY;
  float rectArea = solution->w * solution->h;

  solution->circles.reserve(k2Data->maxTangentPoints - 4);

  float circleArea = 0.0f;
  for (int i = 4; i < k2Data->maxTangentPoints; i++) {
    TangentPoint& tPoint = k2Data->tangentPoints[i];
    solution->circles.emplace_back(tPoint.x, tPoint.y, tPoint.r);
    circleArea += M_PI * tPoint.r * tPoint.r;
  }
  solution->density = circleArea / rectArea;

  freeK2(k2Data);

  return solution;
}

std::shared_ptr<Solution> RemoteCore::getSolution(const std::map<float, int>& circTypes, float precision, float hRatio) {
  Kernel2Data* k2Data = initK2(circTypes);
  float totalArea = 0;
  for (auto it = circTypes.begin(); it != circTypes.end(); it++) {
    totalArea += M_PI * it->first * it->first * it->second;
  }
  float w = (sqrt(totalArea)/0.84f)/hRatio;
  float deltaW = 0.05f;
  narrowToSolution(k2Data, w, deltaW, hRatio, precision);
  auto solution = std::make_shared<Solution>();

  float minX = 0;
  float maxX = 0;
  float minY = 0;
  float maxY = 0;
  for (int i = 0; i < 4; i++) {
    TangentPoint& tPoint = k2Data->tangentPoints[i];
    if (tPoint.x) {
      if (tPoint.r) {
	maxX = tPoint.y;
      }
      else {
	minX = tPoint.y;
      }
    }
    else {
      if (tPoint.r) {
	maxY = tPoint.y;
      }
      else {
	minY = tPoint.y;
      }
    }
  }
  solution->w = maxY - minY;
  solution->h = maxX - minX;
  float rectArea = solution->w * solution->h;

  solution->circles.reserve(k2Data->maxTangentPoints - 4);

  float circleArea = 0.0f;
  for (int i = 4; i < k2Data->maxTangentPoints; i++) {
    TangentPoint& tPoint = k2Data->tangentPoints[i];
    solution->circles.emplace_back(tPoint.x, tPoint.y, tPoint.r);
    //std::cout << "(" << tPoint.x << ", " << tPoint.y << "): " << tPoint.r << std::endl;
    circleArea += M_PI * tPoint.r * tPoint.r;
  }
  solution->density = circleArea / rectArea;

  freeK2(k2Data);

  return solution;
}

RemoteCore::~RemoteCore()
{
  delete sqsClient;
  sqsClient = nullptr;
}
