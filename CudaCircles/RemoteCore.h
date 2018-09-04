#pragma once
#include <aws/sqs/SQSClient.h>
#include <aws/dynamodb/DynamoDBClient.h>
#include "json.hpp"

class SolutionTwo;

class SolutionThree;

namespace two {
  class Kernel2Data;
}

namespace three {
  class Kernel3Data;
}

class RemoteCore
{
 public:
  static std::string QUEUE_URL;
  static std::string TABLE_NAME;
  static const int NO_MSG_LIMIT = 60;

  RemoteCore();

  void run();

  void putInDBCB(const Aws::DynamoDB::DynamoDBClient*, const Aws::DynamoDB::Model::PutItemRequest&, const Aws::DynamoDB::Model::PutItemOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&);

  std::shared_ptr<SolutionTwo> getSolutionTwo(const std::map<float, int>& circTypes, float precision, float hRatio);
  std::shared_ptr<SolutionThree> getSolutionThree(const std::map<float, int>& circTypes, float precision, float hRatio);

  ~RemoteCore();

 private:
  int noMsgCount;
  bool putInDb(const std::string& msgID, const std::map<float, int>& circCounts, const std::map<float, std::string>& extras, float precision, float packingDensity, const std::string& resp, int msgNum);
  bool receiveMsg();
  void handleMsg(const Aws::SQS::Model::Message& msg);
  void handleTwo(const nlohmann::json& bodyJson);
  void handleThree(const nlohmann::json& bodyJson);
  void narrowToSolutionTwo(two::Kernel2Data * k2Data, float w, float deltaW, float hRatio, float precision);
  void narrowToSolutionThree(three::Kernel3Data * k3Data, float w, float deltaW, float precision);
  Aws::SQS::SQSClient* sqsClient;
  Aws::DynamoDB::DynamoDBClient* dbClient;
};
