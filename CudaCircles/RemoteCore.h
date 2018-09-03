#pragma once
#include <aws/sqs/SQSClient.h>
#include <aws/dynamodb/DynamoDBClient.h>

class Solution;
class Kernel2Data;

class RemoteCore
{
 public:
  static std::string QUEUE_URL;
  static std::string TABLE_NAME;
  static const int NO_MSG_LIMIT = 60;

  RemoteCore();

  void run();

  void putInDBCB(const Aws::DynamoDB::DynamoDBClient*, const Aws::DynamoDB::Model::PutItemRequest&, const Aws::DynamoDB::Model::PutItemOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&);

  std::shared_ptr<Solution> getMatchCountSolution(const std::map<float, int>& circPercents, float precision, float w, float h);

  std::shared_ptr<Solution> getSolution(const std::map<float, int>& circTypes, float precision, float hRatio);

  ~RemoteCore();

 private:
  int noMsgCount;
  bool putInDb(const std::string& msgID, const std::map<float, int>& circCounts, const std::map<float, std::string>& extras, float precision, float packingDensity, const std::string& resp, int msgNum);
  bool receiveMsg();
  void handleMsg(const Aws::SQS::Model::Message& msg);
  Kernel2Data * matchCountSolution(const std::map<float, int>& circPercents, float w, float h, float countScale, float deltaCountScale, float precision);
  void narrowToSolution(Kernel2Data * k2Data, float w, float deltaW, float hRatio, float precision);
  Aws::SQS::SQSClient* sqsClient;
  Aws::DynamoDB::DynamoDBClient* dbClient;
};
