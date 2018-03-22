#pragma once
#include <aws/sqs/SQSClient.h>

class Solution;
class Kernel2Data;

class RemoteCore
{
public:
	static std::string QUEUE_URL;

	RemoteCore();

	void run();

	std::shared_ptr<Solution> getSolution(const std::map<float, int>& circTypes, float precision);

	~RemoteCore();

private:
	void receiveMsg();
	void handleMsg(const Aws::SQS::Model::Message& msg);
	void narrowToSolution(Kernel2Data * k2Data, float w, float deltaW, float precision);
	Aws::SQS::SQSClient* sqsClient;
};

