#include "RemoteCore.h"
#include <aws/sqs/model/ReceiveMessageRequest.h>
#include <aws/sqs/model/SendMessageRequest.h>
#include <aws/sqs/model/DeleteMessageRequest.h>
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

std::string RemoteCore::QUEUE_URL = "https://sqs.us-west-2.amazonaws.com/387396130957/circle.fifo";

RemoteCore::RemoteCore()
{
	Aws::Client::ClientConfiguration clientCfg;
	clientCfg.requestTimeoutMs = 600001;
	clientCfg.region = Aws::Region::US_WEST_2;
	sqsClient = new Aws::SQS::SQSClient(clientCfg);
}

void RemoteCore::run() {
	while (true) {
		receiveMsg();
	}
}

void RemoteCore::receiveMsg() {
	Aws::SQS::Model::ReceiveMessageRequest req;
	req.SetQueueUrl(QUEUE_URL);
	req.SetMaxNumberOfMessages(1);
	req.SetWaitTimeSeconds(20);
	
	auto result = sqsClient->ReceiveMessage(req);
	if (result.IsSuccess()) {
		if (result.GetResult().GetMessages().size() > 0) {
			std::cout << "Message aquired" << std::endl;
			handleMsg(result.GetResult().GetMessages().at(0));
		}
	}
	else {
		std::cerr << "Failed to get sqs msg " << result.GetError().GetMessage() << std::endl;
	}
}

void RemoteCore::handleMsg(const Aws::SQS::Model::Message& msg) {
	auto bodyJson = nlohmann::json::parse(msg.GetBody().c_str());
	auto circTypesJson = bodyJson["circTypes"];

	std::map<float, int> circTypes;
	for (auto it = circTypesJson.begin(); it != circTypesJson.end(); it++) {
		float radius = (*it)["radius"];
		int count = (*it)["count"];
		circTypes.insert(std::make_pair(radius, count));
	}
	float precision = bodyJson["precision"];
	std::shared_ptr<Solution> solution = getSolution(circTypes, precision);
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
	
	std::string respQueueUrl = bodyJson["queueUrl"];
	Aws::SQS::Model::SendMessageRequest sendReq;
	sendReq.SetQueueUrl(respQueueUrl);
	sendReq.SetMessageBody(respBodyJson.dump());

	auto sendResult = sqsClient->SendMessage(sendReq);
	if (sendResult.IsSuccess()) {
		std::cout << "Sent to queue " << respQueueUrl << std::endl;
		Aws::SQS::Model::DeleteMessageRequest deleteReq;
		deleteReq.SetQueueUrl(QUEUE_URL);
		deleteReq.SetReceiptHandle(msg.GetReceiptHandle());
		auto deleteResult = sqsClient->DeleteMessage(deleteReq);
		if (deleteResult.IsSuccess()) {
			std::cout << "Deleted message successfully" << std::endl;
		}
	}
}

void RemoteCore::narrowToSolution(Kernel2Data* k2Data, float w, float deltaW, float precision) {
	int direction = 0;
	while (true) {
		RunResult result = runK2(k2Data, w, w);

		if (deltaW >= 0.00001 || !result.circlesFit) {
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

std::shared_ptr<Solution> RemoteCore::getSolution(const std::map<float, int>& circTypes, float precision) {
	Kernel2Data* k2Data = initK2(circTypes);
	float w = 500.0f;
	float deltaW = 0.1f;
	narrowToSolution(k2Data, w, deltaW, precision);
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

RemoteCore::~RemoteCore()
{
	delete sqsClient;
	sqsClient = nullptr;
}
