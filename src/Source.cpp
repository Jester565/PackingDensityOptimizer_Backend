#include <iostream>
#include "RemoteCore.h"
#include <aws/core/Aws.h>

int main() {
	Aws::SDKOptions options;
	Aws::InitAPI(options);
	RemoteCore rc;
	rc.run();
}