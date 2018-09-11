# PackingDensityOptimizer_Backend
Performs packing density optimization for 3D printing particles using 2D and 3D simulations.  The 2D simulation are based on [this research paper]() 
but using CUDA which decreased the runtime by a factor of 70.  The 3D simulation is experimental and similar to the 2D algorithm.  However, it uses dynamic parallelism
to lower the amount of unused threads.
<br />
<br />
The requirements for and analysis of the 2D simulation was done by Don Kim and was a major part of the 
"2018 Annual International Conference Additive Manufacturing& Powder Metallurgy" presentation titled
"A Preliminary Study of Optimal Powder Size Distribution for Powder Bed Additive Manufacturing Using High Performance Cloud Computing".

## Building
Requirements: CUDA (5.0), AWS C++ SDK (1.6.9)
#### Requirement Installation (Ubuntu Only)
1. [CUDA Installation Guide](https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04)
2. [AWS C++ SDK](https://github.com/aws/aws-sdk-cpp#aws-sdk-cpp)

## Cloud infrastructure
You must link to a DynamoDB table using countID as a primary key.  Edit the [TABLE_NAME variable](https://github.com/Jester565/PackingDensityOptimizer_Backend/blob/master/src/RemoteCore.cpp#L49) to match your Cloud Infrastructure.
<br/>
<br/>
You must link to an SQS queue to read in simulation requests.  Edit the [QUEUE_URL variable](https://github.com/Jester565/PackingDensityOptimizer_Backend/blob/master/src/RemoteCore.cpp#L50) to point to your SQS queue.
<br/>
<br/>
Put your AWS credentials in your home directory [like so](https://docs.aws.amazon.com/cli/latest/userguide/cli-config-files.html).
#### Build
```
mkdir build
cd build
cmake ..
make
```
