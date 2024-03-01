# VFSTL

## Prerequisites
1. docker engine
2. nvidia-docker-toolkit
## How to run 
1. Change to the project directory 
2. Build the docker with `docker build -t harryting/vfstlpy37_piptorch_cuda12.3:v1 .`
3. Run this container by `docker run -it --gpus all -v ./:/app/vfstl/src harryting/vfstlpy37_piptorch_cuda12.3:v1`
