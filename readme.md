# VFSTL


## Setup

### Pull the project with sub-modules
1. `git pull --recurse-submodules git@github.com:harryting-yiting/VFSTL.git`

### 1. Build Docker environemnt 
* Prerequisites
    1. docker engine
    2. nvidia-docker-toolkit
* Build Docker image 
    1. Change to the project directory 
    2. Build the docker with `docker build -t harryting/vfstlpy37_piptorch_cuda12.3:v1 .`
    3. Run this container by `docker run -it --gpus all -v ./:/app/vfstl/src harryting/vfstlpy37_piptorch_cuda12.3:v1`

### 2. Build 
