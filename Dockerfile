# Use Ubuntu 20.04 LTS as the base image
FROM ubuntu:20.04

# Avoid prompts from apt during build
ARG DEBIAN_FRONTEND=noninteractive

# Update and install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    vim \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda create -n myenv python=3.7

WORKDIR /app
COPY test.py /app

RUN conda run -n myenv conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
RUN conda run -n myenv python test.py

# install mojuco
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y vim 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y virtualenv 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y wget \
    xpra 

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev 

 RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \   
    software-properties-common \
    net-tools \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

# install mujoco_py
RUN DEBIAN_FRONTEND=noninteractive apt-get install -q -y libglfw3 libgl1-mesa-glx libosmesa6
RUN git clone https://github.com/openai/mujoco-py
WORKDIR /app/mujoco-py
RUN DEBIAN_FRONTEND=noninteractive apt-get install -q -y gcc patchelf
RUN DEBIAN_FRONTEND=noninteractive apt-get install -q -y libffi-dev
RUN conda run -n myenv pip install "cython<3"
RUN conda run -n myenv pip install --no-cache-dir -r requirements.txt
RUN conda run -n myenv pip install --no-cache-dir -r requirements.dev.txt
RUN conda run -n myenv python setup.py develop
RUN conda init
# install safetyGym
RUN conda run -n myenv conda install -c conda-forge pygraphviz
RUN conda run -n myenv pip install setuptools==65.5.0 
RUN conda run -n myenv pip install wheel==0.38.0
RUN conda run -n myenv pip install stable-baselines3==1.6.2
WORKDIR /app/vfstl
COPY . /app/vfstl/
RUN conda run -n myenv pip install -e ./GCRL-LTL/zones/envs/safety/safety-gym/
# install antRoom

# install pybullet

# Set the default command for the container
CMD ["bash"]