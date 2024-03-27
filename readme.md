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

### 2. Build Conda environment
* Using conda and install `pygraphviz`.
    ```bash
    conda install -c conda-forge pygraphviz
    ```
* Install [mujoco](https://www.roboti.us) and [mujoco-py](https://github.com/openai/mujoco-py)
* Install [safty-gym](https://github.com/openai/safety-gym).
    ```
    pip install -e GCRL-LTL/zones/envs/safety/safety-gym/
    ```
* Install required packages
    ```
    conda update --file environment.yaml
    ```

## Experiments
### plot
* save the video of specified STL task
```bash
python exp.py --type='plot' --stl='eventually[0,5] (J0 >= 0.8)'
```

### run 100 experiment and see the statistics of stl robustness, euclidean distance robustness
* chain experiments e.g. $j U (r U (y))$
```bash
python exp.py --type='exp' --task='chain'
```

* avoid experiments e.g. $\neg y U (j \wedge (\neg wUr))$
```bash
python exp.py --type='exp' --task='chain'
```

* stable experiments e.g. $GF(r)$
```bash
python exp.py --type='exp' --task='stable'
```
* once the experiment done, user can view result with ```controller_evaluation_result/box_plot.ipynb```



