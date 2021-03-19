## Prerequisites

- Linux or mac OSX
- Python 3.6+
- PyTorch 1.6+ (not tested with v1.8)
- CUDA 10.2

## Installation

1. Create a conda environment and activate it.

```
$ conda create --name reid-usl python=3.8 -y
$ conda activate reid-usl
```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).

```
$ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

3. Clone the repository.

```
$ git clone https://github.com/hexifer/reid-usl.git
$ cd reid-usl
```

4. Install build requirements and runtime requirements, and then install reid-usl

```
$ pip install -r requirements/build.txt
$ pip install -r requirements/runtime.txt
$ pip install -e .
```
