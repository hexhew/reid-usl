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

## Prepare datasets

Public datasets are available from official websites. It is recommanded to download and extract the dataset somewhere outside the project directory and symlink the dataset root to `reid-usl/data` as below. You can change the corresponding paths in the config files if you want to use a different folder structure.

```
reid-usl
|-- configs
|-- data
|   |-- market1501
|   |   |-- bounding_box_train
|   |   |-- bounding_box_test
|   |   |-- query
|   |-- duke
|   |   |-- bounding_box_train
|   |   |-- bounding_box_test
|   |   |-- query
```


## Train models on standard datasets

### Training on multiple GPUs

We provide `tools/dist_train.sh` to launch training on multiple GPUs like [mmdetection](https://github.com/open-mmlab/mmdetection). The basic usage is as follows.

```
$ ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} \
    [optional arguments]
```

During training, log files and checkpoints will be saved to the working ditectory, which is the basename of `${CONFIG_FILE}` or specified in config file.

This tool accepts optional arguments. Please see the [source code](../tools/train.py) or run `python tools/train.py --help` for details.

### Training on a single GPU

We provide `tools/train.py` to launch training jobs on a single GPU. The basic usage is as follows.

```
$ python tools/train.py ${CONFIG_FILE} [optional arguments]
```

## Test models

Choose the proper script to perform testing.

```
# single-gpu testing
$ python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}

# multi-gpu testing
$ ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}
```

## Examples


1. Train SpCL on Market-1501 on 2 GPUs:

```
$ ./tools/dist_train.sh configs/spcl/spcl_market1501.py 2
```

2. Test your model after training:

```
$ ./tools/dist_test.sh configs/spcl/spcl_market1501.py work_dir/spcl_market1501/latest.pth 2
```
