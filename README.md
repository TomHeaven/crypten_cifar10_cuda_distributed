# CrypTen Cifar10 Distributed Demo with CUDA

This project shows how to run CrypTen Cifar10 demo on distributed servers with CUDA enanbled.

## Environment preparation

+ Install CUDA environment, pytorch, and make sure they are working normally.
+ Install [CrypTen](https://github.com/facebookresearch/CrypTen) using source (recommended).
+ Glone this project, and change into the project directory.

## Data preparation

Manually download cifar10 python dataset `cifar-10-python.tar.gz` and put it in `data` foler, or you can ignore the step and let the program download the dataset automatically each time it runs.

## Test locally on single server with two Nvidia GPUs

+ Assume you have two parties, start the first party process by

```shell
# rank 0
python launch_distributed.py  --evaluate \
    --world_size 2 \
    --rank 0 \
    --master_address 127.0.0.1 \
    --master_port 12345 \
    --gpu_id 0 \
    --model-location model/model_best.pth.tar \
    --resume  \
    --batch-size 50 \
    --print-freq 1 \
    --skip-plaintext \
    --distributed
```

+ Start the second party process by

```shell
# rank 1
python launch_distributed.py  --evaluate \
    --world_size 2 \
    --rank 1 \
    --master_address 127.0.0.1 \
    --master_port 12345 \
    --gpu_id 1 \
    --model-location model/model_best.pth.tar \
    --resume  \
    --batch-size 50 \
    --print-freq 1 \
    --skip-plaintext \
    --distributed
```
Note the master_address, master_port, and gpu_id can be changed but must be available.

## Test distributedly on two servers with an Nvidia GPU
+ Ensure the two servers are connected with on the same network.
+ Assume you have two parties, and the master node IP address is 192.168.0.1.
+ Start the first party process on the master node by

```shell
# rank 0
python launch_distributed.py  --evaluate \
    --world_size 2 \
    --rank 0 \
    --master_address 192.168.0.1 \
    --master_port 12345 \
    --gpu_id 0 \
    --model-location model/model_best.pth.tar \
    --resume  \
    --batch-size 50 \
    --print-freq 1 \
    --skip-plaintext \
    --distributed
```

+ Start the second party process  the other node by

```shell
# rank 1
python launch_distributed.py  --evaluate \
    --world_size 2 \
    --rank 1 \
    --master_address 192.168.0.1 \
    --master_port 12345 \
    --gpu_id 0 \
    --model-location model/model_best.pth.tar \
    --resume  \
    --batch-size 50 \
    --print-freq 1 \
    --skip-plaintext \
    --distributed
```

Note the master_address, master_port, and gpu_id can be changed but must be available.