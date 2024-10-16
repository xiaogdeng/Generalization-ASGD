## Stability and Generalization of Asynchronous SGD: Sharper Bounds Beyond Lipschitz and Smoothness


This repository is the official implementation of paper **Stability and Generalization of Asynchronous SGD: Sharper Bounds Beyond Lipschitz and Smoothness**. 

### Requirements

```python
# setup
# GPU environment required
torch>=1.10.0
torchvision>=0.11.1
numpy>=1.19.5
```

### Dataset

The CIFAR-10 and CIFAR-100 datasets can be downloaded automatically by `torchvision.datasets`. The RCV1 data set from the LIBSVM database are available with `libsvm_data.py`.

### Example Usage

```python
# train
python server.py --cuda-ps --model 'ResNet' --dataset 'CIFAR100' \
                 --delay-type 'random' --delay 16 \
                 --num-workers 16 --lr 0.01 \
                 --num-epochs 200 --seed 42
```

### Usage

```python
usage: server.py [-h]
    [--model  {ResNet, Linearnet_rcv1}]
    [--dataset  {CIFAR10, CIFAR100, RCV1}]
    [--delay  Number of delays]
    [--num-workers  Number of workers]
    [--lr  Learning rate] [--logdir LOGDIR]
    [--batch-size  Batch size] 
    [--num-epochs  Epoch]
    [--seed Random  Seed]
```

#### Note

* We provide a demo bash script file `bashrun.sh`

