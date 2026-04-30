# mlframework

Neural network framework implemented from scratch in C++20.

## Features

* N-dimensional tensor with stride-based memory layout
* Automatic differentiation using a dynamic computation graph
* Optimizers: SGD (momentum), Adam
* Layers: Linear, BatchNorm1d, Dropout, ReLU, Sigmoid
* Loss: Cross-entropy with fused softmax backward
* DataLoader: MNIST (IDX parsing and shuffling)
* Learning rate scheduling: Cosine Annealing with Warm Restarts
* Matrix multiplication accelerated with OpenBLAS

## Results

### Model

* MLP: 784 → 128 → 64 → 10
* Batch Normalization
* Dropout (p = 0.3)

### Training

* Optimizer: Adam
* Learning rate schedule: Cosine Annealing with Warm Restarts
* Epochs: 15
* Batch size: 64
* Learning rate: 0.001

### Performance

* 98.2% test accuracy on MNIST

## Project Structure

```
include/mlframework/   Public headers  
src/                   Implementations  
tests/                 Unit tests  
cli/                   Training CLI  
data/mnist/            MNIST dataset (not tracked)  
```

## Dataset Setup

```bash
mkdir -p data/mnist
cd data/mnist

wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz
```

## Build

```bash
# Dependencies
sudo apt update
sudo apt install libopenblas-dev

# Debug build
cmake --preset debug
cmake --build --preset debug

# Release build
cmake --preset release
cmake --build --preset release
```

## Training

```bash
./build/release/mlframework \
  --data data/mnist \
  --epochs 15 \
  --batch 64 \
  --lr 0.001
```