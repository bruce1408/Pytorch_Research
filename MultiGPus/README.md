# pytorch-multigpu
Multi GPU Training Code for Deep Learning with PyTorch. Train PyramidNet for CIFAR10 classification task. This code is for comparing several ways of multi-GPU training.

# Requirement
- Python 3
- PyTorch 1.0.0+
- TorchVision
- TensorboardX

# Usage
### single gpu
```
cd single_gpu
python train.py 
```

### DataParallel
```
cd data_parallel
python train.py --gpu_devices 0 1 2 3 --batch_size 768
```

### DistributedDataParallel
```
cd dist_parallel
python train.py --gpu_device 0 1 2 3 --batch_size 768
```

# Performance
### single gpu
- batch size: 240
- batch time: 6s
- training time: 22 min 
- gpu util: 99 %
- gpu memory: 10 G

### DataParallel(4 k80)
- batch size: 768
- batch time: 5s
- training time: 5 min 
- gpu util: 99 %
- gpu memory: 10 G * 4


`torch.nn.DataParallel` 和 `torch.utils.data.distributed` 的主要区别在于它们各自解决的问题不同，分别针对模型并行和数据分布：

1. **torch.nn.DataParallel(单进程多GPU)**  
   - **用途**：用于在单个进程内将模型在多个 GPU 上并行运行。  
   - **工作原理**：  
     - 将整个模型复制（复制参数和计算图）到每个 GPU 上。  
     - 自动将输入 batch 按 GPU 数量进行分割，并在各 GPU 上同时前向传播。  
     - 汇总各个 GPU 的输出，并在主 GPU 上进行反向传播和梯度合并。  
   - **适用场景**：当你在单个机器上使用多个 GPU 时，可以方便地利用 DataParallel 来提升计算速度。不过，它在跨多节点分布式训练时表现不如 DistributedDataParallel。

2. **torch.utils.data.distributed(多进程多GPU)**  
   - **用途**：主要用于数据分布，通常配合分布式训练来均匀地分配数据。  
   - **组件**：最常用的组件是 `DistributedSampler`。  
     - **DistributedSampler**：在多进程（每个进程通常对应一个 GPU）环境中，将数据集划分为不同的子集，每个进程只处理自己分到的数据，这样可以保证在一个 epoch 内，每个样本只被处理一次，同时避免重复和遗漏。  
   - **适用场景**：当你采用分布式训练（例如使用 `torch.nn.parallel.DistributedDataParallel`）时，通过 DistributedSampler 能够确保每个 GPU 进程拿到不同的数据，充分利用集群资源。

### 总结

- **DataParallel**：针对模型计算，在单个进程内复制模型到多个 GPU 并行执行计算；适用于单机多卡场景，但不一定能最优利用资源，且在扩展到多机时不建议使用。  
- **torch.utils.data.distributed (DistributedSampler)**：针对数据加载和分发，在分布式训练中将数据集分片，确保各个进程（或 GPU）处理不同的数据；是分布式训练环境下的重要组件。

两者经常在分布式训练中配合使用：模型训练通常采用 `DistributedDataParallel`（而非 DataParallel），而数据加载则借助 DistributedSampler 从 `torch.utils.data.distributed` 来实现数据的均匀分配。



`NVLink` 和 `NCCL` 都与 GPU 之间的高效通信密切相关，但它们的作用不同，它们是相辅相成的：

### 1. **NVLink**（NVIDIA NVLink）
- **是什么**：  
  NVLink 是 NVIDIA 开发的一种高速互联技术，用于在多个 GPU 之间提供高带宽、低延迟的数据通信。它是一个硬件层面的技术，具体来说是 NVIDIA 提供的 GPU 间通信接口，旨在提高多 GPU 之间的带宽，尤其是在多 GPU 系统内。
  
- **作用**：  
  - NVLink 提供了比传统 PCIe 更高的带宽和更低的延迟。
  - 允许多个 GPU 之间直接传输数据，从而加速深度学习中的模型训练和推理过程。
  - 在支持 NVLink 的系统中，GPU 之间的连接比 PCIe 更快，尤其是在大规模并行计算中（例如梯度交换等操作），能够显著减少通信瓶颈。

- **主要特点**：
  - 高带宽：相比 PCIe，NVLink 提供了更高的数据传输速度。
  - 更低的延迟：减少了 GPU 之间数据交换的延迟，优化了多 GPU 计算的效率。
  - 可扩展性：支持多个 GPU 之间的直接互联，可以形成更大的 GPU 集群。

### 2. **NCCL**（NVIDIA Collective Communication Library）
- **是什么**：  
  NCCL 是 NVIDIA 提供的一个高效的集体通信库，专门用于深度学习中多 GPU 或多节点分布式训练中的数据通信（尤其是梯度汇总和参数同步等操作）。

- **作用**：
  - NCCL 在深度学习训练中提供了一些重要的集体通信操作，如 **All-Reduce**、**All-Gather**、**Reduce**、**Broadcast** 等。
  - 它是一个软件层面的库，设计用于在多个进程（通常是多个 GPU 或多个节点）之间高效地交换数据。
  - NCCL 自动选择最佳的硬件通信方式。如果在使用支持 NVLink 的 GPU，它会利用 NVLink 提供的高速连接；如果在没有 NVLink 的环境中，NCCL 会回退到 PCIe 或 RDMA。

- **主要特点**：
  - 高效的并行计算支持：NCCL 能够充分利用 GPU 间的带宽进行通信，特别是在大规模分布式训练时，能够显著减少通信延迟。
  - 支持多 GPU 和多节点：无论是在单机多卡训练还是跨多节点的分布式训练中，NCCL 都能高效地进行数据交换。
  - 软件与硬件优化：在硬件支持的情况下，NCCL 会自动利用 NVLink 或 RDMA 等硬件加速通信。

### **NVLink 和 NCCL 的关系**
- **硬件与软件的配合**：  
  NVLink 是硬件层面的 GPU 间通信技术，提供高带宽和低延迟的通信通道。而 NCCL 是软件层面的通信库，专门为深度学习应用中的数据交换提供高效的算法和接口。当 NCCL 执行跨 GPU 或跨节点的通信操作时，**如果系统中支持 NVLink**，它会自动使用 NVLink 作为通信通道，以充分利用 NVLink 提供的高带宽。

- **共同工作**：  
  在使用支持 NVLink 的多 GPU 系统时，NCCL 会使用 NVLink 来加速数据的交换。特别是在执行如 **All-Reduce** 等操作时，NCCL 会通过 NVLink 实现 GPU 之间的高速通信。
  
  在没有 NVLink 的环境中，NCCL 会回退到其他方式（如 PCIe 或 RDMA）进行通信，但在支持 NVLink 的环境中，使用 NVLink 会显著提升通信效率。

### 总结
- **NVLink** 提供了 **硬件层面** 的高速连接，帮助 GPU 之间实现高带宽、低延迟的通信。
- **NCCL** 是 **软件层面** 的库，提供了集体通信的接口，优化了跨 GPU 或跨节点的通信。在支持 NVLink 的环境中，NCCL 会利用 NVLink 的高带宽来加速数据交换。

因此，NVLink 和 NCCL 是协同工作的，NVLink 提供了硬件基础，而 NCCL 提供了高效的通信协议，二者一起帮助提升分布式训练中的通信效率。