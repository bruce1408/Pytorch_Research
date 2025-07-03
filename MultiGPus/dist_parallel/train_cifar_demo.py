import os
import time
import datetime
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from spectrautils import logging_utils, print_utils
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from model import pyramidnet
import argparse

logger = None

# from tensorboardX import SummaryWriter
parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default="./dist_output/last.pth", help='断点训练')
parser.add_argument('--batch_size', type=int, default=256, help='')
parser.add_argument('--num_workers', type=int, default=10, help='')
parser.add_argument('--epoch', type=int, default=6, help='训练epoch次数')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0, 1, 2, 3], help="gpu设备编号")

parser.add_argument('--gpu', default=2, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='GPU通信方式用nccl')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='总共的进程数目')
parser.add_argument('--distributed', action='store_true', help='')
parser.add_argument("--output", default="./dist_output", help="the path of save model")
args = parser.parse_args()

logger_manager = logging_utils.AsyncLoggerManager(work_dir='./logs')
logger = logger_manager.logger
    
def setup_environment(args: argparse.Namespace) -> None:
    """设置环境"""
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    cudnn.benchmark = True
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

     

def main():
    args = parser.parse_args()
    setup_environment(args)
    os.makedirs(args.output, exist_ok=True)
    ngpus_per_node = torch.cuda.device_count()
    print("has gpu nums: ", ngpus_per_node)
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    # print("main worker: ", gpu)
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()
    print("Use GPU: {} for training".format(args.gpu))
    # print(args.gpu, args.rank)

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    print('==> Making model..')
    torch.cuda.set_device(args.gpu)
    net = pyramidnet()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    net.cuda(args.gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)            
            
    print('==> Preparing data..')
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset_train = CIFAR10(
        root='./', 
        train=True, 
        download=True, 
        transform=transforms_train)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              num_workers=args.num_workers,
                              sampler=train_sampler)

    # there are 10 classes so the dataset name is cifar-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)
    
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            if dist.get_rank() == 0:
                print("=> 正在加载断点 '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
            state_dict = checkpoint['model']
            net.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print_utils.print_colored_text("=> 成功加载断点 '{}' (epoch {})".format(args.resume, start_epoch))
        else:
            print("=> 未找到断点 '{}'".format(args.resume))

    # if len(os.listdir(args.output)) != 0:
    #     print("pretrained model has exist!")
    #     checkpoints = torch.load(os.path.join(args.output, "last.pth"), map_location="cpu")
    #     state_dict = checkpoints['model']
    #     net.load_state_dict(state_dict)
    #     optimizer.load_state_dict(checkpoints["optimizer"])
    
    train(net, criterion, optimizer, train_loader, start_epoch, args.epoch, args.gpu)
    

def train(net, criterion, optimizer, train_loader, start_epoch, epoch, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for idx in range(start_epoch, epoch):
        epoch_start = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            start = time.time()

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100 * correct / total

            batch_time = time.time() - start

            # 不会重复打印
            if batch_idx % 20 == 0 and dist.get_rank() == 0:
                logger.info('Epoch: [{}/{}], Batch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                     idx+1, epoch, batch_idx, len(train_loader), train_loss / (batch_idx + 1), acc, batch_time))
            if batch_idx % 100 == 0 and dist.get_rank() == 0:
                state = {
                    "model": net.state_dict(),
                    "batch": batch_idx,
                    "optimizer": optimizer.state_dict(),
                    "epoch": idx + 1,
                }
                torch.save(state, os.path.join(args.output, "last.pth"))
        if dist.get_rank() == 0:
            print_utils.print_colored_box("Epoch {} has been finished!".format(idx+1))

        # print_utils.print_colored_box("model has been saved!")

        elapse_time = time.time() - epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)
        print("Training time {}".format(elapse_time))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")