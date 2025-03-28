import os
from unittest.mock import patch, MagicMock
from spectrautils.onnx_utils import visualize_onnx_model_weights,visualize_torch_model_weights
import torch, torchvision, onnx
from  torchvision.models import ResNet18_Weights

if __name__ == '__main__':
    
    onnx_path = "/share/cdd/onnx_models/od_bev_0317.onnx"
    model_name = "od_bev_test"
    # results_dir = "/mnt/share_disk/bruce_trie/workspace/Pytorch_Research/SpectraUtils/od_bev_onnx_visualization_results"
    visualize_onnx_model_weights(onnx_path, model_name)
    
    # Example usage with different models
    # model_old = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # 加载你的本地模型
    # model_new = torch.load('/mnt/share_disk/bruce_trie/workspace/Pytorch_Research/SpectraUtils/spectrautils/resnet_model_cle_bc.pt')

    # model_new = torchvision.models.resnet18(pretrained=False)

    
    # visualize_torch_model_weights(model_new, "resnet18_new")
    
