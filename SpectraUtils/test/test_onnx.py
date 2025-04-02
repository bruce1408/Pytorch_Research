import os
import torch
from  torchvision.models import ResNet18_Weights
from spectrautils.onnx_utils import visualize_onnx_model_weights,visualize_torch_model_weights
from spectrautils.onnx_utils import export_model_onnx


if __name__ == '__main__':
    
    # 加载onnx模型
    onnx_path = "/share/cdd/onnx_models/od_bev_0317.onnx"
    model_name = "od_bev_test"
    visualize_onnx_model_weights(onnx_path, model_name)
    
    # 加载torch模型
    model_new = torch.load('/share/cdd/onnx_models/resnet_model_cle_bc.pt')
    visualize_torch_model_weights(model_new, "resnet18_new")
    
