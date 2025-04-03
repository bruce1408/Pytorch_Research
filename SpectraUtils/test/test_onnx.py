import os
import torch
from torchvision import models
from  torchvision.models import ResNet18_Weights
from spectrautils.onnx_utils import visualize_onnx_model_weights,visualize_torch_model_weights
from spectrautils.onnx_utils import export_model_onnx


if __name__ == '__main__':
    
    # 加载onnx模型
    onnx_path = "/share/cdd/onnx_models/od_bev_0317.onnx"
    model_name = "od_bev_test"
    # visualize_onnx_model_weights(onnx_path, model_name)
    
    # 加载torch模型
    model_new = torch.load('/share/cdd/onnx_models/resnet_model_cle_bc.pt')
    # visualize_torch_model_weights(model_new, "resnet18_new")
    
    
    # 加载预训练的ResNet18模型
    model = models.resnet18(pretrained=True)
    
    # 定义输入信息
    input_info = {'input': (1, 3, 224, 224)}  # ResNet18 期望的输入尺寸
    
    # 定义输出名称
    output_names = ['output']
    export_model_onnx(model, 
                      input_info,
                      "/mnt/share_disk/bruce_trie/workspace/perception_quanti/demo_18/resnet18.onnx",
                      output_names
                    )
    
    
