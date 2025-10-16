import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from torchvision import models
from  torchvision.models import ResNet18_Weights
from spectrautils.onnx_utils import visualize_onnx_model_weights,visualize_torch_model_weights
from spectrautils.onnx_utils import export_model_onnx, visualize_onnx_model_weights

if __name__ == '__main__':
    
    # 加载onnx模型
    # onnx_path = "/home/bruce_ultra/workspace/perception_quanti/avp_parkspace/20240603/psd2d_v1_1_0_8650_eca_simplifier.onnx"
    # model_name = "psd2d"
    # visualize_onnx_model_weights(onnx_path, model_name)
    
    # exit(0)
    
    # 加载torch模型
    # model_new = torch.load('/home/bruce_ultra/workspace/quant_workspace/Quantizer-Tools/_outputs/models/resnet18-f37072fd.pth')
    model_new = models.resnet18(pretrained=True)
    visualize_torch_model_weights(model_new, "resnet18_new")
    
    # exit(0)
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
    
    
    

    
    
