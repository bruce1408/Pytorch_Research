import os
import onnx
import torch

dogs_cats_dataset_path = '/mnt/share_disk/bruce_trie/misc_data_products/dogs_cats'


config_operator = {
    "save_dir" : "/mnt/share_disk/bruce_trie/workspace/Pytorch_Research/ONNX_Operator_Vis/output_onnx"
}



config_spectrautils = {
    "LAYER_HAS_WEIGHT_ONNX" : ['Conv', 'Gemm', 'ConvTranspose', 'PRelu', 'BatchNormalization'],
    
    "LAYER_HAS_WEIGHT_TORCH" : [
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.linear.Linear,
        torch.nn.modules.batchnorm.BatchNorm2d
    ],
    
    "VALID_WEIGHT_TYPES" : {
        onnx.TensorProto.FLOAT, 
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.INT8,
        onnx.TensorProto.INT16,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
    }
    
}


