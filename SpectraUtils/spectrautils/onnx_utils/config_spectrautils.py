import os
import torch, onnx

config_spectrautils = {
    "LAYER_HAS_WEIGHT_ONNX" : ['Conv', 'Gemm', 'ConvTranspose', 'PRelu', 'BatchNormalization'],
    
    "LAYER_HAS_WEIGHT_TORCH" : [
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.linear.Linear,
        torch.nn.modules.conv.ConvTranspose2d,
        torch.nn.modules.activation.PReLU,
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