import os
from pathlib import Path

import onnx
import torch

dogs_cats_dataset_path = '/mnt/share_disk/bruce_trie/misc_data_products/dogs_cats'


imagenet_data_path = "/mnt/share_disk/bruce_trie/workspace/outputs/imagenet_dataset"


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 更换机器时只需修改这里，或设置环境变量 ONNX_OPERATOR_VIS_OUTPUT_DIR。
# 统一输出根目录，各脚本在此目录下拼接自己的多级子目录。
ONNX_OPERATOR_VIS_OUTPUT_DIR = Path(
    os.environ.get(
        "ONNX_OPERATOR_VIS_OUTPUT_DIR",
        "/share_data/bruce/workspace/Outputs",
    )
).expanduser()

# 保留原字典接口，避免其他代码引用 config_operator 时失效。
config_operator = {
    "save_dir": str(ONNX_OPERATOR_VIS_OUTPUT_DIR),
}


def get_onnx_operator_output_dir(*subdirs):
    """返回并创建 ONNX 算子模型输出目录。"""
    output_dir = ONNX_OPERATOR_VIS_OUTPUT_DIR.joinpath(*subdirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def get_onnx_operator_output_path(filename, *subdirs):
    """返回 ONNX 输出文件路径，并确保父目录存在。"""
    return str(Path(get_onnx_operator_output_dir(*subdirs)) / filename)

config_onnx_inference = {
    
}

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
