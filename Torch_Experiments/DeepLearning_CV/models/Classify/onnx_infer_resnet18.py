#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet18 ONNX Runtime 推理代码
用于加载和运行 ResNet18 ONNX 模型进行图像分类推理
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import onnxruntime as ort
import argparse
from typing import Tuple, List, Dict, Optional


class ResNet18ONNXInference:
    """ResNet18 ONNX 模型推理类"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        初始化 ResNet18 ONNX 推理器
        
        Args:
            model_path: ONNX 模型文件路径
            device: 推理设备 ('cpu', 'cuda', 'tensorrt' 等)
        """
        self.model_path = model_path
        self.device = device.lower()
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX 模型文件不存在: {model_path}")
        
        # 设置 ONNX Runtime 会话选项
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 根据设备设置执行提供者
        if self.device == 'cuda' or self.device == 'gpu':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif self.device == 'tensorrt':
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # 创建 ONNX Runtime 会话
        print(f"正在加载 ONNX 模型: {model_path}")
        print(f"使用设备: {self.device}")
        
        try:
            self.session = ort.InferenceSession(
                model_path, 
                sess_options=self.session_options,
                providers=providers
            )
        except Exception as e:
            print(f"加载 ONNX 模型失败: {e}")
            print("尝试使用默认提供者...")
            self.session = ort.InferenceSession(model_path)
        
        # 获取模型输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 获取输入形状
        self.input_shape = self.session.get_inputs()[0].shape
        print(f"模型输入形状: {self.input_shape}")
        print(f"输入名称: {self.input_name}")
        print(f"输出名称: {self.output_name}")
        
        # 假设输入形状为 [batch, channels, height, width]
        if len(self.input_shape) == 4:
            self.batch_size, self.channels, self.height, self.width = self.input_shape
        else:
            # 默认 ResNet18 输入尺寸
            self.batch_size = 1
            self.channels = 3
            self.height = 224
            self.width = 224
        
        # ImageNet 类别标签 (简化版，实际应有1000类)
        self.imagenet_labels = self._load_imagenet_labels()
        
        print(f"模型加载成功！输入尺寸: {self.height}x{self.width}")
    
    def _load_imagenet_labels(self) -> List[str]:
        """加载 ImageNet 类别标签"""
        # 这里使用简化的标签，实际应该加载完整的 ImageNet 标签文件
        # 你可以从 https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json 下载
        labels = [
            "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
            "electric ray", "stingray", "rooster", "hen", "ostrich"
        ]
        # 为了演示，我们只返回前10个标签，实际应该有1000个
        return labels
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        预处理图像，使其符合模型输入要求
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            预处理后的 numpy 数组
        """
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        
        # 调整尺寸
        img = img.resize((self.width, self.height), Image.Resampling.BILINEAR)
        
        # 转换为 numpy 数组并归一化
        img_array = np.array(img).astype(np.float32)
        
        # 归一化到 [0, 1]
        img_array = img_array / 255.0
        
        # ImageNet 标准化 (使用标准 ImageNet 均值和标准差)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        
        img_array = (img_array - mean) / std
        
        # 转换维度顺序: HWC -> CHW
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # 添加批次维度: CHW -> NCHW
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def inference(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行推理
        
        Args:
            input_data: 预处理后的输入数据
            
        Returns:
            模型输出
        """
        # 确保输入数据类型正确
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        # 执行推理
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        inference_time = time.time() - start_time
        
        print(f"推理时间: {inference_time * 1000:.2f} ms")
        
        return outputs[0]
    
    def postprocess(self, output: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        后处理输出结果
        
        Args:
            output: 模型输出
            top_k: 返回前K个预测结果
            
        Returns:
            包含预测结果的字典列表
        """
        # 假设输出是 softmax 后的概率分布
        # 获取 top-k 预测
        if len(output.shape) == 2:
            output = output[0]  # 取第一个批次
        
        # 获取 top-k 索引和概率
        top_k_indices = np.argsort(output)[-top_k:][::-1]
        top_k_probs = output[top_k_indices]
        
        results = []
        for idx, prob in zip(top_k_indices, top_k_probs):
            label = self.imagenet_labels[idx] if idx < len(self.imagenet_labels) else f"类别 {idx}"
            results.append({
                "class_id": int(idx),
                "class_name": label,
                "probability": float(prob)
            })
        
        return results
    
    def predict_image(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """
        完整的图像预测流程
        
        Args:
            image_path: 输入图像路径
            top_k: 返回前K个预测结果
            
        Returns:
            预测结果列表
        """
        # 1. 预处理
        print(f"处理图像: {image_path}")
        input_data = self.preprocess_image(image_path)
        
        # 2. 推理
        output = self.inference(input_data)
        
        # 3. 后处理
        results = self.postprocess(output, top_k)
        
        return results
    
    def benchmark(self, image_path: str, num_iterations: int = 100):
        """
        性能基准测试
        
        Args:
            image_path: 测试图像路径
            num_iterations: 迭代次数
        """
        print(f"\n开始性能基准测试 ({num_iterations} 次迭代)...")
        
        # 预处理图像
        input_data = self.preprocess_image(image_path)
        
        # 预热
        for _ in range(10):
            _ = self.session.run([self.output_name], {self.input_name: input_data})
        
        # 正式测试
        start_time = time.time()
        for i in range(num_iterations):
            _ = self.session.run([self.output_name], {self.input_name: input_data})
        
        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        
        print(f"总时间: {total_time:.2f} 秒")
        print(f"平均推理时间: {avg_time * 1000:.2f} ms")
        print(f"FPS: {1.0 / avg_time:.2f}")
        
        return avg_time


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ResNet18 ONNX Runtime 推理')
    parser.add_argument('--model', type=str, default='/home/mic-743/Documents/demo/resnet18.onnx',
                       help='ONNX 模型文件路径 (默认: resnet18.onnx)')
    parser.add_argument('--image', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'tensorrt'],
                       help='推理设备 (默认: cpu)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='显示前K个预测结果 (默认: 5)')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能基准测试')
    parser.add_argument('--iterations', type=int, default=100,
                       help='基准测试迭代次数 (默认: 100)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        return
    
    # 创建推理器
    try:
        inferencer = ResNet18ONNXInference(args.model, args.device)
    except Exception as e:
        print(f"初始化推理器失败: {e}")
        print("\n请确保:")
        print("1. ONNX 模型文件存在")
        print("2. onnxruntime 已安装 (pip install onnxruntime)")
        print("3. 如果使用 GPU，请安装 onnxruntime-gpu")
        return
    
    # 执行推理
    print(f"\n正在推理图像: {args.image}")
    results = inferencer.predict_image(args.image, args.top_k)
    
    # 显示结果
    print(f"\nTop-{args.top_k} 预测结果:")
    print("-" * 50)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['class_name']} (ID: {result['class_id']}): "
              f"{result['probability'] * 100:.2f}%")
    
    # 运行基准测试
    if args.benchmark:
        inferencer.benchmark(args.image, args.iterations)


if __name__ == "__main__":
    main()