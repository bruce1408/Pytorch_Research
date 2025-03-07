#!/usr/bin/env python
# -*- coding: utf-8 -*-

import onnx
import argparse
import sys
from collections import Counter

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='打印ONNX模型中的所有算子')
    parser.add_argument('--model_path', type=str, 
                        default="/home/bruce_ultra/workspace/8620_code_repo/8620_code_x86/onnx_models/od_bev_0306.onnx",
                        help='ONNX模型文件路径')
    parser.add_argument('--count',   '-c', default=True, help='统计每种算子的数量')
    parser.add_argument('--details', '-d', default=True, help='显示每个算子的详细信息')
    parser.add_argument('--output', '-o', type=str, default="./od_bev_0306.txt", help='输出文件路径，不指定则输出到控制台')
    return parser.parse_args()  # 这里需要返回解析结果

def print_operators(model_path, count_ops=False, show_details=False, output_file=""):
    """加载ONNX模型并打印所有算子"""
    try:
        # 加载ONNX模型
        model = onnx.load(model_path)
        
        # 获取图结构
        graph = model.graph
        
        # 如果指定了输出文件，将输出重定向到文件
        if output_file:
            sys.stdout = open(output_file, 'w', encoding='utf-8')
        
        print(f"模型: {model_path}")
        print(f"IR版本: {model.ir_version}")
        print(f"生产者名称: {model.producer_name}" if model.producer_name else "生产者名称: 未知")
        print(f"生产者版本: {model.producer_version}" if model.producer_version else "生产者版本: 未知")
        print(f"模型版本: {model.model_version}")
        print(f"算子集版本: {model.opset_import[0].version}")
        print(f"节点数量: {len(graph.node)}")
        print("-" * 50)
        
        # 收集所有算子类型
        op_types = [node.op_type for node in graph.node]
        
        if count_ops:
            # 统计每种算子的数量
            op_counter = Counter(op_types)
            print("算子统计:")
            for op_type, count in sorted(op_counter.items(), key=lambda x: x[1], reverse=True):
                print(f"  {op_type}: {count}")
            print("-" * 50)
        
        if show_details:
            # 显示每个算子的详细信息
            print("算子详细信息:")
            for i, node in enumerate(graph.node):
                print(f"[{i}] 类型: {node.op_type}")
                print(f"    名称: {node.name}" if node.name else "    名称: 未指定")
                print(f"    输入: {', '.join(node.input)}")
                print(f"    输出: {', '.join(node.output)}")
                if node.attribute:
                    print("    属性:")
                    for attr in node.attribute:
                        print(f"      {attr.name}: {onnx.helper.get_attribute_value(attr)}")
        else:
            # 只显示唯一的算子类型和名称
            print("模型中的算子类型和名称:")
            for i, node in enumerate(graph.node):
                print(f"  [{i}] {node.op_type}: {node.name}" if node.name else f"  [{i}] {node.op_type}: 未指定名称")
        
        # 如果指定了输出文件，恢复标准输出并关闭文件
        if output_file:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            print(f"结果已保存到文件: {output_file}")
            
    except Exception as e:
        # 确保异常时恢复标准输出
        if output_file and sys.stdout != sys.__stdout__:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        print(f"错误: {str(e)}", file=sys.stderr)
        return 1
    
    return 0

def main():
    args = parse_args()
    return print_operators(args.model_path, args.count, args.details, args.output)

if __name__ == "__main__":
    sys.exit(main())