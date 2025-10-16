import sys
import onnx
from collections import Counter

def print_operator_summary(model_path: str, detailed: bool = False) -> None:
    """
    打印 ONNX 模型中的算子信息
    Args:
        model_path: ONNX 模型文件的路径
        detailed: 是否打印详细信息，默认为 False
    """
    # 加载 ONNX 模型
    model = onnx.load(model_path)
    
    # 获取所有算子
    operators = {}
    for node in model.graph.node:
        op_type = node.op_type
        operators[op_type] = operators.get(op_type, 0) + 1
    
    print("\n=== ONNX Model Operators ===")
    print(f"Total unique operators: {len(operators)}")
    print("\nOperator distribution:")
    
    # 按照出现次数排序
    sorted_ops = sorted(operators.items(), key=lambda x: x[1], reverse=True)
    for op_type, count in sorted_ops:
        print(f"- {op_type}: {count}")
        if detailed:
            # 打印使用该算子的节点的详细信息
            print("  Nodes:")
            for node in model.graph.node:
                if node.op_type == op_type:
                    print(f"    Input: {node.input}")
                    print(f"    Output: {node.output}")
                    if node.attribute:
                        print("    Attributes:")
                        for attr in node.attribute:
                            print(f"      - {attr.name}")
                    print()
                    
                    
def print_operator_details(model_path, count_ops=True, show_details=True, output_file=""):
    """加载ONNX模型并打印所有算子"""
    
    def write_and_print(f, text=""):
        print(text)
        if f:
            f.write(text + '\n')
            
    f = None
    original_stdout = sys.stdout # 保存原始的标准输出

    try:
        # 加载ONNX模型
        model = onnx.load(model_path)
        
        # 获取图结构
        graph = model.graph
        
        # 如果指定了输出文件，将输出重定向到文件
        if output_file:
            f = open(output_file, 'w', encoding='utf-8')
        
        header_info = [
            f"模型: {model_path}",
            f"IR版本: {model.ir_version}",
            f"生产者名称: {model.producer_name or '未知'}",
            f"生产者版本: {model.producer_version or '未知'}",
            f"模型版本: {model.model_version}",
            f"算子集版本: {model.opset_import[0].version}",
            f"节点数量: {len(graph.node)}",
            "-" * 50
        ]
        
        for line in header_info:
            write_and_print(f, line)
            
        
        # 收集所有算子类型
        op_types = [node.op_type for node in graph.node]
        
        if count_ops:
            # 统计每种算子的数量
            op_counter = Counter(op_types)
            write_and_print(f, "算子统计:")
            for op_type, count in sorted(op_counter.items(), key=lambda x: x[1], reverse=True):
                write_and_print(f, f"  {op_type}: {count}")
            write_and_print(f, "-" * 50)
        
        if show_details:
            # 显示每个算子的详细信息
            write_and_print(f, "算子详细信息:")
            for i, node in enumerate(graph.node):
                write_and_print(f, f"[{i}] 类型: {node.op_type}")
                write_and_print(f, f"    名称: {node.name or '未指定名称'}")
                write_and_print(f, f"    输入: {', '.join(node.input)}")
                write_and_print(f, f"    输出: {', '.join(node.output)}")
                if node.attribute:
                    write_and_print(f, "    属性:")
                    for attr in node.attribute:
                        write_and_print(f, f"      {attr.name}: {onnx.helper.get_attribute_value(attr)}")
        # else:
        #     # 只显示唯一的算子类型和名称
        #     write_and_print(f, "模型中的算子类型和名称:")
        #     for i, node in enumerate(graph.node):
        #         write_and_print(f, f"  [{i}] {node.op_type}: {node.name or '未指定名称'}")
        
        # 如果指定了输出文件，恢复标准输出并关闭文件
        if f:
            f.close()
            print(f"结果已保存到文件: {output_file}")
            
    except Exception as e:
        # 确保异常时恢复标准输出
        if f:
            f.close()
        print(f"错误: {str(e)}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    
    onnx_path = "/home/bruce_ultra/workspace/quant_workspace/Quantizer-Tools/_outputs/models/yolov8n.onnx"
    # print_operator_summary(onnx_path)
    print_operator_details(onnx_path, show_details=False)
    