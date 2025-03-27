import onnx

def print_model_operators(model_path: str, detailed: bool = False) -> None:
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
                    
if __name__ == "__main__":
    
    onnx_path = "/share/cdd/onnx_models/od_bev_0317.onnx"
    print_model_operators(onnx_path)
    