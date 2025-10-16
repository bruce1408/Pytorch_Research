import onnx

def get_onnx_operators(model_path: str) -> set[str]:
    """
    获取ONNX模型中使用的所有运算符
    
    参数:
        model_path (str): ONNX模型文件路径
        
    返回:
        set[str]: 包含所有唯一运算符类型的集合
    """
    try:
        model = onnx.load(model_path)
        
        # 创建一个空的集合来存储操作符
        operators = set()  
        
        # 遍历模型图中的每个节点
        for node in model.graph.node:          
        
            # 将当前节点的操作类型添加到集合中
            operators.add(node.op_type)  
    
        # 返回包含所有唯一操作符的集合
        return operators  
    except Exception as e:
        print(f"加载或解析ONNX模型时出错: {e}")
        return set()

# 使用示例
if __name__ == "__main__":
    onnx_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/models/resnet18.onnx"
    operators = get_onnx_operators(onnx_path)
    
    if operators:
        print("ONNX模型中使用的运算符:")
        print(*operators, sep="\n")
        