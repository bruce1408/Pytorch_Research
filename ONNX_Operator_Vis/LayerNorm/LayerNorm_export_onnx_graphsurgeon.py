import os
import torch
import onnx
import datetime
import onnxsim
import torch.onnx
import torch.nn as nn
import numpy as np
import onnx_graphsurgeon as gs

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.norm  = nn.LayerNorm(3)
        self.act   = nn.ReLU()

    def forward(self, x):
        _, _, H, W = x.shape
        L = H * W
        x = self.conv1(x)
        x = x.view(x.shape[0], x.shape[1], L).permute(0, 2, 1)
        x = self.norm(x)
        x = self.act(x)
        return x

def export_onnx_graph(onnx_path):
    input  = torch.Tensor(1, 3, 224, 224).uniform_(-1, 1)
    model  = Model()
    model.eval()
    # now = datetime.datetime.now()
    # dir_path_name = os.path.abspath(os.path.dirname(__file__))
    # file = os.path.join(dir_path_name, f"layernorm_{now.strftime('%Y%m%d_%H%M%S')}.onnx")
    # file = os.path.join(dir_path_name, "./sample-ln-before.onnx")
    torch.onnx.export(
            model         = model,
            args          = (input,),
            f             = file,
            input_names   = ["input0"],
            output_names  = ["output0"],
            opset_version = 17)

    print("\nFinished export {}".format(onnx_path))

    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)

    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, onnx_path)



@gs.Graph.register()
def layerNorm(self, inputs, outputs, axis, epsilon):
    attrs = {'axis': np.int64(axis), 'epsilon': np.float(epsilon)}
    return self.layer(op="LayerNormalization", inputs=inputs, outputs=outputs, attrs=attrs)

def change_onnx_graph(onnx_path, export_path):
    graph = gs.import_onnx(onnx.load_model(onnx_path))
    tensors = graph.tensors()

    norm_scale = gs.Constant(name="norm.weight", values=np.ones(shape=[3], dtype=np.float32))
    norm_bias  = gs.Constant(name="norm.bias", values=np.zeros(shape=[3], dtype=np.float32))

    inputs  = [tensors["/Transpose_output_0"]]
    outputs = [tensors["/norm/Div_output_0"]]
    
    # 因为要替换子网，所以需要把子网和周围的所有节点都断开联系
    for item in inputs:
        item.outputs.clear()

    for item in outputs:
        item.inputs.clear()

    # 为了迎合onnx中operator中的设计，这里把scale和bias给加上
    inputs = [tensors["/Transpose_output_0"],
              norm_scale,
              norm_bias]
    
    # 这个onnx中的epsilon，我们给加上。当然，我们也可以选择默认的值
    epsilon = [tensors["/norm/Constant_1_output_0"]]
    print(type(epsilon[0].values))

    # 通过注册的LayerNorm，重新把断开的联系链接起来
    graph.layerNorm(inputs, outputs, axis=-1, epsilon=epsilon[0].values)
    # graph.identity(inputs, outputs)
    # graph.layerNorm_default(inputs, outputs)

    # 删除所有额外的节点
    graph.cleanup()

    onnx.save(gs.export_onnx(graph), export_path)



if __name__ == "__main__":
    # print(os.path.abspath(os.path.dirname(__file__)))
    now = datetime.datetime.now()
    dir_path_name = os.path.abspath(os.path.dirname(__file__))
    file = os.path.join(dir_path_name, f"layernorm_{now.strftime('%Y%m%d_%H%M%S')}.onnx")
    export_path = os.path.join(dir_path_name, f"layernorm_{now.strftime('%Y%m%d_%H%M%S')}_fuse.onnx")
    # export_onnx_graph(file)
    
    change_onnx_graph("/mnt/share_disk/cdd/Pytorch_learning/ONNX_Operator_Vis/LayerNorm/layernorm_20240202_185911.onnx", export_path)