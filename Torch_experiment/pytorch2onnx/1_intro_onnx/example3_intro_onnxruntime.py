import torch 
from torch import nn 
from torch.nn.functional import interpolate 
import torch.onnx 
import cv2 
import numpy as np 


class NewInterpolate(torch.autograd.Function): 
    """
    内部定义插值算子，需要定义两个静态方法(@staticmethod):forward & backward
    算子的推理行为由算子的 foward 方法决定。该方法的第一个参数必须为 ctx，后面的参数为算子的自定义输入，
    我们设置两个输入，分别为被操作的图像和放缩比例。
    为保证推理正确，需要把 [1, 1, w, h] 格式的输入对接到原来的 interpolate 函数上。
    我们的做法是截取输入张量的后两个元素，把这两个元素以 list 的格式传入 interpolate 的 scale_factor 参数。
    
    接下来，我们要决定新算子映射到 ONNX 算子的方法。映射到 ONNX 的方法由一个算子的 symbolic 方法决定。
    symbolic 方法第一个参数必须是g，之后的参数是算子的自定义输入，和 forward 函数一样。
    ONNX 算子的具体定义由 g.op 实现。g.op 的每个参数都可以映射到 ONNX 中的算子属性：
    """
 
    @staticmethod 
    def symbolic(g, input, scales): 
        return g.op("Resize", 
                    input, 
                    g.op("Constant", value_t=torch.tensor([], dtype=torch.float32)), 
                    scales, 
                    coordinate_transformation_mode_s="pytorch_half_pixel", 
                    cubic_coeff_a_f=-0.75, 
                    mode_s='cubic', 
                    nearest_mode_s="floor") 
 
    @staticmethod 
    def forward(ctx, input, scales): 
        scales = scales.tolist()[-2:] 
        return interpolate(input, 
                           scale_factor=scales, 
                           mode='bicubic', 
                           align_corners=False) 
 
 
class StrangeSuperResolutionNet(nn.Module): 
 
    def __init__(self): 
        super().__init__() 
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4) 
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0) 
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2) 
 
        self.relu = nn.ReLU() 
 
    def forward(self, x, upscale_factor): 
        x = NewInterpolate.apply(x, upscale_factor) 
        out = self.relu(self.conv1(x)) 
        out = self.relu(self.conv2(out)) 
        out = self.conv3(out) 
        return out 
 
 
def init_torch_model(): 
    torch_model = StrangeSuperResolutionNet() 
 
    state_dict = torch.load('../../models/srcnn.pth')['state_dict'] 
 
    # Adapt the checkpoint 
    for old_key in list(state_dict.keys()): 
        new_key = '.'.join(old_key.split('.')[1:]) 
        state_dict[new_key] = state_dict.pop(old_key) 
 
    torch_model.load_state_dict(state_dict) 
    torch_model.eval() 
    return torch_model 
 

def prepare_model():
    model = init_torch_model() 
    factor = torch.tensor([1, 1, 3, 3], dtype=torch.float) 
    
    input_img = cv2.imread('../../models/face.png').astype(np.float32) 
    
    # HWC to NCHW 
    input_img = np.transpose(input_img, [2, 0, 1]) 
    input_img = np.expand_dims(input_img, 0) 
    
    # Inference 
    torch_output = model(torch.from_numpy(input_img), factor).detach().numpy() 
    
    # NCHW to HWC 
    torch_output = np.squeeze(torch_output, 0) 
    torch_output = np.clip(torch_output, 0, 255) 
    torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8) 
    
    # Show image 
    cv2.imwrite("face_torch_3.png", torch_output) 
    
    return model, input_img, torch_output
    
 
def export_onnx_model():
    x = torch.randn(1, 3, 256, 256) 
    with torch.no_grad(): 
        torch.onnx.export(model, (x, factor), 
                        "srcnn3.onnx", 
                        opset_version=11, 
                        input_names=['input', 'factor'], 
                        output_names=['output']) 


def onnxruntime():
    # 这里把3倍的上采样改成4倍，也能运行；
    import onnxruntime 
    input_factor = np.array([1, 1, 4, 4], dtype=np.float32) 
    ort_session = onnxruntime.InferenceSession("../../models/srcnn3.onnx") 
    ort_inputs = {'input': input_img, 'factor': input_factor} 
    ort_output = ort_session.run(None, ort_inputs)[0] 
    
    ort_output = np.squeeze(ort_output, 0) 
    ort_output = np.clip(ort_output, 0, 255) 
    ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8) 
    cv2.imwrite("face_ort_3.png", ort_output)
    

if __name__ == "__main__":
    # export_onnx_model()
    
    model, input_img, torch_output = prepare_model()
    # 绕过 PyTorch 本身的限制，凭空“捏”出了一个 ONNX 算子。
    
    
    # 事实上不仅可以创建现有的 ONNX 算子，还可以定义新的 ONNX 算子以拓展 ONNX 的表达能力。
    onnxruntime()