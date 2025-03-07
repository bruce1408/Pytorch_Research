import tensorrt as trt

# 自定义日志记录器
class MyLogger(trt.Logger):
    def __init__(self, verbosity=trt.Logger.VERBOSE):
        super().__init__(verbosity)

    def log(self, severity, msg):
        if severity >= self.verbosity:
            print(msg)

# 创建 TensorRT Logger 对象
logger = MyLogger()

# 创建 builder 和网络定义
builder = trt.Builder(logger)
network = builder.create_network()

# 载入 ONNX 模型
onnx_file = '/home/bruce_ultra/workspace/8620_code_repo/8620_code_x86/onnx_models/od_bev_250220.onnx'
onnx_file = "/home/bruce_ultra/workspace/8620_code_repo/8620_code_x86/onnx_models/od_bev_0306.onnx"
onnx_parser = trt.OnnxParser(network, logger)
with open(onnx_file, 'rb') as f:
    onnx_parser.parse(f.read())

# 打印日志
print("TensorRT Logs:")
