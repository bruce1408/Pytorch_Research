import torch
import torch.nn as nn

# 1. 定义参数
batch_size = 2
seq_len = 5
dim_in = 4
dim_out = 8

# 2. 创造输入数据 [Batch, Length, Dim]
input_tensor = torch.randn(batch_size, seq_len, dim_in)

# -------------------------------------------------
# 方式 A: 使用 nn.Linear
# -------------------------------------------------
linear_layer = nn.Linear(dim_in, dim_out)
# Linear 直接吃 [2, 5, 4]
out_linear = linear_layer(input_tensor) 
print(f"Linear Output Shape: {out_linear.shape}") 
# -> [2, 5, 8]

# -------------------------------------------------
# 方式 B: 使用 nn.Conv1d (模拟 Linear)
# -------------------------------------------------
# 为了让 Conv1d 模拟 Linear，我们需要把 Linear 的权重复制给 Conv1d
conv_layer = nn.Conv1d(dim_in, dim_out, kernel_size=1)

# 手动把 Linear 的权重赋值给 Conv1d (注意 shape 对应关系)
# Linear weight: [dim_out, dim_in]
# Conv1d weight: [dim_out, dim_in, 1] -> 需要 unsqueeze 增加一维
with torch.no_grad():
    conv_layer.weight.copy_(linear_layer.weight.unsqueeze(-1))
    conv_layer.bias.copy_(linear_layer.bias)

# 【关键步骤】Conv1d 需要输入是 [Batch, Dim, Length]，所以需要转置
input_transposed = input_tensor.transpose(1, 2) # [2, 4, 5]

# 进行卷积
out_conv = conv_layer(input_transposed) # [2, 8, 5]

# 【关键步骤】把结果转置回来，变回 [Batch, Length, Dim]
out_conv = out_conv.transpose(1, 2) # [2, 5, 8]

print(f"Conv1d Output Shape: {out_conv.shape}")

# -------------------------------------------------
# 验证结果是否完全一致
# -------------------------------------------------
is_same = torch.allclose(out_linear, out_conv, atol=1e-6)
print(f"Are results identical? {is_same}") 
# -> True