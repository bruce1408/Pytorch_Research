import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from spectrautils.common_utils import enter_workspace
enter_workspace()


os.environ['CUDA_VISIBLES_DEVICES'] = '0'
if torch.cuda.is_available():
    print("multi cuda")
print("torch version: ", torch.__version__)

dtype = torch.FloatTensor
# S: 解码器输入开始的标志
# E: 解码器输出结束的标志
# P: 填充标志，用于将序列补齐到相同长度

# ==========================================================================================
# 1. 数据准备与预处理
# ==========================================================================================

# 准备一个用于训练的句子对
# 格式: [德语源句, 英语目标句(解码器输入), 英语目标句(真实标签)]
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
# sentences = [
#     # enc_input           dec_input         dec_output
#     ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
#     ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E'],
#     ['ich mochte ein fanta P', 'S i want a fanta .', 'i want a fanta . E']
# ]


# --- 源语言 (德语) 词典 ---
# 'P' (Padding) 的索引为 0，这在后续处理掩码时很重要
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
src_vocab_size = len(src_vocab)

# --- 目标语言 (英语) 词典 ---
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
number_dict = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)
print(number_dict)
# 定义源序列和目标序列的最大长度
src_len = 5
tgt_len = 5

# ==========================================================================================
# 2. 模型超参数定义
# ==========================================================================================
d_model = 512      # 词嵌入的维度 (Embedding Size)
d_ff = 2048        # 前馈神经网络的隐藏层维度
d_k = d_v = 64     # Q, K, V 向量的维度
n_layers = 6       # 编码器和解码器层的数量
n_heads = 8        # 多头注意力机制中的头数


# ==========================================================================================
# 3. 工具函数
# ==========================================================================================

# 设置随机种子以保证实验结果的可复现性
def randomSeed(SEED):
    seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


SEED = 1234
randomSeed(SEED)


def make_batch(sentences):
    """
    将原始句子数据转换成模型需要的张量 (Tensor) 格式。
    """
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    
    # 将 Python 列表转换为 PyTorch 的 LongTensor
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


def get_sinusoid_encoding_table(n_position, d_model):
    """
    生成正弦/余弦位置编码表。
    这是 Transformer 论文中提出的位置编码方法。
    """
    def cal_angle(position, hid_idx):
        # 计算角度
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        
        # 计算一个位置上所有维度的角度
        cal_angle_list = []
        for hid_j in range(d_model):
            cal_angle_list.append(cal_angle(position, hid_j))
        return cal_angle_list
        # return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    # 生成位置编码
    # sinusoid_table = np.array([get_posi_angle_vec(pos_i)
    #                            for pos_i in range(n_position)])  # [6, 512]
    
    sin_list = []
    for pos_i in range(n_position):
        sin_list.append(get_posi_angle_vec(pos_i))
        
    sinusoid_table = np.array(sin_list)
    # 偶数维度使用sin函数
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i 偶数列
    
    # 奇数维度使用cos函数
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1 奇数列
    return torch.FloatTensor(sinusoid_table)


def get_attn_pad_mask(seq_q, seq_k):
    """
    生成填充掩码 (Padding Mask)，用于在注意力计算中忽略填充部分 'P'。
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # eq(zero) is PAD token，因为k的pad没有任何意义，所以这里不计算q为pad的情况
    # [batch_size, 1, len_k] one is masking
    # 如果 seq_k 中的一个词是填充（索引为0），那么它的值为 True
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)

    # 这里维度再进行扩展一下[batch, 1, len_k] -> [batch, len_q, len_k] = [1, 5, 5] 重复复制
    # 将掩码扩展，使其维度和注意力分数矩阵匹配
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequent_mask(seq):
    """
    生成子序列掩码 (Subsequent Mask)，用于防止解码器在预测时看到未来的信息。
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # 创建一个上三角矩阵，对角线以上都是1
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


# ==========================================================================================
# 4. Transformer 模型组件
# ==========================================================================================
# ----------------- 4.1 注意力机制核心 -----------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q = [1, 8, 5, 64], attn_mask = [1, 8, 5, 5]
        # scores : [batch_size, n_heads, len_q, len_k]
        # 1. 计算 Q 和 K 的点积，并除以维度的平方根进行缩放
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # Fills elements of self tensor with value where mask is one. 填充mask 是1的地方换成很小的一个数字，这样softmax之后就是0，可以忽略
        # 2. 应用掩码，将需要忽略的位置填充一个极小值
        scores.masked_fill_(attn_mask, -1e9)
        # 3. 应用 softmax 函数，得到注意力权重
        attn = nn.Softmax(dim=-1)(scores)  # [1, 8, 5, 5] 和 attn_mask 没变
        # 4. 将权重与 V 相乘，得到加权的上下文向量
        context = torch.matmul(attn, V)  # [1, 8, 5, 64] 和 Q 没变
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        
        # 定义用于生成 Q, K, V 的线性层
        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 512, 64 * 8 = 512
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 512, 64 * 8 = 512
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 512, 64 * 8 = 512
        # 注意：这里的 fc 和 layer_norm 应该在 __init__ 中定义一次
        # 在 forward 中反复定义会很低效
        # self.fc = nn.Linear(n_heads * d_v, d_model)
        # self.layer_norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(n_heads * d_v, d_model) # <--- 移到这里，并定义为类属性
        self.layer_norm = nn.LayerNorm(d_model) # LayerNorm 也建议放在 init 里

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)

        # (B, S, D) proj-> (B, S, D) -split-> [B, S, H, W] -trans-> [B, H, src_len, W] = [1, 8, 5, 64]
        # q_s:[batch_size, n_heads, len_q, d_k]
        # 1. 将输入 Q, K, V 通过线性层，并分割成多个头
        # [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # k_s:[batch_size, n_heads, len_k, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # v_s:[batch_size, n_heads, len_k, d_v]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # attention mask 计算部分 [batch_size, n_heads, len_q, len_k] = [1, 8, 5, 5]
        # 2. 为多头准备掩码
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        # 3. 计算缩放点积注意力
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # context:[batch_size, len_q, n_heads, d_v] = [1, 5, 8, 64]
        # 4. 合并多头的输出
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, n_heads * d_v)  # [1, 5, 8*64 = 512]
        
        # 5. 通过最后的线性层
        # 【注意】在 forward 中创建 nn.Module 是不推荐的，会很低效
        # output = nn.Linear(n_heads * d_v, d_model)(context)
        output = self.linear(context) # <--- 这里调用 self.linear

        # output: [batch_size x len_q x d_model]
        # 6. 进行残差连接和层归一化
        # 【注意】 LayerNorm 也应该在 __init__ 中定义
        # return nn.LayerNorm(d_model)(output + residual), attn
        return self.layer_norm(output + residual), attn # 使用 init 定义的 LayerNorm
    


class PoswiseFeedForwardNet(nn.Module):
    """
    逐位置前馈网络。这里使用两个一维卷积来实现，这是一种等效且高效的实现方式。
    """
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        
        # 1. 第一个卷积（线性变换）+ ReLU 激活
        # inputs.transpose(1, 2) -> [batch_size, d_model, seq_len]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        
        # 2. 第二个卷积（线性变换）
        output = self.conv2(output).transpose(1, 2)
        
        # 3. 残差连接和层归一化
        return nn.LayerNorm(d_model)(output + residual)


class EncoderLayer(nn.Module):
    """ 单个编码器层 """
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  # 自注意力层
        self.pos_ffn = PoswiseFeedForwardNet()     # 前馈网络层

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs to same Q, K, V = [batch, src_len], enc_self_attn_mask = [batch, q_len, k_len]
        # 1. 计算自注意力 (Q, K, V都来自编码器输入), enc_outputs = [batch, src_len, d_model]
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)


        # 2. 通过前馈网络
        enc_outputs = self.pos_ffn(enc_outputs)  # [batch_size, len_q, d_model]

        return enc_outputs, attn


class DecoderLayer(nn.Module):
    """ 单个解码器层 """
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_enc_attn = MultiHeadAttention()  # 编解码器注意力层
        self.dec_self_attn = MultiHeadAttention() # 解码器自注意力层
        self.pos_ffn = PoswiseFeedForwardNet()    # 前馈网络层

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # 1. 计算带掩码的解码器自注意力 (Q, K, V都来自解码器输入)
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        # decoder 和 encoder 级联部分的柱注意力机制
        # 2. 计算编解码器注意力 (Q来自解码器，K和V来自编码器)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)

        # 3. 通过前馈网络
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    """ 完整的编码器 """
    def __init__(self):
        super(Encoder, self).__init__()
        # 源语言词嵌入层

        # 词向量
        self.src_emb = nn.Embedding(src_vocab_size, d_model)

        # get_sinusoid_encoding_table 返回的是一个[6, 512]的torch的tensor
        # 位置编码层，这里通过加载预计算的表来实现
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(src_len + 1, d_model), freeze=True)
        
        # 堆叠 N 个编码器层
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):  # enc_inputs : [batch_size x source_len]

        # pos是按照索引取值, 可以直接换成enc_inputs, enc_embed = [1, 5, 512]
        # 1. 词嵌入 + 位置编码
        seq_len = enc_inputs.size(1)
        pos_tensor = torch.arange(1, seq_len + 1, dtype=torch.long, device=enc_inputs.device)
        pos_tensor = pos_tensor.unsqueeze(0).expand_as(enc_inputs)
        pos_tensor = pos_tensor.masked_fill(enc_inputs.eq(0), 0)
        enc_embed = self.src_emb(enc_inputs) + self.pos_emb(pos_tensor)

        # enc_self_attn_mask = [batch_size, len_q, len_k]
        # 2. 生成填充掩码
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        # 3. 逐层通过编码器
        enc_self_attns = []
        for layer in self.layers:
            enc_embed, enc_self_attn = layer(enc_embed, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_embed, enc_self_attns


class Decoder(nn.Module):
    """ 完整的解码器 """
    def __init__(self):
        super(Decoder, self).__init__()
        # 目标语言词嵌入层
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        # 位置编码层
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len + 1, d_model), freeze=True)
        # 堆叠 N 个解码器层
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs, dec_inputs, enc_outputs):
        # 1. 词嵌入 + 位置编码
        # 【注意】这里位置编码的索引也是硬编码的！
        # dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5, 1, 2, 3, 4]]))
        seq_len = dec_inputs.size(1)
        # 生成 0, 1, 2... 的位置索引，并确保它在 GPU 上(如果使用了cuda)
        pos_tensor = torch.arange(0, seq_len , dtype=torch.long, device=dec_inputs.device)
        pos_tensor = pos_tensor.unsqueeze(0).expand_as(dec_inputs)
        pos_tensor = pos_tensor.masked_fill(dec_inputs.eq(0), 0)

        # print(dec_inputs, pos_tensor)
        # dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos_tensor)
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5, 1, 2, 3, 4]]))

    
        # 2. 生成解码器自注意力的填充掩码
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # 3. 生成解码器自注意力的子序列掩码
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        # 4. 合并两种掩码
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        # 5. 生成编解码器注意力的填充掩码
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        # 6. 逐层通过解码器
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns



class Transformer(nn.Module):
    """ 完整的 Transformer 模型 """
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # 最后的线性层，将解码器输出映射到目标词表大小
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        # 1. 通过编码器
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # 2. 通过解码器
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(enc_inputs, dec_inputs, enc_outputs)
        # 3. 投影到词汇表空间
        dec_logits = self.projection(dec_outputs)
        # 返回最终的 logits 和各层的注意力权重（用于分析和可视化）
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


def showgraph(attn, name):
    """ 可视化注意力权重的函数 """

    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads))  # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticks(range(len(sentences[0].split()) + 1)) # <--- 1. 添加这行来设置X轴刻度位置
    ax.set_yticks(range(len(sentences[2].split()) + 1)) # <--- 2. 添加这行来设置Y轴刻度位置
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    # plt.show()
    plt.savefig(f"attn_{name}.png")
    

if __name__ == "__main__":
    model = Transformer()
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ========================== 训练过程 ==========================
    for epoch in range(20):
        optimizer.zero_grad()
        enc_inputs, dec_inputs, target_batch = make_batch(sentences)
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()


    
    # ========================== 测试(推理) ==========================
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns, "enc_self_attns")

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns, "dec_self_attns")

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns, "dec_enc_attns")

