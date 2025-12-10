import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =========================================================================================
# 辅助函数与环境设置
# =========================================================================================

def randomSeed(SEED):
    """
    设置随机种子，以确保实验结果的可复现性。
    在机器学习中，很多操作（如权重初始化）都是随机的，固定种子能保证我们每次运行
    代码时，这些随机过程都产生相同的结果，方便调试和比较。
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # 如果使用GPU，也需要为CUDA设置种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    # 确保CUDA使用确定性的卷积算法，进一步保证一致性
    torch.backends.cudnn.deterministic = True

# 设置一个全局的种子值
SEED = 1234
randomSeed(SEED)

# 检查是否有可用的GPU (NVIDIA显卡)，如果有就使用GPU，否则使用CPU。
# GPU可以极大地加速神经网络的训练。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =========================================================================================
# 1. 数据准备与预处理
# =========================================================================================

# 定义我们的训练数据。这是一个包含三元组的列表：
# [源语言句子(德语), 目标语言句子(英语)作为解码器输入, 真实的目标语言句子作为标签]
sentences = [['ich mochte ein bier', 'S i want a beer', 'i want a beer E']]

# --- 词典 (Vocabulary) ---
# 将每个单词映射到一个唯一的整数ID。计算机只能处理数字，所以这是必须的步骤。
# 'P' (Padding) 是一个特殊的填充符号，ID为0，用于将长短不一的句子补齐。
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
src_vocab_size = len(src_vocab)  # 源语言词典的大小

tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
# 创建一个反向词典，方便之后将数字ID转换回单词，用于查看模型预测结果。
number_dict = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)  # 目标语言词典的大小

# 定义序列的最大长度。在真实任务中，这通常会根据数据集动态计算。
src_len = 4 # 'ich mochte ein bier' -> 4个词
tgt_len = 5 # 'S i want a beer' -> 5个词

# =========================================================================================
# 2. 模型超参数定义 (Hyperparameters)
# =========================================================================================
# 超参数是我们在训练前手动设置的数值，它们控制了模型的结构和学习过程。
# 注：对于只有一个句子的“玩具”数据集，这些参数其实太大了，但我们保留
# 原始论文的尺寸来学习其结构。
voc_dim = 512      # 词嵌入(Embedding)的维度。每个词将被表示为一个512维的向量。
d_ff = 2048        # 前馈神经网络(FFN)的隐藏层维度。
d_k = d_v = 64     # Query, Key, Value向量的维度。
n_layers = 6       # 编码器和解码器中堆叠的层数。
n_heads = 8        # 多头注意力机制中的“头”数。代表从8个不同的角度去理解句子。

# =========================================================================================
# 3. 工具函数
# =========================================================================================

def make_batch(sentences):
    """
    将原始的字符串句子数据，转换成模型需要的数字ID张量(Tensor)格式，并移动到指定设备(GPU或CPU)。
    """
    input_batch = [[src_vocab[n] for n in sentences[0][0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[0][1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[0][2].split()]]
    
    # 使用 .to(device) 将张量发送到之前检测到的设备上
    return torch.LongTensor(input_batch).to(device), \
           torch.LongTensor(output_batch).to(device), \
           torch.LongTensor(target_batch).to(device)

def get_sinusoid_encoding_table(n_position, voc_dim):
    """
    生成正弦/余弦位置编码表。它为句子中的每个位置创建一个独特的、固定的向量，
    让模型能够理解单词的顺序。
    """
    def cal_angle(position, hid_idx):
        # 这是位置编码的数学公式
        return position / np.power(10000, 2 * (hid_idx // 2) / voc_dim)
    
    def get_posi_angle_vec(position):
        # 对于一个给定的位置，计算出它对应的voc_dim维的角度向量
        return [cal_angle(position, hid_j) for hid_j in range(voc_dim)]

    # 为所有可能的位置 (0 到 n_position-1) 生成角度表
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    # 根据公式，偶数维度使用sin函数
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    # 奇数维度使用cos函数
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).to(device)

def get_attn_pad_mask(seq_q, seq_k):
    """
    生成填充掩码(Padding Mask)，用于在注意力计算中忽略填充部分('P')。
    当我们计算注意力时，我们不希望模型关注到这些无意义的填充词。
    """
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    
    # .eq(0)会找到所有ID为0（即'P'）的位置，并标记为True
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # 形状: [batch_size, 1, len_k]
    
    # 将掩码扩展，使其维度和注意力分数矩阵(len_q, len_k)匹配
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequent_mask(seq):
    """
    生成后续掩码(Subsequent Mask)，用于防止解码器在预测时“偷看”未来的信息。
    这是解码器自注意力机制的核心，确保模型在预测第i个词时，只能看到第i个词和它之前的词。
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # np.triu(..., k=1) 创建一个上三角矩阵，对角线(k=0)以上的部分为1，其余为0。
    # 值为1的地方就是要被遮盖的“未来”位置。
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    return torch.from_numpy(subsequent_mask).byte().to(device)

# =========================================================================================
# 4. Transformer 模型组件定义
# =========================================================================================

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制的实现"""
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # 1. 计算Q和K的点积，得到原始的注意力分数。
        # 除以sqrt(d_k)是为了进行缩放，防止梯度消失。
        # 用点积来衡量 Query 和 Key 之间的匹配程度
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # [1,8,4,64] * [1,8,64,4] = [1,8,4,4]
        
        # 2. 应用掩码，将所有需要被遮盖的位置填充一个极小的负数。
        #    这样在经过Softmax后，这些位置的权重会趋近于0。
        scores.masked_fill_(attn_mask, -1e9)  # [1,8,4,4] -> [1,8,4,4]
        
        # 3. 对分数进行Softmax运算，得到0到1之间的注意力权重。
        attn = nn.Softmax(dim=-1)(scores)
        
        # 4. 将注意力权重与V相乘，得到加权的上下文向量。
        context = torch.matmul(attn, V)     # [1,8,4,4] * [1,8,4,64] = [1,8,4,64]
        return context, attn

class MultiHeadAttention(nn.Module):
    """多头注意力机制的实现"""
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(voc_dim, d_k * n_heads)
        self.W_K = nn.Linear(voc_dim, d_k * n_heads)
        self.W_V = nn.Linear(voc_dim, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, voc_dim)
        self.layer_norm = nn.LayerNorm(voc_dim)

    def forward(self, Q, K, V, attn_mask):
        # residual用于残差连接，batch_size用于重塑张量
        residual, batch_size = Q, Q.size(0)  # Q shape [1,4,512], attn_mask shape [1,4,4]
        
        # 1. 将Q, K, V通过线性层，并分割成多个头。
        # view和transpose操作是为了将 [batch, len, voc_dim] 变为 [batch, n_heads, len, d_k]，即[1,4,512] -> [1,8,4,64]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        
        # 2. 为多头准备掩码，将其扩展一个维度以匹配q_s, k_s, v_s的形状。
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        # 3. 计算缩放点积注意力，为每个头独立计算。
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        
        # 4. 合并多头的输出。
        #    transpose和contiguous().view()是高效重塑张量的标准操作。
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        
        # 5. 通过最后的线性层进行整合。
        output = self.linear(context)
        
        # 6. 进行残差连接和层归一化，这是Transformer的重要技巧，有助于稳定训练。
        return self.layer_norm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    """逐位置前馈网络(FFN)的实现"""
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        
        # 实际上就是两个线性层，但用一维卷积可以更高效地实现。
        self.conv1 = nn.Conv1d(in_channels=voc_dim, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=voc_dim, kernel_size=1)
        self.layer_norm = nn.LayerNorm(voc_dim)

    def forward(self, inputs):
        residual = inputs
        
        # 卷积层需要 [batch, channels, length] 的输入，所以需要转置。
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2) # 计算后转置回来
        
        # 同样进行残差连接和层归一化。
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    """单个编码器层的实现，由一个多头自注意力和一个前馈网络组成。"""
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        
        # Q, K, V都来自编码器的输入，这被称为“自”注意力。
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    """单个解码器层的实现，比编码器层多一个用于与编码器交互的注意力层。"""
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()  # 解码器自注意力
        self.dec_enc_attn = MultiHeadAttention()   # 编码器-解码器注意力
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        
        # 1. 解码器自注意力，Q,K,V都来自解码器，且使用后续掩码防止“作弊”。
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        
        # 2. 编码器-解码器注意力，Q来自解码器，K和V来自编码器的最终输出。
        # 这是解码器“看”源语言句子的地方。
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    """完整的编码器，由词嵌入、位置编码和N个编码器层堆叠而成。"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, voc_dim)
        
        # 加载预计算好的正余弦位置编码表，并设为不可训练(freeze=True)。
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len + 1, voc_dim), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs): # enc_inputs: [batch_size, src_len]
        # 1. 动态生成从0开始的位置索引。
        seq_len = enc_inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand_as(enc_inputs)
        
        # 2. 将词嵌入和位置编码相加，得到最终的输入表示。
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(pos)
        
        # 3. 生成填充掩码。
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        
        enc_self_attns = []
        # 4. 逐层通过编码器。
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn) # 保存注意力权重用于后续分析
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    """完整的解码器，由词嵌入、位置编码和N个解码器层堆叠而成。"""
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, voc_dim)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len + 1, voc_dim), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs, dec_inputs, enc_outputs):
        # 1. 动态生成解码器输入的位置索引。
        seq_len = dec_inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand_as(dec_inputs)
        
        # 2. 将词嵌入和位置编码相加。
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos)
        
        # 3. 生成解码器自注意力的填充掩码和后续掩码，并合并它们。
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        
        # 用于防止解码器在预测时“偷看”未来的信息
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        
        # 最终屏蔽掉padding 和 未来的信息，得到自注意力掩码
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0) # 大于0返回True, 否则返回False

        # 4. 生成编码器-解码器注意力的填充掩码。
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        
        # 5. 逐层通过解码器。
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    """完整的Transformer模型，由一个编码器、一个解码器和一个最终的投影层组成。"""
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # 最后的线性层，将解码器的输出映射到目标词典的大小，得到每个词的得分(logits)。
        self.projection = nn.Linear(voc_dim, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        
        # 1. 将源句子通过编码器。
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)  # enc_outputs shape [1,4,512], enc_self_attns[0] shape [1,8,4,4]
        
        # 2. 将编码器输出和目标句子（解码器输入）一起通过解码器。
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(enc_inputs, dec_inputs, enc_outputs)
        
        # 3. 将解码器的最终输出通过投影层，得到预测得分。
        dec_logits = self.projection(dec_outputs)
        
        # 返回最终的 logits 和各层的注意力权重（用于分析和可视化）。
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

def showgraph(attn, name):
    """ 可视化注意力权重的函数 """

    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.cpu().numpy()
    fig = plt.figure(figsize=(n_heads, n_heads))  # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    
     # --- 核心修改在这里 ---
    # 从新的 sentences 结构中正确索引到字符串
    source_sentence = sentences[0][0]
    target_sentence = sentences[0][2]
    
    ax.set_xticks(range(len(source_sentence.split()) + 1))
    ax.set_yticks(range(len(target_sentence.split()) + 1))
    
    # 使用正确的字符串变量进行 split
    ax.set_xticklabels([''] + source_sentence.split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + target_sentence.split(), fontdict={'fontsize': 14})
    
    # plt.show()
    plt.savefig(f"attn_{name}.png")
    
    
# =========================================================================================
# 5. 训练与测试
# =========================================================================================
if __name__ == "__main__":
    # 实例化模型并移动到指定设备
    model = Transformer().to(device)
    
    # 定义损失函数。CrossEntropyLoss内部包含了Softmax，所以模型输出原始logits即可。
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器。Adam是一种常用的、效果很好的优化算法。
    # 降低学习率(lr)是一个重要的技巧，有助于模型稳定训练。
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("Start Training...")
    
    # 训练循环
    for epoch in range(50): # 增加训练轮数以保证收敛
        optimizer.zero_grad() # 每个epoch开始前，清空上一轮的梯度
        # 准备一个批次的数据
        enc_inputs, dec_inputs, target_batch = make_batch(sentences)
        
        # 模型前向传播
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        
        # 计算损失
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        
        # 每5个epoch打印一次损失，方便观察训练进程
        if (epoch + 1) % 5 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        
        # 反向传播，计算梯度
        loss.backward()
        
        # 更新模型参数
        optimizer.step()

    # --- 测试 ---
    print("\nTesting...")
    
    # 准备测试数据
    enc_inputs, dec_inputs, _ = make_batch(sentences)
    
    # 模型预测，在测试阶段我们不关心注意力权重
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    
    # .max(1, keepdim=True)[1] 找到每个位置得分最高的那个词的ID
    predict = predict.data.max(1, keepdim=True)[1]
    
    
    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns, "enc_self_attns")

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns, "dec_self_attns")

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns, "dec_enc_attns")
