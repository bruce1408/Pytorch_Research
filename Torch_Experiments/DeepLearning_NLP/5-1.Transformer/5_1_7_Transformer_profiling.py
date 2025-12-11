import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time  # >>> NEW: 用于计时
from spectrautils.print_utils import print_colored_text

# ==========================================
# 1. 基础配置 (已瘦身版)
# ==========================================

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据集
sentences = [['ich mochte ein bier', 'S i want a beer', 'i want a beer E']]

# 词典
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
number_dict = {i: w for i, w in enumerate(tgt_vocab)}

src_len = 4
tgt_len = 5

# ---【关键修改】超参数瘦身 ---
d_model = 64   # 原 512 -> 改为 64
d_ff = 256     # 原 2048 -> 改为 256
d_k = d_v = 32 # d_model / n_heads = 32
n_layers = 2   # 原 6 -> 改为 2
n_heads = 2    # 原 8 -> 改为 2

# ==========================================
# 2. 工具函数 (保持不变)
# ==========================================
def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0][0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[0][1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[0][2].split()]]
    return torch.LongTensor(input_batch).to(device), \
           torch.LongTensor(output_batch).to(device), \
           torch.LongTensor(target_batch).to(device)

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).to(device)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    return seq_k.data.eq(0).unsqueeze(1).expand(batch_size, len_q, len_k)

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    return torch.from_numpy(np.triu(np.ones(attn_shape), k=1)).byte().to(device)

# ==========================================
# 3. 模型组件 (带 KV Cache 逻辑)
# ==========================================

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask, layer_cache=None, type=None):
        residual, batch_size = Q, Q.size(0)
        
        # 1. 投影
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        
        # Cross Attention Cache 复用逻辑
        if layer_cache is not None and type == "cross" and layer_cache["k"] is not None:
            k_s = layer_cache["k"]
            v_s = layer_cache["v"]
        else:
            k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
            v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # 2. KV Cache 更新逻辑
        if layer_cache is not None:
            if type == "self":
                # 自注意力：当前K拼接历史K
                if layer_cache["k"] is not None:
                    k_s = torch.cat([layer_cache["k"], k_s], dim=2)
                    v_s = torch.cat([layer_cache["v"], v_s], dim=2)
                layer_cache["k"] = k_s
                layer_cache["v"] = v_s
            elif type == "cross":
                # 交叉注意力：保存一次即可
                if layer_cache["k"] is None:
                    layer_cache["k"] = k_s
                    layer_cache["v"] = v_s

        # 3. 计算注意力
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        # 你也可以换成 Linear，这里保持原样
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask, layer_cache=None):
        self_cache = layer_cache['self'] if layer_cache else None
        cross_cache = layer_cache['cross'] if layer_cache else None

        dec_outputs, dec_self_attn = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask, 
            layer_cache=self_cache, type="self"
        )
        dec_outputs, dec_enc_attn = self.dec_enc_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask, 
            layer_cache=cross_cache, type="cross"
        )
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(len(src_vocab), d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len + 1, d_model), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        pos = torch.arange(enc_inputs.size(1), device=device).unsqueeze(0).expand_as(enc_inputs)
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(pos)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        for layer in self.layers:
            enc_outputs, _ = layer(enc_outputs, enc_self_attn_mask)
        return enc_outputs

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(len(tgt_vocab), d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(100, d_model), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs, past_cache=None, step=None):
        if step is not None:
            pos = torch.tensor([step], dtype=torch.long, device=device)
        else:
            pos = torch.arange(dec_inputs.size(1), device=device).unsqueeze(0).expand_as(dec_inputs)
            
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos)

        if past_cache is None:
            # 训练模式：需要Mask
            dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
            dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)  # 上三角矩阵
            dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
            dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # 交叉注意力Mask
        else:
            # 推理模式：不需要Mask (Batch size=1)
            dec_self_attn_mask = None
            dec_enc_attn_mask = None

        if past_cache is None and step is not None:
            past_cache = [{'self': {'k': None, 'v': None}, 'cross': {'k': None, 'v': None}} for _ in range(n_layers)]

        for i, layer in enumerate(self.layers):
            layer_cache = past_cache[i] if past_cache else None
            dec_outputs, _, _ = layer(
                dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask, layer_cache=layer_cache
            )
            
        return dec_outputs, past_cache

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, len(tgt_vocab), bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs, _ = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1))

    # 原来的 inference（保留方便看结果）
    def inference(self, enc_input_tensor):
        enc_outputs = self.encoder(enc_input_tensor)
        dec_input = torch.zeros(1, 1).type_as(enc_input_tensor.data).fill_(tgt_vocab['S'])
        
        cache = None
        generated_ids = []
        print(f"--- Start Inference (Source: {sentences[0][0]}) ---")
        
        for i in range(10):
            dec_outputs, cache = self.decoder(
                dec_input, enc_input_tensor, enc_outputs, past_cache=cache, step=i
            )
            projected = self.projection(dec_outputs)
            next_word_id = projected.squeeze(0).max(dim=-1)[1].item()
            
            print(f"Step {i}: Input='{number_dict[dec_input.item()]}' -> Pred='{number_dict[next_word_id]}'")
            
            if next_word_id == tgt_vocab['E']:
                break
            generated_ids.append(next_word_id)
            dec_input = torch.tensor([[next_word_id]], device=device)
            
        return ' '.join([number_dict[w] for w in generated_ids])

    # >>> NEW: 使用 KV cache 的“安静版”生成（无打印，只用于计时）
    def generate_with_cache(self, enc_input_tensor, max_len):
        self.eval()
        enc_outputs = self.encoder(enc_input_tensor)
        dec_input = torch.zeros(1, 1, dtype=torch.long, device=device).fill_(tgt_vocab['S'])
        cache = None
        generated_ids = []
        for step in range(max_len):
            dec_outputs, cache = self.decoder(
                dec_input, enc_input_tensor, enc_outputs, past_cache=cache, step=step
            )
            logits = self.projection(dec_outputs)  # (1,1,V)
            next_word_id = logits.squeeze(0).max(dim=-1)[1].item()
            generated_ids.append(next_word_id)
            if next_word_id == tgt_vocab['E']:
                break
            dec_input = torch.tensor([[next_word_id]], device=device)
        return generated_ids

    # >>> NEW: 不使用 KV cache，每步重算整句的生成
    def generate_without_cache(self, enc_input_tensor, max_len):
        self.eval()
        enc_outputs = self.encoder(enc_input_tensor)
        dec_input_ids = [tgt_vocab['S']]  # 当前已生成的 prefix（含起始符）
        generated_ids = []
        for _ in range(max_len):
            # 每一步把完整前缀送入 decoder，使用训练模式的 Mask
            dec_input = torch.tensor([dec_input_ids], dtype=torch.long, device=device)  # (1, L)
            dec_outputs, _ = self.decoder(
                dec_input, enc_input_tensor, enc_outputs,
                past_cache=None, step=None
            )
            logits = self.projection(dec_outputs)  # (1, L, V)
            # 只取最后一个位置的分布
            last_logits = logits[0, -1]  # (V,)
            next_word_id = last_logits.max(dim=-1)[1].item()
            generated_ids.append(next_word_id)
            if next_word_id == tgt_vocab['E']:
                break
            dec_input_ids.append(next_word_id)
        return generated_ids


def load_and_inference(model_path):
    """
    加载已保存的模型权重，并进行一次完整的推理。
    """
    print("\n" + "="*30)
    print(" Running Inference with Loaded Model ")
    print("="*30)

    loaded_model = Transformer().to(device)

    print(f"Loading weights from: {model_path}")
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()

    with torch.no_grad():
        test_enc_inputs, _, _ = make_batch(sentences)
        result = loaded_model.inference(test_enc_inputs)
        print_colored_text(f"\n✔ Inference Result from Loaded Model: '{result}'", 'green', attrs=['bold'])

    return loaded_model  # >>> NEW: 返回模型给后面 benchmark 用

# >>> NEW: 简单的 KV cache vs 无 cache 的性能对比
def benchmark_inference(model, max_len=50, iters=50):
    print("\n" + "="*30)
    print(f" KV Cache Benchmark (max_len={max_len}, iters={iters}) ")
    print("="*30)

    enc_inputs, _, _ = make_batch(sentences)

    # 预热（warmup），避免第一次 CUDA 初始化干扰
    with torch.no_grad():
        for _ in range(5):
            model.generate_with_cache(enc_inputs, max_len)
            model.generate_without_cache(enc_inputs, max_len)

    # 计时：有 KV cache
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            model.generate_with_cache(enc_inputs, max_len)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    # 计时：无 KV cache
    with torch.no_grad():
        for _ in range(iters):
            model.generate_without_cache(enc_inputs, max_len)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t2 = time.time()

    avg_with = (t1 - t0) / iters * 1000
    avg_without = (t2 - t1) / iters * 1000

    print(f"Avg time WITH KV cache   : {avg_with:.3f} ms / sequence")
    print(f"Avg time WITHOUT KV cache: {avg_without:.3f} ms / sequence")


# ==========================================
# 4. 执行
# ==========================================
if __name__ == "__main__":
    MODEL_SAVE_PATH = "./transformer_model.pth"
    train_mode = False
    if train_mode:
        model = Transformer().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("Training...")
        enc_inputs, dec_inputs, target_batch = make_batch(sentences)
        
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, target_batch.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            if (epoch+1) % 20 == 0:
                print(f"Epoch {epoch+1} loss: {loss.item():.6f}")

        print("\n" + "="*30)
        print(f"Saving trained model to {MODEL_SAVE_PATH}")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print_colored_text("✔ Model saved successfully!", 'green')
        
    else:
        # 加载模型并做一次正常推理
        model = load_and_inference(MODEL_SAVE_PATH)

        # >>> NEW: 做 KV cache vs 无 cache 的耗时对比
        benchmark_inference(model, max_len=50, iters=100)
