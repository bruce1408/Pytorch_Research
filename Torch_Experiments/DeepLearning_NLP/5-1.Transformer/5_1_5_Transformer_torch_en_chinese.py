import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import re

# ==========================================
# 1. 配置与环境
# ==========================================
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# ================================ fake data for testing ==========================================
# # 你的样本数据 (直接写入文件模拟真实环境)
# RAW_DATA = """I'll sauté a few sweet potatoes and sprinkle them with sugar.\t我去炸一點紅薯，然後在上面撒上糖。\tCC-BY 2.0
# I've been a public school teacher for the past thirteen years.\t過去 13 年，我都是公立學校的老師。\tCC-BY 2.0
# If someone irritates you, it is best not to react immediately.\t如果有人激怒你，你最好不要立刻做出反应。\tCC-BY 2.0
# If they hadn't noticed, there wouldn't have been any problems.\t要是他们没有发现，就没问题了。\tCC-BY 2.0
# Islam first reached China about the middle of the 7th century.\t伊斯蘭教大約在七世纪中傳到中國。\tCC-BY 2.0
# It is becoming important for us to know how to use a computer.\t知道如何使用电脑对我们来说变得很重要。\tCC-BY 2.0
# Japan is now very different from what it was twenty years ago.\t相比二十年前的日本，现在的日本有了翻天覆地的变化。\tCC-BY 2.0
# Japan is now very different from what it was twenty years ago.\t现在的日本与二十年前大不相同。\tCC-BY 2.0
# Language is the means by which people communicate with others.\t语言是人们与他人交流的手段。\tCC-BY 2.0
# Let me stop you right there. We don't want to hear about that.\t讓我在這打斷你。我們不想聽那個話題。\tCC-BY 2.0
# London was very important for economical and cultural reasons.\t倫敦過去因為經濟和文化的緣故，十分重要。\tCC-BY 2.0"""

# # 写入本地文件 cmn.txt
# with open("cmn.txt", "w", encoding="utf-8") as f:
#     f.write(RAW_DATA)
# ================================ fake data for testing ==========================================

    
# 超参数
MAX_LEN = 128           # 句子最大长度
BATCH_SIZE = 64        # 样本少，Batch size 设小点
# 修改后的建议 (标准 Base 模型规模)
D_MODEL = 128           # 嵌入维度变大 (显存占用主要来源之一) d_k * n_heads
D_FF = 512             # 前馈网络维度变大 (通常是 D_MODEL * 4)
N_LAYERS = 3            # 层数加深 (计算量和显存都会增加)
N_HEADS = 4
D_K = D_V = 32
LR = 0.001
EPOCHS = 100            # 样本极少，多跑几轮过拟合它，看看效果
USE_AMP = False
# ==========================================
# 2. 数据集准备 (适配中文)
# ==========================================
class En2ZhDataset(Dataset):
    def __init__(self):
        self.file_path = "/home/bruce_ultra/data/datasets/cmn.txt"
        # self.file_path = "cmn.txt"
        self.raw_data = []
        
        # 读取数据
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    eng, zho = parts[0], parts[1]
                    # 1. 英文分词 (按空格和标点)
                    src_tokens = self.tokenize_en(eng)
                    # 2. 中文分词 (按字切分，Character-level)
                    tgt_tokens = self.tokenize_cn(zho)
                    
                    self.raw_data.append((src_tokens, tgt_tokens))

        # 构建词表
        self.src_vocab, self.src_idx2word = self.build_vocab([x[0] for x in self.raw_data])
        self.tgt_vocab, self.tgt_idx2word = self.build_vocab([x[1] for x in self.raw_data], is_target=True)
        
        print(f"✅ 数据加载完成: {len(self.raw_data)} 条")
        print(f"✅ 源语言词表(英文): {len(self.src_vocab)}")
        print(f"✅ 目标语言词表(中文-字级): {len(self.tgt_vocab)}")

    
    def tokenize_en(self, text):
        text = text.lower()
        # 把标点符号用空格隔开，这样 split 就能把它们独立出来
        text = re.sub(r"([?.!,])", r" \1 ", text)
        # 去除多余空格
        text = re.sub(r'[" "]+', " ", text)
        return text.strip().split()

    def tokenize_cn(self, text):
        # 中文按字切分是没问题的
        return [char for char in text.strip()]
    
    
    # def tokenize_en(self, text):
    #     # 英文：转小写，把标点符号单独拆出来
    #     text = text.lower()
    #     text = re.sub(r"([?.!,])", r" \1 ", text) # 在标点前后加空格
    #     text = re.sub(r'[" "]+', " ", text)       # 合并多余空格
    #     return text.strip().split()

    # def tokenize_cn(self, text):
    #     # 中文：字级别分词 (每个汉字算一个token)
    #     # 这种方法最简单且不需要安装 jieba
    #     return [char for char in text.strip()]

    def build_vocab(self, sentences, is_target=False):
        counter = Counter()
        for sent in sentences:
            counter.update(sent)
        
        # 0:Pad, 1:Unk
        vocab = {'<P>': 0, '<UNK>': 1}
        if is_target:
            vocab['<S>'] = 2
            vocab['<E>'] = 3
        
        # 将所有词加入词典
        for word, _ in counter.items():
            if word not in vocab:
                vocab[word] = len(vocab)
            
        idx2word = {v: k for k, v in vocab.items()}
        return vocab, idx2word

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.raw_data[idx]
        
        # 转 ID
        src_ids = [self.src_vocab.get(w, self.src_vocab['<UNK>']) for w in src_tokens]
        tgt_ids = [self.tgt_vocab.get(w, self.tgt_vocab['<UNK>']) for w in tgt_tokens]
        
        # 截断
        src_ids = src_ids[:MAX_LEN]
        tgt_ids = tgt_ids[:MAX_LEN]
        
        # Decoder Input: <S> + 句子
        dec_input = [self.tgt_vocab['<S>']] + tgt_ids
        
        # Target Label: 句子 + <E>
        dec_label = tgt_ids + [self.tgt_vocab['<E>']]
        
        return torch.LongTensor(src_ids), torch.LongTensor(dec_input), torch.LongTensor(dec_label)

def collate_fn(batch):
    src_list, dec_input_list, dec_label_list = [], [], []
    for src, dec_in, dec_lbl in batch:
        src_list.append(src)
        dec_input_list.append(dec_in)
        dec_label_list.append(dec_lbl)
    
    src_pad = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=0)
    dec_in_pad = nn.utils.rnn.pad_sequence(dec_input_list, batch_first=True, padding_value=0)
    dec_lbl_pad = nn.utils.rnn.pad_sequence(dec_label_list, batch_first=True, padding_value=0)
    
    return src_pad, dec_in_pad, dec_lbl_pad

# 初始化数据
dataset = En2ZhDataset()
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 在 DataLoader 中添加 num_workers
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=8,        # 设置为 4 或 8 (取决于你 CPU 的核心数)
    pin_memory=True       # 开启锁页内存，加速 CPU 到 GPU 的传输
)

# ==========================================
# 3. Transformer 模型 (保持原样，无需修改)
# ==========================================
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    return torch.from_numpy(subsequent_mask).byte().to(seq.device)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(D_K)
        scores.masked_fill_(attn_mask, -1e9)
        # scores.masked_fill_(attn_mask, -1e4) # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(D_MODEL, D_K * N_HEADS)
        self.W_K = nn.Linear(D_MODEL, D_K * N_HEADS)
        self.W_V = nn.Linear(D_MODEL, D_V * N_HEADS)
        self.linear = nn.Linear(N_HEADS * D_V, D_MODEL)
        self.layer_norm = nn.LayerNorm(D_MODEL)
    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, N_HEADS, D_V).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, N_HEADS, 1, 1)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, N_HEADS * D_V)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(D_MODEL)
    def forward(self, inputs):
        residual = inputs
        output = self.fc2(self.relu(self.fc1(inputs)))
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(len(dataset.src_vocab), D_MODEL)
        self.pos_emb = nn.Embedding(MAX_LEN + 1, D_MODEL)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(N_LAYERS)])
    def forward(self, enc_inputs):
        seq_len = enc_inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=enc_inputs.device).unsqueeze(0).expand_as(enc_inputs)
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(pos)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(len(dataset.tgt_vocab), D_MODEL)
        self.pos_emb = nn.Embedding(MAX_LEN + 1, D_MODEL)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(N_LAYERS)])
    def forward(self, enc_inputs, dec_inputs, enc_outputs):
        seq_len = dec_inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=dec_inputs.device).unsqueeze(0).expand_as(dec_inputs)
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(D_MODEL, len(dataset.tgt_vocab), bias=False)
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(enc_inputs, dec_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


def eval_model(model, dataset):
    # ==========================================
    # 5. 测试 (贪婪解码)
    # ==========================================
    print("\n🧪 测试最后一条数据:")
    # 取最后一条长一点的句子: "Let me stop you right there..."
    test_sample = dataset.raw_data[-1] 
    src_text = "Let me stop you right there." # 手动指定一个测试句，看效果
    src_text = "If you had left home a little earlier you would have been in time."
    print(f"原文 (En): {src_text}")

    model.eval()
    # 构造输入
    src_tokens = dataset.tokenize_en(src_text)
    src_idxs = [dataset.src_vocab.get(w, 1) for w in src_tokens][:MAX_LEN]
    src_tensor = torch.LongTensor([src_idxs]).to(device)

    # 解码
    dec_input = torch.LongTensor([[dataset.tgt_vocab['<S>']]]).to(device)

    print("预测 (Zh): ", end="")
    for i in range(MAX_LEN):
        with torch.no_grad():
            outputs, _, _, _ = model(src_tensor, dec_input)
            pred_token = outputs.argmax(dim=1)[-1].item()
            
            if pred_token == dataset.tgt_vocab['<E>']:
                break
                
            print(dataset.tgt_idx2word[pred_token], end="") # 中文不需要空格拼接
            
            dec_input = torch.cat([dec_input, torch.LongTensor([[pred_token]]).to(device)], dim=1)
    print("\n")
    
    
# ==========================================
# 4. 训练循环
# ==========================================
model = Transformer().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
# optimizer = optim.Adam(model.parameters(), lr=LR)
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)

# 3. 添加学习率调度器
scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.0005, steps_per_epoch=len(dataloader), epochs=EPOCHS, pct_start=0.15
)
    
print(f"🔥 开始训练 {EPOCHS} 个 Epochs (数据量: {len(dataset)} 条)...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    for i, (enc_inputs, dec_inputs, dec_targets) in enumerate(dataloader):
        enc_inputs, dec_inputs, dec_targets = enc_inputs.to(device), dec_inputs.to(device), dec_targets.to(device)
        optimizer.zero_grad()
        
        
        
        if USE_AMP:
            # 开启自动混合精度上下文
            with torch.cuda.amp.autocast():
                outputs, _, _, _ = model(enc_inputs, dec_inputs)
                loss = criterion(outputs, dec_targets.view(-1))
        else:
            outputs, _, _, _ = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_targets.view(-1))
            
        loss.backward()
        optimizer.step()
        scheduler.step() # 更新学习率
        total_loss += loss.item()
    
    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss / len(dataloader):.4f}")

    if (epoch + 1) % 10 == 0:
        eval_model(model, dataset)

MODEL_SAVE_PATH = "transformer_en_zh.pth" 
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ 模型已保存到: {MODEL_SAVE_PATH}")

eval_model(model, dataset)