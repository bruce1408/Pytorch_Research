# #!/usr/bin/env python

# # Generative Adversarial Networks (GAN) example in PyTorch.
# import numpy as np
# import torch
# import os
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# matplotlib_is_available = True
# try:
#     from matplotlib import pyplot as plt
# except ImportError:
#     print("Will skip plotting; matplotlib is not available.")
#     matplotlib_is_available = False

# # Data params
# data_mean = 4
# data_stddev = 1.25

# # ### Uncomment only one of these to define what Dataset is actually sent to the Discriminator
# # (name, preprocess, d_input_func) = ("Raw Dataset", lambda Dataset: Dataset, lambda x: x)
# # (name, preprocess, d_input_func) = ("Data and variances", lambda Dataset: decorate_with_diffs(Dataset, 2.0), lambda x: x * 2)
# # (name, preprocess, d_input_func) = ("Data and diffs", lambda Dataset: decorate_with_diffs(Dataset, 1.0), lambda x: x * 2)
# (name, preprocess, d_input_func) = ("Only 4 moments", lambda data: get_moments(data), lambda x: 4)

# print("Using Dataset [%s]" % (name))


# # ##### DATA: Target Dataset and generator input Dataset

# def get_distribution_sampler(mu, sigma):
#     return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian


# def get_generator_input_sampler():
#     return lambda m, n: torch.rand(m, n)  # Uniform-dist Dataset into generator, _NOT_ Gaussian


# # ##### MODELS: Generator model and discriminator model

# class Generator(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, f):
#         super(Generator, self).__init__()
#         self.map1 = nn.Linear(input_size, hidden_size)
#         self.map2 = nn.Linear(hidden_size, hidden_size)
#         self.map3 = nn.Linear(hidden_size, output_size)
#         self.f = f

#     def forward(self, x):
#         x = self.map1(x)
#         x = self.f(x)
#         x = self.map2(x)
#         x = self.f(x)
#         x = self.map3(x)
#         return x


# class Discriminator(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, f):
#         super(Discriminator, self).__init__()
#         self.map1 = nn.Linear(input_size, hidden_size)
#         self.map2 = nn.Linear(hidden_size, hidden_size)
#         self.map3 = nn.Linear(hidden_size, output_size)
#         self.f = f

#     def forward(self, x):
#         x = self.f(self.map1(x))
#         x = self.f(self.map2(x))
#         return self.f(self.map3(x))


# def extract(v):
#     return v.data.storage().tolist()


# def stats(d):
#     return [np.mean(d), np.std(d)]


# def get_moments(d):
#     # Return the first 4 moments of the Dataset provided
#     mean = torch.mean(d)
#     diffs = d - mean
#     var = torch.mean(torch.pow(diffs, 2.0))
#     std = torch.pow(var, 0.5)
#     zscores = diffs / std
#     skews = torch.mean(torch.pow(zscores, 3.0))
#     kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0  # excess kurtosis, should be 0 for Gaussian
#     final = torch.cat((mean.reshape(1, ), std.reshape(1, ), skews.reshape(1, ), kurtoses.reshape(1, )))
#     return final


# def decorate_with_diffs(data, exponent, remove_raw_data=False):
#     mean = torch.mean(data.data, 1, keepdim=True)
#     mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
#     diffs = torch.pow(data - Variable(mean_broadcast), exponent)
#     if remove_raw_data:
#         return torch.cat([diffs], 1)
#     else:
#         return torch.cat([data, diffs], 1)


# def train():
#     # Model parameters
#     g_input_size = 1  # Random noise dimension coming into generator, per output vector
#     g_hidden_size = 5  # Generator complexity
#     g_output_size = 1  # Size of generated output vector
#     d_input_size = 500  # Minibatch size - cardinality of distributions
#     d_hidden_size = 10  # Discriminator complexity
#     d_output_size = 1  # Single dimension for 'real' vs. 'fake' classification
#     minibatch_size = d_input_size

#     d_learning_rate = 1e-3
#     g_learning_rate = 1e-3
#     sgd_momentum = 0.9

#     num_epochs = 5000
#     print_interval = 100
#     d_steps = 20
#     g_steps = 20

#     dfe, dre, ge = 0, 0, 0
#     d_real_data, d_fake_data, g_fake_data = None, None, None

#     discriminator_activation_function = torch.sigmoid
#     generator_activation_function = torch.tanh

#     d_sampler = get_distribution_sampler(data_mean, data_stddev)
#     gi_sampler = get_generator_input_sampler()
#     G = Generator(input_size=g_input_size,
#                   hidden_size=g_hidden_size,
#                   output_size=g_output_size,
#                   f=generator_activation_function)
#     D = Discriminator(input_size=d_input_func(d_input_size),
#                       hidden_size=d_hidden_size,
#                       output_size=d_output_size,
#                       f=discriminator_activation_function)
#     criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
#     d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)
#     g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate, momentum=sgd_momentum)

#     for epoch in range(num_epochs):
#         for d_index in range(d_steps):
#             # 1. Train D on real+fake
#             D.zero_grad()

#             #  1A: Train D on real
#             d_real_data = Variable(d_sampler(d_input_size))
#             d_real_decision = D(preprocess(d_real_data))
#             # d_real_error = criterion(d_real_decision, Variable(torch.ones([1, 1])))  # ones = true
#             d_real_error = criterion(d_real_decision.view(1, 1), Variable(torch.ones([1, 1])))  # ones = true

#             d_real_error.backward()  # compute/store gradients, but don't change params

#             #  1B: Train D on fake
#             d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
#             d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
#             d_fake_decision = D(preprocess(d_fake_data.t()))
#             # d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([1, 1])))  # zeros = fake
#             d_fake_error = criterion(d_fake_decision.view(1, 1), Variable(torch.zeros([1, 1])))
#             d_fake_error.backward()
#             d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

#             dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]

#         for g_index in range(g_steps):
#             # 2. Train G on D's response (but DO NOT train D on these labels)
#             G.zero_grad()

#             gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
#             g_fake_data = G(gen_input)
#             dg_fake_decision = D(preprocess(g_fake_data.t()))
#             # g_error = criterion(dg_fake_decision, Variable(torch.ones([1, 1])))  # Train G to pretend it's genuine
#             g_error = criterion(dg_fake_decision.view(1, 1), Variable(torch.ones([1, 1])))
#             g_error.backward()
#             g_optimizer.step()  # Only optimizes G's parameters
#             ge = extract(g_error)[0]

#         if epoch % print_interval == 0:
#             print("Epoch %s: D (%s real_err, %s fake_err) G (%s err); Real Dist (%s),  Fake Dist (%s) " %
#                   (epoch, dre, dfe, ge, stats(extract(d_real_data)), stats(extract(d_fake_data))))

#     if matplotlib_is_available:
#         print("Plotting the generated distribution...")
#         values = extract(g_fake_data)
#         print(" Values: %s" % (str(values)))
#         plt.hist(values, bins=50)
#         plt.xlabel('Value')
#         plt.ylabel('Count')
#         plt.title('Histogram of Generated Distribution')
#         plt.grid(True)
#         plt.show()


# train()



#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- 优化 1: 自动设备选择 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data params
data_mean = 4
data_stddev = 1.25

# 数据预处理配置
(name, preprocess, d_input_func) = ("Only 4 moments", lambda data: get_moments(data), lambda x: 4)
print("Using Dataset [%s]" % (name))

# --- 数据采样器 ---
def get_distribution_sampler(mu, sigma):
    # 生成真实的高斯分布数据
    return lambda n: torch.normal(mean=mu, std=sigma, size=(1, n)).to(device)

def get_generator_input_sampler():
    # 生成器输入的随机噪声 (Uniform)
    return lambda m, n: torch.rand(m, n).to(device)

# --- 模型定义 ---
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(), # Generator 常用 Tanh 或 ReLU
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(), # 中间层可以用 Sigmoid 或 LeakyReLU
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size)
            # --- 优化 2: 移除最后一层 Sigmoid，配合 BCEWithLogitsLoss 使用 ---
        )

    def forward(self, x):
        return self.net(x)

# --- 辅助函数 ---
def get_moments(d):
    # 计算数据的 4 个统计矩：均值、标准差、偏度、峰度
    mean = torch.mean(d)
    diffs = d - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0 
    final = torch.cat((mean.reshape(1), std.reshape(1), skews.reshape(1), kurtoses.reshape(1)))
    return final

def train():
    # Model parameters
    g_input_size = 1
    g_hidden_size = 50 #稍微增加一点复杂度
    g_output_size = 1
    d_input_size = 500
    d_hidden_size = 50 
    d_output_size = 1
    minibatch_size = d_input_size

    d_learning_rate = 1e-3
    g_learning_rate = 1e-3
    
    num_epochs = 5000
    print_interval = 500
    d_steps = 20
    g_steps = 20

    # 初始化模型并移动到 GPU
    G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size).to(device)
    D = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size).to(device)

    # --- 优化 2: 使用 BCEWithLogitsLoss 提升数值稳定性 ---
    criterion = nn.BCEWithLogitsLoss() 
    
    # --- 优化 5: 建议使用 Adam，这里为了保持原意保留 SGD，但去掉了 momentum 参数如果不需要 ---
    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=0.9)
    g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate, momentum=0.9)

    d_sampler = get_distribution_sampler(data_mean, data_stddev)
    gi_sampler = get_generator_input_sampler()

    for epoch in range(num_epochs):
        # 1. Train Discriminator
        for d_index in range(d_steps):
            D.zero_grad()

            # 1A: Real Data
            d_real_data = d_sampler(d_input_size)
            d_real_decision = D(preprocess(d_real_data))
            # 目标是 1
            d_real_error = criterion(d_real_decision.view(1, 1), torch.ones(1, 1).to(device))
            d_real_error.backward()

            # 1B: Fake Data
            d_gen_input = gi_sampler(minibatch_size, g_input_size)
            d_fake_data = G(d_gen_input).detach() # detach 防止梯度传回 G
            d_fake_decision = D(preprocess(d_fake_data.t()))
            # 目标是 0
            d_fake_error = criterion(d_fake_decision.view(1, 1), torch.zeros(1, 1).to(device))
            d_fake_error.backward()
            
            d_optimizer.step()

        # 2. Train Generator
        for g_index in range(g_steps):
            G.zero_grad()

            gen_input = gi_sampler(minibatch_size, g_input_size)
            g_fake_data = G(gen_input)
            dg_fake_decision = D(preprocess(g_fake_data.t()))
            
            # Generator 的目标是让 Discriminator 认为是 1 (真)
            g_error = criterion(dg_fake_decision.view(1, 1), torch.ones(1, 1).to(device))
            g_error.backward()
            g_optimizer.step()

        if epoch % print_interval == 0:
            # --- 优化 4: 使用 .item() 获取数值，更 pythonic ---
            print("Epoch %d: D (real_err: %.4f, fake_err: %.4f) G (err: %.4f)" %
                  (epoch, d_real_error.item(), d_fake_error.item(), g_error.item()))
            
            # 简单的统计数据打印
            real_np = d_real_data.detach().cpu().numpy()
            fake_np = d_fake_data.detach().cpu().numpy()
            print(f"  Real Mean: {np.mean(real_np):.2f}, Std: {np.std(real_np):.2f}")
            print(f"  Fake Mean: {np.mean(fake_np):.2f}, Std: {np.std(fake_np):.2f}")

    # Plotting
    print("Plotting the generated distribution...")
    values = g_fake_data.detach().cpu().numpy().flatten()
    plt.hist(values, bins=50, alpha=0.6, label='Generated')
    
    
    # 画一个真实的分布对比一下
    real_values = np.random.normal(data_mean, data_stddev, 5000)
    plt.hist(real_values, bins=50, alpha=0.4, label='Real')
    
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Histogram: Generated vs Real')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('gan.pdf')

if __name__ == '__main__':
    train()