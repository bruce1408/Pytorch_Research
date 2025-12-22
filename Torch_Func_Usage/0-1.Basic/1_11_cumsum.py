import torch
import numpy as np

def basic_usage():
    a = np.array(
    [[ 1,  2,  3],
    [ 4,  5,  6],
    [ 7,  8,  9]]
    )

    a = torch.tensor(a)

    cumsum = torch.cumsum(a, 0)

    print(cumsum)

    cumsum = torch.cat((cumsum[:1], cumsum[1:] - cumsum[:-1]))

    print(cumsum)



# 1. 假设这是已经排序好的特征，有3个分组
# 组1: [1, 2], [3, 4] (索引都是10)
# 组2: [5, 6] (索引是25)
# 组3: [7, 8], [9, 10] (索引是30)
geom_feats = torch.tensor([
    [1, 2],  # <-- 组1开始
    [3, 4],  # <-- 组1结束
    [5, 6],  # <-- 组2开始&结束
    [7, 8],  # <-- 组3开始
    [9, 10]  # <-- 组3结束
])

# 2. 这是分组的边界 (每个分组最后一个元素的位置)
# 对应索引 [10, 10, 25, 30, 30]
#           |    |   |   |    |
# 位置      0    1   2   3    4
keep = torch.tensor([False, True, True, False, True])


# --- 开始计算 ---

# 3. 计算前缀和
cumsum_feats = torch.cumsum(geom_feats, 0)
print("--- 前缀和 (cumsum_feats) ---")
print(cumsum_feats)
# 预期输出:
# [[ 1,  2],
#  [ 4,  6],
#  [ 9, 12],
#  [16, 20],
#  [25, 30]]


# 4. 提取每个分组末尾的前缀和
group_prefix_sums = cumsum_feats[keep]
print("\n--- 每个分组末尾的前缀和 (group_prefix_sums) ---")
print(group_prefix_sums)
# 预期输出 (对应 keep=True 的第1, 2, 4行):
# [[ 4,  6],   (组1的和)
#  [ 9, 12],  (组1+组2的和)
#  [25, 30]]  (组1+组2+组3的和)


# 5. 求差，得到每个分组自己的和
summed_feats = torch.cat((group_prefix_sums[:1], group_prefix_sums[1:] - group_prefix_sums[:-1]))
print("\n--- 最终结果：每个分组自己的和 (summed_feats) ---")
print(summed_feats)

# 预期结果:
# [[4, 6],   (等于 [1,2] + [3,4])
#  [5, 6],   (等于 [9,12] - [4,6])
#  [16, 18]]  (等于 [25,30] - [9,12])