import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageAugmentation(object):
    """
    [IDA] 图像空间数据增强
    功能: 随机旋转、缩放、裁剪、翻转
    核心: 维护一个 3x3 的仿射变换矩阵，记录所有的操作
    """
    def __init__(self, output_size=(256, 704)):
        self.output_size = output_size # (H, W)

    def sample_augmentation(self):
        """随机生成增强参数"""
        # 1. 随机缩放 (Resize)
        resize_scale = np.random.uniform(0.5, 1.5)
        
        # 2. 随机旋转 (Rotate) - 角度
        rotate_angle = np.random.uniform(-10, 10)
        
        # 3. 随机裁剪 (Crop)
        # 这里简化处理，假设在中心附近裁剪
        crop_x = np.random.randint(0, 50)
        crop_y = np.random.randint(0, 20)
        
        # 4. 随机翻转 (Flip)
        flip = np.random.random() > 0.5
        
        return resize_scale, rotate_angle, (crop_x, crop_y), flip

    def get_rot_trans_matrix(self, resize, rotate, crop, flip, H_img, W_img):
        """
        计算累积的仿射变换矩阵 (3x3)
        顺序: Flip -> Rotate -> Resize -> Crop
        """
        # 1. 初始单位矩阵
        post_rot = np.eye(2)
        post_tran = np.zeros(2)
        
        # --- Flip (翻转) ---
        if flip:
            # 水平翻转: x -> W - x
            # [-1, 0]
            # [ 0, 1]
            flip_mat = np.array([[-1, 0], [0, 1]])
            post_rot = flip_mat @ post_rot
            # 翻转会引入一个平移 W
            post_tran += np.array([W_img, 0]) 

        # --- Rotate (旋转) ---
        if rotate != 0:
            # OpenCV 的旋转矩阵以图像中心为圆心
            center = (W_img / 2, H_img / 2)
            rot_mat_cv2 = cv2.getRotationMatrix2D(center, rotate, 1.0) # 2x3
            rot_mat = rot_mat_cv2[:2, :2]
            trans_val = rot_mat_cv2[:2, 2]
            
            post_rot = rot_mat @ post_rot
            post_tran = rot_mat @ post_tran + trans_val

        # --- Resize (缩放) ---
        if resize != 1.0:
            scale_mat = np.array([[resize, 0], [0, resize]])
            post_rot = scale_mat @ post_rot
            post_tran = scale_mat @ post_tran # 平移量也要缩放

        # --- Crop (裁剪) ---
        # 裁剪就是把坐标系原点移动，相当于减去 crop 的 (x, y)
        post_tran -= np.array(crop)

        return post_rot, post_tran

    def augment(self, img):
        """执行增强"""
        H, W = img.shape[:2]
        
        # 1. 获取随机参数
        resize, rotate, crop, flip = self.sample_augmentation()
        
        # 2. 计算变换矩阵 (用于 Model 去抵消)
        post_rot, post_tran = self.get_rot_trans_matrix(resize, rotate, crop, flip, H, W)
        
        # 3. 实际对图片进行 OpenCV 操作 (用于可视化验证)
        img_aug = img.copy()
        
        if flip:
            img_aug = cv2.flip(img_aug, 1) # 1 为水平翻转
            
        if rotate != 0:
            M = cv2.getRotationMatrix2D((W/2, H/2), rotate, 1.0)
            img_aug = cv2.warpAffine(img_aug, M, (W, H))
            
        if resize != 1.0:
            new_W, new_H = int(W * resize), int(H * resize)
            img_aug = cv2.resize(img_aug, (new_W, new_H))
            
        # Crop
        crop_x, crop_y = crop
        img_aug = img_aug[crop_y : crop_y + self.output_size[0], 
                          crop_x : crop_x + self.output_size[1]]
        
        # 填充或裁剪到固定大小 (这里简化，假设裁剪后正好符合 output_size)
        # 实际代码需要做 Padding 处理防止尺寸不匹配
        if img_aug.shape[0] != self.output_size[0] or img_aug.shape[1] != self.output_size[1]:
             img_aug = cv2.resize(img_aug, (self.output_size[1], self.output_size[0]))

        return img_aug, post_rot, post_tran

# ================= 测试 IDA =================
# 模拟一张图片
fake_img = np.zeros((512, 1024, 3), dtype=np.uint8)
# 画个框方便看旋转
cv2.rectangle(fake_img, (100, 100), (400, 400), (255, 255, 255), 5) 
cv2.putText(fake_img, "Ego Car", (450, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

ida = ImageAugmentation(output_size=(256, 704))
img_aug, post_rot, post_tran = ida.augment(fake_img)

print("=== IDA (Image-View Augmentation) ===")
print("Post Rotation Matrix (2x2):\n", post_rot)
print("Post Translation (2):\n", post_tran)
# plt.imshow(img_aug) # 这里可以显示图片