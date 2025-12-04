from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmcv import Config

# 加载配置文件
config_file = 'configs/path_to_your_config.py'  # 配置文件路径
cfg = Config.fromfile(config_file)

# 修改配置中的 checkpoint 路径
cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint='open-mmlab://regnetx_800mf')

# 构建模型
model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

# 加载预训练模型
checkpoint = load_checkpoint(model, cfg.load_from, map_location='cpu')  # 如果需要 GPU，可以使用 'cuda:0'

print("RegNetX-800mf model loaded successfully.")
