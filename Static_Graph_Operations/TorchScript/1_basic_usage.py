# Python 侧
from common import enter_workspace
import torch, torchvision as tv

enter_workspace()

m = tv.models.resnet18(weights="IMAGENET1K_V1").eval()
ex = torch.randn(1,3,224,224)

scripted = torch.jit.trace(m, ex)      # 或 torch.jit.script(m)
scripted.save("resnet18.ts")
