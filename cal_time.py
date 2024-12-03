import numpy as np
import torch
from torch.backends import cudnn
import tqdm
from torchvision.models import resnet50
from timm.models.vision_transformer import vit_base_patch16_224
from timm.models.swin_transformer import swin_large_patch4_window7_224


cudnn.benchmark = True

device = 'cuda:0'
model = swin_large_patch4_window7_224().to(device)
repetitions = 1000

dummy_input1 = torch.rand(1, 3, 224, 224).to(device)

# 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
print('warm up ...\n')
with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_input1)

# synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
torch.cuda.synchronize()

# 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# 初始化一个时间容器
timings = np.zeros((repetitions, 1))

print('testing ...\n')
with torch.no_grad():
    for rep in tqdm.tqdm(range(repetitions)):
        starter.record()
        _ = model(dummy_input1)
        ender.record()
        torch.cuda.synchronize()  # 等待GPU任务完成
        curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
        timings[rep] = curr_time

avg = timings.sum() / repetitions
print('\navg={}\n'.format(avg))
