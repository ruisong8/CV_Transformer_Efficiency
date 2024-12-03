import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


save_path = "resnet50"

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.images = []
        self.masks = []

        if self.mode == 'train':
            image_dir = os.path.join(root_dir, 'train', 'image')
            mask_dir = os.path.join(root_dir, 'train', 'mask')
            self.images = glob.glob(os.path.join(image_dir, '*.jpg'))
            self.masks = glob.glob(os.path.join(mask_dir, '*.png'))
        elif self.mode == 'test':
            test_folders = ['ADE20K', 'COCO', 'MINC', 'NYUD', 'PASCAL', 'SUNRGBD']
            for folder in test_folders:
                image_dir = os.path.join(root_dir, 'test', folder, 'image')
                mask_dir = os.path.join(root_dir, 'test', folder, 'mask')
                self.images.extend(glob.glob(os.path.join(image_dir, '*.jpg')))
                self.masks.extend(glob.glob(os.path.join(mask_dir, '*.png')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except (OSError, ValueError) as e:
            print(f"Error loading image {img_path}: {e}")
            # 跳过此索引，返回一个空图像和掩膜
            return self.__getitem__((idx + 1) % len(self.images))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # 将掩膜转换为二进制格式
        mask = (mask > 0).float()  # 假设掩膜中非零值表示目标

        return image, mask

# 图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 数据集和数据加载器
train_dataset = CustomDataset(root_dir='dataset/PMD', mode='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

val_dataset = CustomDataset(root_dir='dataset/PMD', mode='test', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

# 初始化 ResNet50 模型
class ResNet50Segmentation(nn.Module):
    def __init__(self):
        super(ResNet50Segmentation, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # 去掉全连接层
        self.conv1x1 = nn.Conv2d(2048, 1, kernel_size=1)  # 输出一个通道的 logits
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)  # 上采样到原图大小

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1x1(x)
        x = self.upsample(x)  # 上采样
        return x
    
model = ResNet50Segmentation().to('cuda' if torch.cuda.is_available() else 'cpu')
# print(model)

# 优化器设置
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

# 损失函数
criterion = nn.BCEWithLogitsLoss()  # 用于二分类问题

# 训练过程
num_epochs = 100
val_interval = 1  # 设置每多少个 epoch 进行一次验证
best_val_loss = float('inf')  # 初始化最好的验证损失

# 学习率调度器
def adjust_learning_rate(optimizer, epoch):
    lr = 1e-3 * (1 - epoch / num_epochs) ** 0.9  # 多项式学习率策略
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(num_epochs):
    model.train()
    adjust_learning_rate(optimizer, epoch)

    # 训练阶段
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for images, masks in train_loader:
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            masks = masks.to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, masks)  # 添加一维以适应输出形状
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)  # 更新进度条

    # 每 val_interval 个 epoch 进行一次验证
    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_loss = 0.0
        total_mae = 0.0
        num_samples = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
                masks = masks.to('cuda' if torch.cuda.is_available() else 'cpu')

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs)  # 获取概率值
                mae = torch.mean(torch.abs(preds - masks))  # 计算 MAE
                total_mae += mae.item() * images.size(0)  # 累加 MAE
                num_samples += images.size(0)  # 累加样本数

        val_loss /= len(val_loader)  # 计算平均验证损失
        mean_mae = total_mae / num_samples  # 计算平均 MAE

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, MAE: {mean_mae:.4f}')

        # 只保存最好的模型参数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path + 'best_model.pth')  # 保存最好的模型
            print(f'Saved best model at epoch {epoch + 1} with validation loss: {best_val_loss:.4f}')

print("Training complete.")