import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


extract_path = 'dataset'
model_name = 'swin'
save_folder = model_name + '/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

print("We use " + model_name)
if model_name == 'swin':
    from timm.models.swin_transformer import swin_tiny_patch4_window7_224
    image_size = (224, 224)
elif model_name == 'vit':
    from timm.models.vision_transformer import vit_small_patch16_224
    image_size = (224, 224)
elif model_name == 'resnet':
    image_size = (224, 224)
else:
    assert False

print("\nDataset directory structure:")
for root, dirs, files in os.walk(extract_path):
    level = root.replace(extract_path, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    # Show only first 5 files in each directory
    for f in files[:5]:
        print(f"{subindent}{f}")
    if len(files) > 5:
        print(f"{subindent}...")
        
def count_files(directory):
    """Count number of files in specified directory"""
    return len([name for name in os.listdir(directory)
               if os.path.isfile(os.path.join(directory, name))])

try:
    # Set correct paths including 'AIGC-Detection-Dataset' folder
    base_path = os.path.join(extract_path, 'AIGC-Detection-Dataset')
    train_real = os.path.join(base_path, 'train', '0_real')
    train_fake = os.path.join(base_path, 'train', '1_fake')
    val_real = os.path.join(base_path, 'val', '0_real')
    val_fake = os.path.join(base_path, 'val', '1_fake')

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training - Real images: {count_files(train_real)}")
    print(f"Training - Fake images: {count_files(train_fake)}")
    print(f"Validation - Real images: {count_files(val_real)}")
    print(f"Validation - Fake images: {count_files(val_fake)}")

    # Calculate total images
    total_train = count_files(train_real) + count_files(train_fake)
    total_val = count_files(val_real) + count_files(val_fake)
    print(f"\nTotal training images: {total_train}")
    print(f"Total validation images: {total_val}")
    print(f"Total dataset size: {total_train + total_val}")

except Exception as e:
    print(f"Error counting files: {str(e)}")
    print("Please verify the extracted folder structure is correct")

# Save paths for future use
dataset_paths = {
    'base_path': base_path,
    'train': {
        'real': train_real,
        'fake': train_fake
    },
    'val': {
        'real': val_real,
        'fake': val_fake
    }
}

print("\nPaths have been configured successfully!")

transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize images to ResNet input size
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

class AIGCDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Label 0 for real images
        for img_name in os.listdir(real_dir):
            self.image_paths.append(os.path.join(real_dir, img_name))
            self.labels.append(0)  # 0 = real

        # Label 1 for fake/generated images
        for img_name in os.listdir(fake_dir):
            self.image_paths.append(os.path.join(fake_dir, img_name))
            self.labels.append(1)  # 1 = fake

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
    
class AIGCDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(AIGCDetector, self).__init__()
        # Load pretrained ResNet101
        self.resnet = models.resnet50(pretrained=pretrained)

        # Modify last layer for binary classification (real=0, fake=1)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1),  # Single output node
            nn.Sigmoid()  # Sigmoid for binary classification
        )
        
    def forward(self, x):
        # Output will be between 0 and 1
        # Close to 0 = real, Close to 1 = fake
        return self.resnet(x)


class AIGC_swin(nn.Module):
    def __init__(self, pretrained=True):
        super(AIGC_swin, self).__init__()
        # Load pretrained
        self.resnet = swin_tiny_patch4_window7_224(pretrained=True)

        # Modify last layer for binary classification (real=0, fake=1)
        num_features = self.resnet.head.in_features
        self.resnet.head.fc = nn.Sequential(
            nn.Linear(num_features, 1),  # Single output node
            # nn.Sigmoid()  # Sigmoid for binary classification
        )

    def forward(self, x):
        # Output will be between 0 and 1
        # Close to 0 = real, Close to 1 = fake
        return torch.sigmoid(self.resnet(x))
    
class AIGC_vit(nn.Module):
    def __init__(self, pretrained=True):
        super(AIGC_vit, self).__init__()
        # Load pretrained
        self.resnet = vit_small_patch16_224(pretrained=True)

        # Modify last layer for binary classification (real=0, fake=1)
        num_features = self.resnet.head.in_features
        self.resnet.head = nn.Sequential(
            nn.Linear(num_features, 1),  # Single output node
            # nn.Sigmoid()  # Sigmoid for binary classification
        )

    def forward(self, x):
        # Output will be between 0 and 1
        # Close to 0 = real, Close to 1 = fake
        return torch.sigmoid(self.resnet(x))
    
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # 确保模型在训练模式
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)

        loss.backward()
        optimizer.step()

        # 累积整个epoch的统计数据，而不是只看最后几个batch
        running_loss += loss.item() * labels.size(0)  # 乘以batch size
        predicted = (outputs.squeeze() > 0.5).float()
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    # 计算整个epoch的平均值
    epoch_loss = running_loss / total_samples
    epoch_acc = 100 * total_correct / total_samples
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()  # 确保模型在评估模式
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float()

            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)

            running_loss += loss.item() * labels.size(0)
            predicted = (outputs.squeeze() > 0.5).float()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = 100 * total_correct / total_samples
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            # print(outputs.shape)
            loss = criterion(outputs.squeeze(), labels)

            loss.backward()
            optimizer.step()

            # 更新当前batch的统计信息
            train_running_loss += loss.item() * labels.size(0)
            predicted = (outputs.squeeze() > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            # 更新进度条显示当前batch的信息
            current_loss = train_running_loss / train_total
            current_acc = 100 * train_correct / train_total
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        # 计算整个epoch的训练指标
        train_loss = train_running_loss / train_total
        train_acc = 100 * train_correct / train_total

        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 学习率调整
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_folder + 'best_model.pth')

        # 打印epoch结果
        print(f'\nEpoch {epoch+1} Results:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
def predict_image(model, image_path, device):
    """
    Predict whether an image is real or fake
    Returns: probability of being fake (0=real, 1=fake)
    """
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prob = output.squeeze().item()
        prediction = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob if prob > 0.5 else (1 - prob)
        print(f"Prediction: {prediction}, Confidence: {confidence:.2%}")
        return prob
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("Warning: Using CPU. Training might be slow. Consider using GPU if available.")
        batch_size = 16
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        batch_size = 64

    print(f"Batch size: {batch_size}")

    base_path = 'dataset/AIGC-Detection-Dataset'
    dataset_paths = {
        'train': {
            'real': os.path.join(base_path, 'train', '0_real'),
            'fake': os.path.join(base_path, 'train', '1_fake')
        },
        'val': {
            'real': os.path.join(base_path, 'val', '0_real'),
            'fake': os.path.join(base_path, 'val', '1_fake')
        }
    }

    train_dataset = AIGCDataset(
        real_dir=dataset_paths['train']['real'],
        fake_dir=dataset_paths['train']['fake'],
        transform=transform
    )

    val_dataset = AIGCDataset(
        real_dir=dataset_paths['val']['real'],
        fake_dir=dataset_paths['val']['fake'],
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("\nDataset Information:")
    print(f"Training set size: {len(train_dataset):,}")
    print(f"Validation set size: {len(val_dataset):,}")
    print(f"Total images: {len(train_dataset) + len(val_dataset):,}")
    print(f"Number of batches per epoch: {len(train_loader):,}")

    print("\nInitializing " + model_name)
    if model_name == 'swin':
        model = AIGC_swin(pretrained=True)
    elif model_name == 'vit':
        model = AIGC_vit(pretrained=True)
    elif model_name == 'resnet':
        model = AIGCDetector(pretrained=True)
    
    print("Model created. Starting training...\n")

    try:
        train_model(model, train_loader, val_loader, num_epochs=15)
        print("\nTraining completed successfully!")

        torch.save({
            'model_state_dict': model.state_dict(),
            'device': device.type,
            'batch_size': batch_size,
            'transform': transform
        }, save_folder + 'aigc_detector_final.pth')
        print("Model saved as 'aigc_detector_final.pth'")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        torch.save({
            'model_state_dict': model.state_dict(),
            'device': device.type,
            'batch_size': batch_size,
            'transform': transform
        }, save_folder + 'aigc_detector_checkpoint.pth')
        print("Checkpoint saved as 'aigc_detector_checkpoint.pth'")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

