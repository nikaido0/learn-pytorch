import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据目录路径
data_train = './data/train/'  # 训练集路径
# pathlib.Path转为Path对象用于方便处理文件和目录路径
data_train = pathlib.Path(data_train)

data_val = './data/val/'  # 验证集路径
data_val = pathlib.Path(data_val)

# 类别名称(5种缺陷类别)
CLASS_NAMES = np.array(['Cr', 'In', 'Pa', 'PS', 'Rs', 'Sc'])  # 数据类别列表

# 设置批次大小和图像大小
BATCH_SIZE = 64  # 每个批次的大小
IMG_HEIGHT = 32  # 图像的高度
IMG_WIDTH = 32  # 图像的宽度

# 数据预处理：归一化处理
# transforms.Compose 是一个容器，用来将多个图像预处理操作串联起来执行。每个操作会依次对图像进行处理。
# 数据预处理，包括将图像尺寸调整、转换为Tensor、按通道标准化
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),  # 将图像调整到指定大小
    transforms.ToTensor(),  # 转换为Tensor类型，像素值范围调整为[0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对图片的像素值进行标准化
])

# 加载训练集和验证集
# 会读取 root 指定的目录下的图片文件，并将这些图片按目录结构进行分组
# ImageFolder 会遍历这些子文件夹，将其中的图像文件作为数据读取，并将它们的标签设为对应文件夹的名称
train_dataset = datasets.ImageFolder(root=str(data_train), transform=transform)  # 训练集
val_dataset = datasets.ImageFolder(root=str(data_val), transform=transform)  # 验证集

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 训练集DataLoader
# 验证集通常不打乱
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 验证集DataLoader


# 定义卷积神经网络模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 定义卷积层：输入通道3（RGB图像），输出通道6，卷积核大小5x5
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        # 定义池化层：2x2大小的最大池化，步幅为2
        self.pool = nn.MaxPool2d(2, 2)
        # 第二个卷积层：输入通道6，输出通道16，卷积核大小5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第三个卷积层：输入通道16，输出通道120，卷积核大小5x5
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        # 动态计算展平后的特征数
        self.flattened_size = self._get_flattened_size()

        # 全连接层：输入是展平后的特征大小，输出是84个神经元
        self.fc1 = nn.Linear(self.flattened_size, 84)
        # 输出层：6个神经元，对应6个类别
        self.fc2 = nn.Linear(84, 6)

    # 动态计算展平后的维数
    def _get_flattened_size(self):
        with torch.no_grad():
            # 用假数据“运行”一遍前几层，确定展平后的特征数
            x = torch.zeros(1, 3, 32, 32)  # 假设输入为32x32的RGB图像
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = torch.relu(self.conv3(x))
            return x.numel()  # 展平后的元素个数,计算总的元素数量

    def forward(self, x):
        # 卷积层1 + 激活函数（ReLU） + 池化层
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积层2 + 激活函数（ReLU） + 池化层
        x = self.pool(F.relu(self.conv2(x)))
        # 卷积层3 + 激活函数（ReLU）
        x = F.relu(self.conv3(x))
        # 展平层，将多维的输入一维化，传入全连接层
        x = x.view(-1, self.flattened_size)  # 展平后的张量维度
        # 全连接层1 + 激活函数（ReLU）
        x = F.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x


# 实例化模型
model = CNNModel()

# 定义损失函数：交叉熵损失（用于分类问题）
# nn.CrossEntropyLoss 损失函数在内部已经包含了 softmax 操作。当你使用这个损失函数时，你不需要在模型的输出层显式地应用 softmax
criterion = nn.CrossEntropyLoss()

# 定义优化器：使用Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50  # 训练的轮数
train_losses = []  # 存储每个epoch的训练损失
val_losses = []  # 存储每个epoch的验证损失
train_accuracies = []  # 存储每个epoch的训练准确率
val_accuracies = []  # 存储每个epoch的验证准确率

for epoch in range(num_epochs):
    running_loss = 0.0  # 用于累积整个训练周期（epoch）中的总损失
    correct_train = 0  # 用于记录在训练过程中模型正确预测的样本数量
    total_train = 0  # 用于记录在训练过程中模型总共预测的样本数量

    for inputs, labels in train_loader:
        optimizer.zero_grad()  # 梯度清空
        outputs = model(inputs)  # 正向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新权重

        running_loss += loss.item()  # 累加训练损失
        _, predicted = torch.max(outputs, 1)  # 获取预测的类别
        # 张量在第 0 维的大小，通常在 PyTorch 中表示批次（batch）中的样本数量
        total_train += labels.size(0)  # 总的样本数
        correct_train += (predicted == labels).sum().item()  # 计算正确预测的样本数

    # 记录每个epoch的训练损失和准确率
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct_train / total_train)

    # 验证模型
    model.eval()  # 设置模型为评估模式
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():  # 评估时不需要计算梯度
        for inputs, labels in val_loader:
            outputs = model(inputs)  # 正向传播
            loss = criterion(outputs, labels)  # 计算验证损失
            val_loss += loss.item()  # 累加验证损失
            _, predicted = torch.max(outputs, 1)  # 获取预测的类别
            total_val += labels.size(0)  # 总的样本数
            correct_val += (predicted == labels).sum().item()  # 计算正确预测的样本数

    # 记录每个epoch的验证损失和准确率
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(correct_val / total_val)

    # 打印当前epoch的训练和验证损失及准确率
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, "
          f"Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

# 保存训练好的模型
torch.save(model.state_dict(), "../models/model3.pth")

# 绘制训练过程图像
# 绘制损失值图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.title("CNN Loss")
plt.legend()

# 绘制准确率图
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='train')
plt.plot(val_accuracies, label='val')
plt.title("CNN Accuracy")
plt.legend()

plt.show()
