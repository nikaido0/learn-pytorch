import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms

# 类别名称(5种缺陷类别)
CLASS_NAMES = np.array(['Cr', 'In', 'Pa', 'PS', 'Rs', 'Sc'])

# 图片大小
IMG_HEIGHT = 32  # 图像的高度
IMG_WIDTH = 32  # 图像的宽度


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
model.load_state_dict(torch.load('../models/model3.pth'))
model.eval()  # 设置模型为评估模式

# 数据读取与预处理
src = cv2.imread("data/val/In/In_10.bmp")
src = cv2.resize(src, (IMG_WIDTH, IMG_HEIGHT))
src = src.astype("float32")
src = src / 255  # 归一化到 [0, 1]
# 转换为张量：torch.tensor(src) 将 NumPy 数组 src 转换为 PyTorch 张量。此时，张量的形状是 (height, width, channels)，即 HWC 格式。
# 排列维度：permute(2, 0, 1) 重新排列张量的维度。它将通道维度移动到第一个位置，得到 (channels, height, width) 的格式，即 CHW。
# 添加批次维度：unsqueeze(0) 在张量的第一个维度上添加一个新维度。这将张量的形状从 (channels, height, width) 改变为 (batch_size, channels, height, width)，其中 batch_size 为 1。这是 PyTorch 模型所需的标准输入格式。
src = torch.tensor(src).permute(2, 0, 1).unsqueeze(0)  # 转换为 [C, H, W] 并添加批次维度

# 进行预测
with torch.no_grad():  # 禁用梯度计算以加速推理
    predicted = model(src)
    score = F.softmax(predicted, dim=1)  # 应用 softmax 函数
    # 获取最大概率的类别和概率值
    _, predicted_class = torch.max(score, 1)
    max_score = torch.max(score).item()

# 打印结果
print('模型预测的结果为{}， 概率为{}%'.format(CLASS_NAMES[predicted_class.item()], max_score * 100))
