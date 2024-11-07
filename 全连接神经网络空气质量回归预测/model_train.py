import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False

# 导入数据
dataset = pd.read_csv('../dataset/data.csv')

# 数据归一化  先归一化，再划分数据集
# 如果我们先划分数据集再分别归一化，那么训练集和测试集的归一化范围可能不同。例如，训练集中的特征可能落在 [0, 1] 范围内，而测试集中的特征可能因数据分布不同而落在 [0.2, 0.8] 范围内
sc = MinMaxScaler(feature_range=(0, 1))
scaled = sc.fit_transform(dataset)

# 将归一化好的数据转化为 DataFrame 格式，方便数据处理
dataset_sc = pd.DataFrame(scaled)

# 提取特征和标签
X = dataset_sc.iloc[:, :-1].values  # 提取特征
Y = dataset_sc.iloc[:, -1].values  # 提取标签

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 转换为 PyTorch 的 Tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
# .view(-1, 1) 的作用是将张量 y_train_tensor 的形状调整为两维形状，其中第一维的大小自动确定，第二维的大小为 1
# 假设 y_train 是一个一维数组 [2.3, 3.1, 4.5, ...]，转换为张量时它的形状是 (n_samples,)。通过 .view(-1, 1)，它会被调整为二维形状 (n_samples, 1)。
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建 DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)


# 定义神经网络模型
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # 定义第一层全连接层，输入大小为 X.shape[1]，输出大小为 10
        self.fc1 = nn.Linear(X.shape[1], 10)
        # 定义第二层全连接层，输入大小为 10，输出大小为 10
        self.fc2 = nn.Linear(10, 10)
        # 定义第三层全连接层，输入大小为 10，输出大小为 1
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        # 使用 ReLU 激活函数对第一层输出进行处理
        x = torch.relu(self.fc1(x))
        # 使用 ReLU 激活函数对第二层输出进行处理
        x = torch.relu(self.fc2(x))
        # 输出层无激活函数，适合回归问题
        x = self.fc3(x)
        return x


# 实例化模型
model = NeuralNet()

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数，适用于回归问题
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

# 训练模型
num_epochs = 100
train_losses = []  # 用于存储训练集的损失值
val_losses = []  # 用于存储验证集的损失值

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    # 遍历训练数据进行训练
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item()  # 累加每批次的损失

    train_losses.append(running_loss / len(train_loader))  # 记录平均训练损失

    # 验证模型
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_losses.append(val_loss / len(test_loader))  # 记录平均验证损失

    # 每隔10轮输出一次训练和验证的损失值
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

# 保存模型
torch.save(model.state_dict(), '../models/model2.pth')

# 绘制训练集和验证集的loss值对比图
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.title("全连接神经网络loss值图")  # 设置图标题
plt.legend()
plt.show()
