import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# 解决画图中文显示问题
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
dataset = pd.read_csv('../dataset/breast_cancer_data.csv')

# 提取数据中的特征和标签(.values将 DataFrame 转换为 NumPy 数组 很多机器学习库（例如 scikit-learn 或深度学习框架）通常要求数据输入是 NumPy 数组格式，而不是 DataFrame。
X = dataset.iloc[:, :-1].values
Y = dataset['target'].values

# 划分训练集和测试集(20%) random_state=42 的主要目的是确保数据划分的结果是可重复的。每次运行代码时，这样会得到相同的训练集和测试集划分
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 将数据标签转化为 one-hot 编码
y_train_one = np.eye(2)[y_train]
y_test_one = np.eye(2)[y_test]

# 归一化 对数据进行归一化，将特征的值压缩到指定的范围（默认是 [0, 1]）
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# 转换为 Tensor(Tensor 是 PyTorch 中的基本数据结构，所有的操作和计算都需要在 Tensor 上进行。
# dtype=torch.float32 是指定 Tensor 数据类型为 32 位浮点数，这对于深度学习任务来说是标准的数据类型，能保证数值的精度并且在计算上高效)
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_one, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_one, dtype=torch.float32)

# 创建 DataLoader,按批次加载数据
train_dataset = TensorDataset(x_train_tensor,
                              y_train_tensor)  # 将训练数据 x_train_tensor 和标签 y_train_tensor 封装成一个数据集，每个样本是 (input, target) 对，输入是特征，目标是标签。
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
# DataLoader 是用于批处理数据加载的工具，它可以自动按批次（batch）加载数据，支持多线程并行加载，并能在每个 epoch 后打乱数据（shuffle），以提高训练的效率和泛化能力
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # in_features：输入数据的特征数（每个样本的维度）。
        # out_features：输出数据的特征数（每个样本变换后的维度）
        self.fc1 = nn.Linear(x_train.shape[1], 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # 在分类问题中，dim=1 是常见的设置，因为每一行通常代表一个样本的预测分数，而每一列代表不同类别的分数
        # dim=1 表示 softmax 会沿着每一行进行操作，也就是说，它对每个样本的类别分数进行归一化。每一行将得到一个在 num_classes 维度上的概率分布，所有的概率值加起来等于 1
        x = torch.softmax(self.fc3(x), dim=1)
        return x


model = Net()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1100  # 训练的总轮数，表示神经网络将通过训练数据集多少次
train_losses = []  # 用于存储每个训练轮次的训练集损失值（Loss），用于跟踪训练过程中损失的变化
val_losses = []  # 用于存储每个训练轮次的验证集损失值（Loss），用于评估模型在验证集上的表现
train_accuracies = []  # 用于存储每个训练轮次的训练集准确率，用于跟踪训练过程中模型的分类准确率
val_accuracies = []  # 用于存储每个训练轮次的验证集准确率，用于评估模型在验证集上的分类准确率

# 训练循环
for epoch in range(num_epochs):  # 训练模型的总轮数
    model.train()  # 将模型设置为训练模式
    running_loss = 0.0  # 初始化一个变量，用于记录当前轮次的训练损失
    correct = 0  # 记录正确预测的样本数
    total = 0  # 记录总样本数

    # 遍历训练数据集
    for inputs, labels in train_loader:  # 从训练数据加载器中获取一批数据
        optimizer.zero_grad()  # 清除梯度缓存，以便进行反向传播
        outputs = model(inputs)  # 将输入数据传入模型，得到预测输出
        loss = criterion(outputs, labels.argmax(dim=1))  # 计算损失函数，argmax是获取标签的最大值索引
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        running_loss += loss.item()  # 累加损失值，loss.item()是获取损失的数值
        # torch.max() 函数返回两个值：
        # 最大值：在指定维度上，张量中每个元素的最大值。
        # 索引：每个最大值的索引位置。
        _, predicted = torch.max(outputs, 1)  # 获取每个样本预测的类别索引 dim 指定要沿着哪个维度计算最大值。例如，dim=1 表示沿着每一行（每个样本）计算最大值
        total += labels.size(0)  # 统计总样本数
        correct += (predicted == labels.argmax(dim=1)).sum().item()  # 统计正确预测的样本数

    # 计算并保存训练损失和准确率
    train_losses.append(running_loss / len(train_loader))  # 计算平均损失，并保存到列表中
    train_accuracies.append(100 * correct / total)  # 计算训练集的准确率，并保存到列表中

    # 验证模型
    model.eval()  # 将模型设置为评估模式，禁用诸如dropout等训练时特有的机制
    val_loss = 0.0  # 初始化验证损失
    correct = 0  # 记录验证集上的正确预测数
    total = 0  # 记录验证集上的总样本数

    with torch.no_grad():  # 在验证过程中不需要计算梯度
        for inputs, labels in test_loader:  # 遍历验证集
            outputs = model(inputs)  # 获取模型输出
            loss = criterion(outputs, labels.argmax(dim=1))  # 计算损失
            val_loss += loss.item()  # 累加损失值
            _, predicted = torch.max(outputs, 1)  # 获取每个样本的预测类别
            total += labels.size(0)  # 统计总样本数
            correct += (predicted == labels.argmax(dim=1)).sum().item()  # 统计正确预测的样本数

    # 计算并保存验证损失和准确率
    val_losses.append(val_loss / len(test_loader))  # 计算验证集的平均损失
    val_accuracies.append(100 * correct / total)  # 计算验证集的准确率

    # 每10个epoch输出一次训练和验证结果
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, '
              f'Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, '
              f'Val Acc: {val_accuracies[-1]:.2f}%')

# 保存模型的状态字典
torch.save(model.state_dict(), '../models/model.pth')

# 绘制训练和验证集的 loss 值对比
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("全连接神经网络 Loss 值图")
plt.legend()
plt.show()

# 绘制训练和验证集的准确率对比图
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title("全连接神经网络 Accuracy 值图")
plt.legend()
plt.show()
