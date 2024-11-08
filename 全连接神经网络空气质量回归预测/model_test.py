import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error

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
model.load_state_dict(torch.load('../models/model2.pth'))
model.eval()  # 设置为评估模式

# 利用训练好的模型进行预测
with torch.no_grad():
    yhat = model(x_test_tensor)
    # 将 PyTorch 的 Tensor 转换为 NumPy 数组
    yhat = yhat.numpy()

# 进行预测值的反归一化
# axis=1 的意思是 沿着列的方向进行拼接，即把两个数组按列合并。
inv_yhat = concatenate((x_test, yhat), axis=1)
inv_yhat = sc.inverse_transform(inv_yhat)
prediction = inv_yhat[:, -1]

# 反归一化真实值
inv_y = concatenate((x_test, y_test.reshape(-1, 1)), axis=1)
inv_y = sc.inverse_transform(inv_y)
real = inv_y[:, -1]

# 均方根误差（RMSE）和 平均绝对百分比误差（MAPE） rmse 和 mape 是回归问题中常用的 评估指标，
rmse = sqrt(mean_squared_error(real, prediction))
mape = np.mean(np.abs((real - prediction) / real))

print('RMSE:', rmse)
print('MAPE:', mape)

# 绘制真实值和预测值的对比图
plt.plot(prediction, label='预测值')
plt.plot(real, label="真实值")
plt.title("全连接神经网络空气质量预测对比图")
plt.legend()
plt.show()
