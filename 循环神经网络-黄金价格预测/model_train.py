import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False

# 加载历史数据文件 index_col=[0] 第一列(Date)为索引
dataset = pd.read_csv('../dataset/LBMA-GOLD.csv', index_col=[0])

# 设置训练集的长度
training_len = 1265 - 200

# 获取训练集/测试集数据
'''
dataset: 这是包含数据的 DataFrame，通常从 CSV 文件加载，如代码中的 LBMA-GOLD.csv 文件。
iloc: 是 Pandas 中的索引方法，用于按位置（位置索引）选取数据。
[:training_len]: 从数据集的第 0 行到 training_len 行（不包括 training_len 本身），相当于选取前 training_len 行的数据。
[[0]]: 选取数据集中第 0 列的数据（Date 列作为索引，不计入列编号），因此 [0] 表示的是数据集的第一个数值列
'''
training_data = dataset.iloc[:training_len, [0]]
test_data = dataset.iloc[training_len:, [0]]

# 将数据集进行归一化(0,1)之间，方便神经网络的训练 经过归一化处理后被转换为了 NumPy 数组
sc = MinMaxScaler(feature_range=(0, 1))
training_data = sc.fit_transform(training_data)
test_data = sc.fit_transform(test_data)

# 设置训练数据特征和标签
x_train, y_train = [], []
x_test, y_test = [], []

#  利用for循环，提取训练集中连续5个采样点的数据作为输入特征x_train，第6个采样点的数据作为标签
for i in range(5, len(training_data)):
    x_train.append(training_data[i - 5:i, 0])
    y_train.append(training_data[i, 0])

# 将训练集由list格式变为array格式 循环神经网络的特征结构为：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], 5, 1))

# 同理划分测试集数据
for i in range(5, len(test_data)):
    x_test.append(test_data[i - 5:i, 0])
    y_test.append(test_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 5, 1))

# 转换为 PyTorch 张量
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# 定义 LSTM 模型
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        '''
        input_size=1：每个时间步输入一个特征，通常是一个标量（如一个价格）。
        hidden_size=80：LSTM 层的隐藏层状态大小，表示 LSTM 层中记忆单元的数量，这个参数决定了模型的容量。
        batch_first=True：确保输入数据的形状是 (batch_size, sequence_length, input_size)，即样本数、序列长度和每个时间步的特征数。
        '''
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=80, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=80, hidden_size=100, batch_first=True)
        self.linear1 = nn.Linear(100, 10)
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        """
        LSTM 层会输出两个对象：
        output：这是 LSTM 层的输出，形状通常为 (batch_size, sequence_length, hidden_size)，其中 hidden_size 是 LSTM 的隐藏层维度（由你设置的 hidden_size=80 决定）。
        (h_n, c_n)：这是 LSTM 的隐藏状态和细胞状态（cell state），分别表示 LSTM 中每个时间步的输出隐藏状态和记忆状态。
        h_n 的形状为 (num_layers, batch_size, hidden_size)，c_n 的形状也为 (num_layers, batch_size, hidden_size)。在默认情况下，LSTM 会返回这两个状态。
        """
        x, _ = self.lstm1(x)
        x = F.relu(x)
        x, _ = self.lstm2(x)
        x = F.relu(x)
        x = x[:, -1, :]  # 取最后一个时间步
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# 初始化模型、损失函数和优化器
model = LSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
batch_size = 32
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]

        # 前向传播
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output.squeeze(), batch_y)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    # 计算并记录损失
    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(x_train).squeeze(), y_train).item()
        val_loss = criterion(model(x_test).squeeze(), y_test).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), '../models/model4.pth')

# 绘制训练集和测试集的loss值对比图
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.title("LSTM神经网络loss值")
plt.legend()
plt.show()
