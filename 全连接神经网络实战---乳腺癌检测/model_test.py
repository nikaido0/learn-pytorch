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


# 用模型进行预测
model = Net()
model.load_state_dict(torch.load('../models/model.pth'))
model.eval()  # 设置为评估模式

# 用模型进行预测
with torch.no_grad():
    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs, 1)

# 将预测结果转化为类别标签
result = ["良性" if label == 0 else "恶性" for label in predicted]
print(result)

# 打印模型的精确度和召回
report = classification_report(y_test, predicted, labels=[0, 1], target_names=["良性", '恶性'])
print(report)
