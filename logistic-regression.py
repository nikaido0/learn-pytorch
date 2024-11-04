# 逻辑回归实战---乳腺癌检测
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

# 读取数据
dataset = pd.read_csv('./dataset/breast_cancer_data.csv')

# 提取特征和标签
X = dataset.iloc[:, :-1]
Y = dataset['target']

# 数据集划分(80%训练集/20%测试集)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# 数据归一化(到（0，1）之间)
scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# 定义逻辑回归模型
lr = LogisticRegression()
# 模型训练
lr.fit(x_train, y_train)

# 打印模型的参数
# print('w:', lr.coef_)
# print('b', lr.intercept_)

# 利用训练好的模型进行推理测试
y_pred = lr.predict(x_test)
# print(y_pred)

# 打印预测结果的概率
y_pred_proba = lr.predict_proba(x_test)
# print(y_pred, y_pred_proba)

# 获取恶性肿瘤的概率
pre_list = y_pred_proba[:, 1]
# print(pre_list)

# 设置阈值
thresholds = 0.3

# 设置保存结果的列表
result = []
result_name = []

for i in range(len(pre_list)):
    if pre_list[i] > thresholds:
        result.append(1)
        result_name.append('恶性')
    else:
        result.append(0)
        result_name.append('良性')

# 打印阈值调整后的结果
# print(result)
# print(result_name)

# 输出结果的精确率和召回还有f1值
report = classification_report(y_test, result, labels=[0, 1], target_names=['良性肿瘤', '恶性肿瘤'])
print(report)
