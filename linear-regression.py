# 定义数据集

# 定义数据特征
x_data = [1, 2, 3]

# 定义标签
y_data = [2, 4, 6]

# 初设化权重w
w = 4


# 定义模型
def forward(x):
    return x * w


# 定义损失函数
def loss(x_values, y_values):
    # 损失值
    cost_value = 0
    for x, y in zip(x_values, y_values):
        # 预测值
        y_pred = forward(x)
        cost_value += (y - y_pred) ** 2

    return cost_value / len(x_values)


# 定义梯度下降函数
def gradient(x_values, y_values):
    grad_value = 0
    for x, y in zip(x_values, y_values):
        grad_value += 2 * (w * x - y) * x

    return grad_value / len(x_values)


# 学习率
learning_rate = 0.01

for epoch in range(100):
    cost_val = loss(x_data, y_data)
    grad_val = gradient(x_data, y_data)

    w = w - learning_rate * grad_val

    print('Epoch: ', epoch, 'Cost: ', cost_val, 'Grad: ', grad_val, 'W: ', w)

print('进行推理', forward(4))
