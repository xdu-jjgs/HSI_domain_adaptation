import torch
import matplotlib.pyplot as plt

from models.utils.functions import adjusted_sigmoid

# 设置k和c的值
k1, c1 = 1, 2.5
k2, c2 = 2, 3

# 创建一个Tensor，包含正值
x = torch.linspace(0, 10, 100)
# 计算函数值
y5 = adjusted_sigmoid(x, k=k1, s=c1) # k=1, c=3
y6 = adjusted_sigmoid(x, k=k2, s=c2) # k=2, c=3

# 绘制图像
plt.figure(figsize=(12, 6))
plt.plot(x.numpy(), y5.numpy(), label='k=1, c=2.5')
plt.plot(x.numpy(), y6.numpy(), label='k=2, c=3')
plt.title("Adjusted Sigmoid Functions with Different Parameters")
plt.xlabel("x")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()