import torch
import matplotlib.pyplot as plt
import numpy as np
import math

class DynamicWeightScheduler:
    def __init__(self, total_epochs):
        self.total_epochs = torch.tensor(total_epochs, dtype=torch.float32)
        self.init_kl=torch.tensor(0.7)   
        self.kl_decay = torch.tensor(0.98)    
        self.min_ce = torch.tensor(0.3)      
        self.max_ce = torch.tensor(0.7)
        self.init_neg = torch.tensor(0.2)  
        self.final_neg = torch.tensor(0.05)
        self.decay_epoch = torch.tensor(total_epochs * 0.6)

    def get_weights(self, epoch):
        epoch_tensor = torch.tensor(epoch, dtype=torch.float32)
        # 计算各项权重（未归一化）
        w_kl = self.init_kl * (self.kl_decay ** epoch_tensor)
        w_ce = self.min_ce + 0.5 * (self.max_ce - self.min_ce) * \
                (1 + math.cos(math.pi * (1 - epoch/self.total_epochs)))
        w_neg = self.init_neg if epoch < self.decay_epoch else self.final_neg
        weight_sum = w_kl + w_ce + w_neg
        w_kl = w_kl / weight_sum
        w_ce = w_ce / weight_sum
        w_neg = w_neg / weight_sum
        return w_kl.item(), w_ce.item(), w_neg.item()

    def get_normalized_weights(self, epoch):
        w_kl, w_ce, w_neg = self.get_weights(epoch)
        weights = torch.tensor([w_kl, w_ce, w_neg], dtype=torch.float32)
        weights_normalized = torch.softmax(weights, dim=0)
        return weights_normalized.tolist()

# 设置总训练轮数为90
total_epochs = 90
scheduler = DynamicWeightScheduler(total_epochs=total_epochs)

epochs = list(range(total_epochs+1))
w_kl_list = []
w_ce_list = []
w_neg_list = []
w_kl_norm = []
w_ce_norm = []
w_neg_norm = []

for epoch in epochs:
    w_kl, w_ce, w_neg = scheduler.get_weights(epoch)
    w_kl_list.append(w_kl)
    w_ce_list.append(w_ce)
    w_neg_list.append(w_neg)
    norm = scheduler.get_normalized_weights(epoch)
    w_kl_norm.append(norm[0])
    w_ce_norm.append(norm[1])
    w_neg_norm.append(norm[2])

# 打印部分轮次的数值
print("Epoch\tw_kl\t\tw_ce\t\tw_neg")
for i in range(0, total_epochs+1, 10):
    print(f"{i:02d}\t{w_kl_list[i]:.4f}\t\t{w_ce_list[i]:.4f}\t\t{w_neg_list[i]:.4f}")

# 绘制未归一化的权重曲线
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(epochs, w_kl_list, label='w_kl')
plt.plot(epochs, w_ce_list, label='w_ce')
plt.plot(epochs, w_neg_list, label='w_neg')
plt.xlabel("Epoch")
plt.ylabel("Weight value")
plt.title("Unnormalized Weights")
plt.legend()

# 绘制归一化后（softmax）的权重曲线
plt.subplot(1,2,2)
plt.plot(epochs, w_kl_norm, label='w_kl (normalized)')
plt.plot(epochs, w_ce_norm, label='w_ce (normalized)')
plt.plot(epochs, w_neg_norm, label='w_neg (normalized)')
plt.xlabel("Epoch")
plt.ylabel("Normalized weight")
plt.title("Normalized Weights (via softmax)")
plt.legend()

plt.tight_layout()
plt.show()