# 实验四：全连接神经网络
<div>
    <span style="float: left">学生姓名: 彭艺琼</span>
    <span style="float: right">指导老师: 胡政伟</span>
</div>
<center>学号: 202410315019 </center>
<br>

## 一、实验目的
1. 理解多层感知机（MLP）的基本原理，掌握全连接层、激活函数的作用；
2. 熟练使用PyTorch框架完成数据集加载、网络搭建、前向传播、反向传播与参数更新；
3. 掌握DataLoader、梯度清零、损失计算、优化器迭代、模型评估的标准训练流程；
4. 观察损失函数与分类准确率随训练轮次的变化规律，分析模型收敛特性；

## 二、实验环境
- 编程语言：Python 3
- 深度学习框架：PyTorch、torchvision
- 开发工具：Jupyter Notebook
- 运行设备：CPU
- 依赖库：numpy、matplotlib

## 三、实验原理
### 3.1 MNIST数据集
MNIST为经典手写数字识别数据集，包含0~9十类灰度图像，其中训练集60000张、测试集10000张，图像尺寸为28×28。为适配全连接网络输入，需将二维图像展平为784维一维向量，用于模型训练与测试。

### 3.2 MLP网络结构
多层感知机（MLP）由全连接层堆叠而成，通过激活函数引入非线性，实现复杂分类任务。
本次网络结构为：784维输入→128维隐藏层（ReLU激活）→10维输出。
ReLU激活函数有效缓解梯度消失，提升模型拟合能力。实验采用交叉熵损失函数，适配多分类任务，可衡量预测值与真实标签的偏差。

### 3.3 优化与训练过程
实验使用SGD随机梯度下降优化器，学习率设置为0.01。训练过程遵循深度学习标准迭代流程：首先清空历史梯度，通过前向传播得到预测结果，计算损失后利用反向传播求解参数梯度，最终更新网络权重，不断最小化损失函数。

### 3.4 模型评估方式
每轮训练结束后，将模型切换为评估模式，关闭梯度计算以提升推理效率。通过计算测试集分类准确率与平均损失，客观评价模型的识别精度与泛化能力。

## 四、实验代码
## 四、实验代码
本实验基于教师提供的代码框架完成，代码如下：
```python
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据集加载
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=2048, shuffle=True)
test_loader = DataLoader(test_data, batch_size=2048, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义MLP模型
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784,128)
        self.fc2 = torch.nn.Linear(128,10)
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        x1 = self.fc1(x)
        x = F.relu(x1)
        a2 = self.fc2(x)
        return a2

model = MLP().to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
epochs = 30

# 训练循环
for epoch in range(epochs):
    model.train()
    for batch_idx,(x,y) in enumerate(train_loader):
        x,y = x.view(x.shape[0],-1).to(device),y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output,y)
        loss.backward()
        optimizer.step()

    # 测试评估
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx,(x,y) in enumerate(test_loader):
            x,y = x.view(x.shape[0],-1).to(device),y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output,y)
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss = test_loss/(batch_idx+1)
    acc = correct/len(test_loader.dataset)
    print("epoch:{}, loss:{:.4f}, acc:{:.4f}".format(epoch,test_loss,acc))
```

## 五、实验结果
- epoch:29, loss:0.5981, acc:0.8638
- 现象：训练初期损失快速下降、准确率快速上升；训练后期收敛变慢，准确率提升幅度变小。

## 六、结果分析
1. 收敛趋势：随着epoch增加，测试损失持续下降，准确率逐步上升，模型正常收敛；
2. 模型上限：单纯MLP不具备卷积的局部感受野、权值共享能力，无法利用图像空间相邻像素信息，因此MNIST上最终准确率低于CNN；
3. 超参数影响：学习率0.01适中；如果学习率太大容易震荡不收敛，太小收敛速度极慢；
4. 过拟合观察：训练集效果一般优于测试集；本实验网络容量不大，过拟合现象不算严重。

## 七、实验总结
本次实验基于PyTorch搭建MLP完成MNIST手写数字分类，完整走完深度学习训练流水线。
通过实验理解了前向传播、损失计算、反向梯度传播、参数更新的完整机制；体会到激活函数带来非线性拟合能力。同时认识到全连接网络在图像任务上的短板，理解卷积网络的必要性，为后续CNN实验打下基础。
