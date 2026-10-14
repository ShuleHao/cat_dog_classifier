# -*- coding: utf-8 -*-
"""
模型定义：一个从零搭建的简易 CNN，外加一个可选的 ResNet18 迁移学习模型。

理解 CNN 的关键概念：
- 卷积层（Conv2d）：用一个小窗口（卷积核）扫过整张图，提取局部特征，比如边缘、纹理。
- 激活函数（ReLU）：把负数变成 0，给网络引入非线性，让它能拟合更复杂的关系。
- 池化层（MaxPool2d）：把特征图缩小一半，只保留每个区域里最强的响应，
  既减少计算量，又让模型对物体的位置不那么敏感（平移不变性）。
- 全连接层（Linear）：把前面提取到的特征汇总，输出每个类别的得分。
"""
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """自己搭的简易卷积网络（入门推荐）。

    结构：卷积 -> ReLU -> 池化，重复 4 次；
    然后用全局平均池化把特征压成 1x1，再接两层全连接输出 2 个类别得分。
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # 通道数逐步翻倍：32 -> 64 -> 128 -> 256，让高层学到更抽象的特征
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            # AdaptiveAvgPool2d：不管输入多大，都压成 1x1，省去手动计算尺寸
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.5),            # 训练时随机丢掉一半神经元，防止过拟合
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_model(model_name="simple", num_classes=2):
    """根据名字创建模型。

    - "simple"   ：上面自己搭的 CNN，训练稍慢但能看清每一层的原理；
    - "resnet18" ：直接使用 ImageNet 预训练好的 ResNet18，只改最后一层，
                   这种“迁移学习”方法收敛快、准确率高，是实战中最常用的套路。
    """
    if model_name == "simple":
        return SimpleCNN(num_classes)
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # 替换最后一层
        return model
    raise ValueError(f"未知模型名称: {model_name}，可选 'simple' 或 'resnet18'")


if __name__ == "__main__":
    # 快速自检：打印网络结构
    net = create_model("simple")
    print(net)
