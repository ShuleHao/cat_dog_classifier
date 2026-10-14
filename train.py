# -*- coding: utf-8 -*-
"""
猫狗图片分类 —— 训练脚本

用法示例:
    python train.py --data-root data --epochs 30 --batch-size 32 --model simple
    python train.py --model resnet18            # 试试迁移学习

训练完成后会在 --save-dir 下保存:
    best_model.pth        验证集准确率最高的模型（预测时用它）
    training_curves.png   训练过程曲线（损失 / 准确率）
"""
import argparse
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # 不弹窗口，直接把图保存成文件（Windows 下更稳定）
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import create_model

# ImageNet 的均值/标准差，迁移学习模型就是用这套参数预训练的
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def make_transforms(image_size=224):
    """训练集加数据增强，验证集只做必要的缩放（保证评估稳定）。"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机左右翻转：猫狗左右对称，白赚一倍样本
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),                   # PIL 图片 -> [0,1] 的 Tensor
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_transform, val_transform


def set_seed(seed=42):
    """固定随机种子，让每次训练结果可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def plot_curves(history, save_path):
    """把训练/验证的 loss 和 acc 画成两张子图保存。"""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], "o-", label="train loss")
    axes[0].plot(epochs, history["val_loss"], "o-", label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], "o-", label="train acc")
    axes[1].plot(epochs, history["val_acc"], "o-", label="val acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="猫狗图片分类训练")
    parser.add_argument("--data-root", default="data",
                        help="数据根目录，里面必须有 train/ 和 val/，每个下面有 cat/、dog/ 子文件夹")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", default="simple", choices=["simple", "resnet18"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-dir", default="saved")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"使用设备: {device}")

    train_root = os.path.join(args.data_root, "train")
    val_root = os.path.join(args.data_root, "val")
    if not (os.path.isdir(train_root) and os.path.isdir(val_root)):
        raise FileNotFoundError(
            f"找不到数据目录 {train_root} 或 {val_root}。"
            "请先按 README.md 的『数据准备』一节把图片放好。"
        )

    train_transform, val_transform = make_transforms(args.image_size)
    # ImageFolder：自动按子文件夹名（cat / dog）作为类别标签
    train_set = datasets.ImageFolder(train_root, transform=train_transform)
    val_set = datasets.ImageFolder(val_root, transform=val_transform)
    classes = train_set.classes
    print(f"类别: {classes}   训练集: {len(train_set)} 张   验证集: {len(val_set)} 张")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = create_model(args.model, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()          # 多分类交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"epoch {epoch:3d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - start:.1f}s")

        # 只在验证集准确率创新高时保存，保证留下的是最好的模型
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": best_acc,
                "image_size": args.image_size,
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  -> 保存最优模型 (val acc {best_acc:.4f})")

    plot_curves(history, os.path.join(args.save_dir, "training_curves.png"))
    print(f"\n训练完成。最优验证集准确率: {best_acc:.4f}")
    print(f"模型保存在: {os.path.join(args.save_dir, 'best_model.pth')}")
    print(f"训练曲线图保存在: {os.path.join(args.save_dir, 'training_curves.png')}")


if __name__ == "__main__":
    main()
