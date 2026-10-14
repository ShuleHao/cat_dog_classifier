# -*- coding: utf-8 -*-
"""
猫狗图片分类 —— 单张图片预测

用法示例:
    python predict.py --image path/to/a_cat.jpg
    python predict.py --image path/to/a_dog.jpg --checkpoint saved/best_model.pth
"""
import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import create_model

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser(description="用训练好的模型预测单张图片")
    parser.add_argument("--image", required=True, help="图片路径")
    parser.add_argument("--checkpoint", default="saved/best_model.pth")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"图片不存在: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载训练时保存的最优模型（weights_only=False：checkpoint 里还存了类别名等非张量信息）
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = create_model(ckpt["model_name"], num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    # 和训练时验证集完全相同的预处理
    transform = transforms.Compose([
        transforms.Resize((ckpt["image_size"], ckpt["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    img = Image.open(args.image).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)   # (1, 3, H, W)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]       # 转成概率，和为 1

    classes = ckpt["classes"]
    for name, p in zip(classes, probs.tolist()):
        print(f"{name}: {p:.4f}")
    pred = classes[probs.argmax().item()]
    print(f"\n预测结果: {pred}   (置信度 {probs.max().item():.2%})")


if __name__ == "__main__":
    main()
