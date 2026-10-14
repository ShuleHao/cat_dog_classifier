# 猫狗图片分类（PyTorch 入门项目）

用卷积神经网络（CNN）判断一张图是猫还是狗。项目不大，但图像分类该有的环节都有：

**数据准备 → 数据增强 → 模型训练 → 模型保存 → 单张图片预测**

## 项目结构

```
cat_dog_classifier/
├── model.py            # 模型定义：自己搭的 SimpleCNN + 可选的 ResNet18 迁移学习
├── train.py            # 训练脚本：训练、验证、画曲线、保存最优模型
├── predict.py          # 预测脚本：对单张图片输出 猫/狗 的概率
├── download_data.py    # 【一键】下载并整理猫狗数据集（约 30MB，无需注册）
├── requirements.txt    # 依赖（torch / torchvision / pillow / matplotlib）
├── data/
│   ├── README.txt      # 数据目录结构与手动获取方法
│   └── train/ val/     # 运行 download_data.py 后自动生成
└── saved/              # 训练后自动生成：best_model.pth + training_curves.png
```

## 一、安装环境

建议用 conda 或 venv 建一个独立环境：

```bash
# conda（推荐，会自动配好 CUDA 版 torch）
conda create -n catdog python=3.10 -y
conda activate catdog
pip install torch torchvision pillow matplotlib

# 或者用 venv + CPU 版
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 二、准备数据

### 方法一（推荐）：一键下载，clone 后直接跑

```bash
python download_data.py
```

脚本会从 Zenodo 下载一个开源的猫狗图片子集（约 30MB，**不用注册、不用登录**），
自动整理成训练要用的目录结构（train 每类 500 张 + val 每类 100 张）。
下完就能直接训练。

### 方法二：手动下载 / 自己收集

数据目录结构如下（文件夹名必须是 cat / dog）：

```
data/
├── train/cat/  train/dog/    # 训练图片
└── val/cat/    val/dog/      # 验证图片
```

### 数据来源

| 数据集 | 链接 | 说明 |
| --- | --- | --- |
| dogs_vs_cats_subset（本项目默认，推荐） | https://zenodo.org/records/10997322 | 1000 张训练图 + 200 张验证图，29.6MB，CC BY 4.0 许可（教学使用请注明来源） |
| Kaggle Dogs vs. Cats 全量（25,000 张） | https://www.kaggle.com/c/dogs-vs-cats/data | 需要注册 Kaggle 账号，图片更多、效果更好 |

> 提示：train 和 val 的图片来源尽量不同，否则验证集准确率会虚高，无法真实反映模型水平。

## 三、训练

```bash
# 用自己搭的 CNN（入门推荐，能看清每一层的作用）
# CPU 上每个 epoch 约 40 秒（1000 张图），跑 10-15 个 epoch 就够看效果
python train.py --data-root data --epochs 15 --batch-size 32

# 换 ResNet18 迁移学习（收敛更快、准确率更高）
python train.py --model resnet18 --epochs 10
```

训练结束后，`saved/` 里会多出两个文件：

- `best_model.pth` —— 验证集准确率最高的模型，预测时用这个
- `training_curves.png` —— 训练/验证的 loss 和 acc 曲线

## 四、预测

```bash
python predict.py --image 某张猫或狗的照片.jpg
```

输出示例：

```
cat: 0.9876
dog: 0.0124

预测结果: cat   (置信度 98.76%)
```

## 五、代码里值得看的地方

| 文件 | 学什么 |
| --- | --- |
| model.py | 卷积层 / 池化层 / 全连接层各自的作用；为什么卷积核通道数逐层翻倍；迁移学习改最后一层 |
| train.py | 训练-验证循环的标准写法；数据增强（翻转、颜色抖动）为什么有用；只保存验证集最好的模型 |
| predict.py | 推理时为什么 `torch.no_grad()`；预处理必须和训练时一致 |

训练准确率很高、验证准确率却上不去，这基本就是**过拟合**。
常见办法：加数据、加强数据增强、加 Dropout、把网络改小，或者直接换迁移学习模型。

## 六、进阶方向

1. 换更深的网络：ResNet34/50、EfficientNet、Vision Transformer；
2. 更大的数据 + 更长时间训练，把验证准确率从 90% 推到 95%+；
3. 学习率调度（lr_scheduler）、早停（early stopping）；
4. 用 TensorBoard 或 wandb 记录训练过程；
5. 把 `predict.py` 扩展成一个批量预测或 Web 接口（FastAPI）。
