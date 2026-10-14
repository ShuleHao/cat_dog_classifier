# 数据说明

## 推荐做法：一键下载

在项目根目录运行：

    python download_data.py

脚本会从 Zenodo 下载开源猫狗图片子集（约 30MB，无需注册/登录），
并自动整理成下面的目录结构：

    data/
    ├── train/
    │   ├── cat/     # 500 张猫的训练图
    │   └── dog/     # 500 张狗的训练图
    └── val/
        ├── cat/     # 100 张猫的验证图
        └── dog/     # 100 张狗的验证图

## 手动下载（备选）

- Zenodo 直链（本项目默认数据，29.6MB）：
  https://zenodo.org/records/10997322/files/dogs_vs_cats_subset.zip?download=1
  解压后把 train / val 里的 cat.*.jpg、dog.*.jpg 分别放进上面的目录。
- Kaggle 全量（需注册）：https://www.kaggle.com/c/dogs-vs-cats/data
- 或自己收集：随便找几十张猫和狗的照片，按比例分到 train/ 和 val/。

## 许可与引用

默认数据来自 Zenodo 上的 dogs_vs_cats_subset_kaggle（取自 Kaggle 竞赛 Dogs vs. Cats），
许可为 CC BY 4.0：教学/科研使用请注明来源
（Zenodo 记录：https://zenodo.org/records/10997322）。

## 几个提醒

- train 和 val 的图片来源尽量不同（比如 train 用一批网站的照片，val 用另一批），
  否则验证集准确率会虚高，无法真实反映模型水平。
- 图片尺寸可以不一样，脚本会自动统一缩放，不需要提前处理。
- data/train 和 data/val 已在 .gitignore 中忽略，不会随仓库提交，重跑 download_data.py 即可恢复。
