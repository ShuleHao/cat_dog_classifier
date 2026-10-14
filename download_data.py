# -*- coding: utf-8 -*-
"""
猫狗图片数据 —— 一键下载与整理（学生友好）

用法:
    python download_data.py

做的事情:
    1. 从 Zenodo 下载开源的猫狗图片子集（约 30MB，无需注册/登录）；
    2. 自动解压，按 cat / dog 整理成训练脚本要求的目录结构：
         data/train/cat  data/train/dog  data/val/cat  data/val/dog

数据来源与许可:
    名称: dogs_vs_cats_subset_kaggle（取自 Kaggle Dogs vs. Cats 竞赛）
    链接: https://zenodo.org/records/10997322
    内容: 训练图 1000 张（猫 500 + 狗 500），验证图 200 张（猫 100 + 狗 100）
    许可: CC BY 4.0（教学使用需标注来源，本项目已在 README 中注明）

如果下载失败（网络原因），请手动下载后按 data/README.txt 的结构放好图片。
"""
import os
import shutil
import sys
import urllib.request
import zipfile

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ZENODO_URL = "https://zenodo.org/records/10997322/files/dogs_vs_cats_subset.zip?download=1"
ZIP_PATH = os.path.join(DATA_ROOT, "dogs_vs_cats_subset.zip")


def download(url, dest, desc="下载"):
    """带进度条地下载文件。"""

    def report(count, block_size, total_size):
        done = count * block_size
        if total_size > 0:
            pct = min(100.0, done * 100.0 / total_size)
            sys.stdout.write(
                f"\r{desc}: {pct:5.1f}% ({done / 1048576:.1f}MB / {total_size / 1048576:.1f}MB)"
            )
            sys.stdout.flush()

    print(f"{desc} {url}")
    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()


def organize():
    """把 zip 里 train/val 的 cat.*.jpg、dog.*.jpg 复制到标准目录结构。"""
    extract_dir = os.path.join(DATA_ROOT, "_tmp_extract")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(extract_dir)
    src_root = os.path.join(extract_dir, "dogs_vs_cats_subset")

    for split in ("train", "val"):
        for cls, prefix in (("cat", "cat."), ("dog", "dog.")):
            dst = os.path.join(DATA_ROOT, split, cls)
            os.makedirs(dst, exist_ok=True)
            src_dir = os.path.join(src_root, split)
            n = 0
            for name in os.listdir(src_dir):
                if name.lower().startswith(prefix) and name.lower().endswith((".jpg", ".jpeg", ".png")):
                    shutil.copy2(os.path.join(src_dir, name), os.path.join(dst, name))
                    n += 1
            print(f"{split}/{cls}: {n} 张")

    shutil.rmtree(extract_dir, ignore_errors=True)


def main():
    os.makedirs(DATA_ROOT, exist_ok=True)

    # 幂等：已经有数据就跳过，避免重复下载
    cat_dir = os.path.join(DATA_ROOT, "train", "cat")
    if os.path.isdir(cat_dir) and os.listdir(cat_dir):
        print("data/train 里已经有图片，跳过下载。")
        print("想重新下载？先手动删除 data/train 和 data/val 两个目录再运行本脚本。")
        return

    try:
        download(ZENODO_URL, ZIP_PATH)
    except Exception as e:
        print(f"\n下载失败: {e}")
        print("请检查网络，或手动下载数据（直链见 README.md 的『数据来源』）。")
        sys.exit(1)

    organize()
    os.remove(ZIP_PATH)
    print("完成！现在可以直接开始训练: python train.py --data-root data")


if __name__ == "__main__":
    main()
