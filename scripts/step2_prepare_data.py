"""
Bước 2 — Tiền xử lý + chia train/val/test + data augmentation.

Việc làm:
  1. Liệt kê toàn bộ ảnh + nhãn (dùng lại class_names.json từ Bước 1).
  2. Chia stratified 80/10/10 (train/val/test) — giữ đúng tỉ lệ lớp ở mỗi
     tập, kể cả lớp ít ảnh nhất (Potato___healthy, 152 ảnh) vẫn có mặt đủ
     ở cả 3 tập thay vì random split có thể làm 1 tập thiếu hẳn 1 lớp.
  3. Lưu danh sách split ra outputs/splits/{train,val,test}.csv để Bước 3
     dùng lại đúng y hệt (không train/test lẫn ảnh).
  4. Tính class_weight từ tập train, lưu outputs/class_weights.json — dùng
     ở Bước 3 để model không lơ là các lớp ít ảnh.
  5. Vẽ demo augmentation: so sánh ảnh gốc vs ảnh sau khi tăng cường, để
     thấy trực quan augmentation đang mô phỏng điều gì.

Chạy:
    python scripts/step2_prepare_data.py [--data-dir data]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_pipeline import (
    IMG_SIZE,
    build_augmentation,
    make_dataset,
    save_split_csv,
    stratified_split,
)
from utils import find_color_dir, list_files_labels, load_class_names

OUT_DIR = Path("outputs")
SPLIT_DIR = OUT_DIR / "splits"


def print_split_stats(split: dict, class_names: list[str]):
    print("\n================  CHIA TẬP  ================")
    for name, (paths, labels) in split.items():
        print(f"{name:5s}: {len(paths):6,d} ảnh")
    print("==============================================\n")

    # kiểm tra lớp ít ảnh nhất vẫn có mặt đủ ở cả 3 tập
    least_class_idx = Counter(split["train"][1] + split["val"][1] + split["test"][1])
    least_idx = min(least_class_idx, key=least_class_idx.get)
    print(f"Kiểm tra lớp ít ảnh nhất ({class_names[least_idx]}):")
    for name, (_, labels) in split.items():
        n = sum(1 for y in labels if y == least_idx)
        print(f"  {name:5s}: {n} ảnh")
    print()


def compute_class_weights(train_labels, class_names: list[str]) -> dict:
    counts = Counter(train_labels)
    total = len(train_labels)
    n_classes = len(class_names)
    weights = {int(i): total / (n_classes * counts[i]) for i in range(n_classes)}
    return weights


def plot_augmentation_demo(paths, out_dir: Path, n_samples=4, n_variants=4):
    aug = build_augmentation()
    rng = np.random.default_rng(0)
    sample_paths = rng.choice(paths, size=n_samples, replace=False)

    cols = n_variants + 1  # gốc + các bản augment
    plt.figure(figsize=(cols * 2.2, n_samples * 2.4))
    for row, p in enumerate(sample_paths):
        raw = tf.io.read_file(p)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

        ax = plt.subplot(n_samples, cols, row * cols + 1)
        ax.imshow(img.numpy().astype("uint8"))
        ax.set_title("gốc" if row == 0 else "", fontsize=9)
        ax.axis("off")

        batch = tf.expand_dims(img, 0)
        for col in range(n_variants):
            out = aug(batch, training=True)[0].numpy()
            out = np.clip(out, 0, 255).astype("uint8")
            ax = plt.subplot(n_samples, cols, row * cols + col + 2)
            ax.imshow(out)
            ax.set_title(f"augment {col+1}" if row == 0 else "", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    path = out_dir / "augmentation_demo.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[saved] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    class_names = load_class_names(OUT_DIR)
    color_dir = find_color_dir(Path(args.data_dir))
    paths, labels = list_files_labels(color_dir, class_names)
    print(f"Tổng số ảnh liệt kê được: {len(paths):,}")

    split = stratified_split(paths, labels, val_size=0.1, test_size=0.1)
    print_split_stats(split, class_names)
    save_split_csv(split, class_names, SPLIT_DIR)
    print(f"[saved] {SPLIT_DIR}/{{train,val,test}}.csv")

    class_weights = compute_class_weights(split["train"][1], class_names)
    with open(OUT_DIR / "class_weights.json", "w", encoding="utf-8") as f:
        json.dump(class_weights, f, indent=2)
    print(f"[saved] {OUT_DIR / 'class_weights.json'}")
    print("Ví dụ vài trọng số lớp (lớp càng ít ảnh, trọng số càng cao):")
    sorted_w = sorted(class_weights.items(), key=lambda kv: -kv[1])
    for idx, w in sorted_w[:3] + sorted_w[-3:]:
        print(f"  {class_names[int(idx)]:45s} weight={w:.2f}")

    # sanity check: build thử 1 batch từ tập train để chắc pipeline chạy được
    train_ds = make_dataset(*split["train"], shuffle=True, augment=True)
    images, ys = next(iter(train_ds))
    print(f"\nSanity check batch: images.shape={images.shape}, labels.shape={ys.shape}")

    plot_augmentation_demo(split["train"][0], OUT_DIR)

    print("\nXong Bước 2. Hãy xem outputs/augmentation_demo.png và báo lại:")
    print("- Augmentation trông có hợp lý không (không làm lá biến dạng quá đà)?")
    print("- Số liệu CHIA TẬP + kiểm tra lớp ít ảnh nhất ở trên.")


if __name__ == "__main__":
    main()
