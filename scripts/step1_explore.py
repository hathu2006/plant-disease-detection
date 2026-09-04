"""
Bước 1 — Tải & khám phá dataset PlantVillage (qua TensorFlow Datasets).

Mục tiêu:
  - Tải dataset (không cần tài khoản Kaggle).
  - Thống kê số ảnh theo từng lớp cây/bệnh.
  - Kiểm tra mất cân bằng lớp (class imbalance).
  - Lưu 1 biểu đồ phân bố + 1 lưới ảnh mẫu để hình dung dữ liệu.
  - Ghi ra outputs/class_names.json (dùng lại ở các bước sau).

CÁCH CHẠY TRÊN GOOGLE COLAB:
  1) Runtime không cần GPU cho bước này.
  2) Upload thư mục dự án, hoặc chỉ cần file này.
  3) Chạy:
        !pip install -q "tensorflow-datasets>=4.9"
        !python scripts/step1_explore.py
     (Lần đầu sẽ tải ~0.8 GB, mất vài phút. TFDS cache lại nên lần sau nhanh.)

CHẠY LOCAL cũng được nếu máy đã cài tensorflow + tensorflow-datasets.
"""

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # lưu file, không cần màn hình
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

DATASET = "plant_village"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def count_labels():
    """
    Đếm nhãn nhanh: dùng SkipDecoding để KHÔNG giải mã ảnh (chỉ cần nhãn),
    nên quét hết ~54k mẫu chỉ trong khoảng 1-2 phút.
    """
    ds, info = tfds.load(
        DATASET,
        split="train",
        with_info=True,
        decoders={"image": tfds.decode.SkipDecoding()},
    )
    class_names = list(info.features["label"].names)

    label_ds = ds.map(lambda ex: ex["label"],
                      num_parallel_calls=tf.data.AUTOTUNE).batch(4096)
    labels = np.concatenate([batch for batch in tfds.as_numpy(label_ds)])
    return labels, class_names, info


def summarize(labels, class_names):
    counts = Counter(int(x) for x in labels)
    per_class = [(class_names[i], counts.get(i, 0)) for i in range(len(class_names))]
    by_count = sorted(per_class, key=lambda r: r[1])
    total = int(sum(counts.values()))
    least_name, least_n = by_count[0]
    most_name, most_n = by_count[-1]

    print("\n================  TỔNG QUAN  ================")
    print(f"Tổng số ảnh          : {total:,}")
    print(f"Số lớp               : {len(class_names)}")
    print(f"Trung bình ảnh / lớp : {total / len(class_names):.0f}")
    print(f"Lớp ÍT ảnh nhất      : {least_name}  = {least_n}")
    print(f"Lớp NHIỀU ảnh nhất   : {most_name}  = {most_n}")
    print(f"Tỉ lệ mất cân bằng   : {most_n / max(least_n, 1):.1f}x  (max / min)")
    print("============================================\n")

    print(f"{'lop':45s} {'so_anh':>8s}  {'ty_le_%':>8s}")
    print("-" * 66)
    for name, c in sorted(per_class, key=lambda r: r[0]):
        print(f"{name:45s} {c:8,d}  {100 * c / total:8.2f}")

    return per_class, by_count


def plot_distribution(by_count):
    names = [r[0] for r in by_count]
    vals = [r[1] for r in by_count]
    plt.figure(figsize=(10, 12))
    bars = plt.barh(range(len(names)), vals)
    plt.yticks(range(len(names)), names, fontsize=7)
    plt.xlabel("Số ảnh")
    plt.title("PlantVillage — phân bố ảnh theo lớp (tăng dần)")
    for b, v in zip(bars, vals):
        plt.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                 str(v), va="center", fontsize=6)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "class_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n[saved] {path}")


def plot_samples(class_names):
    """1 ảnh đại diện cho mỗi lớp."""
    ds = tfds.load(DATASET, split="train", as_supervised=True, shuffle_files=True)
    seen = {}
    for img, lbl in tfds.as_numpy(ds):
        lbl = int(lbl)
        seen.setdefault(lbl, img)
        if len(seen) == len(class_names):
            break

    n = len(class_names)
    cols = 6
    rows_n = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 2.2, rows_n * 2.5))
    for i in range(n):
        ax = plt.subplot(rows_n, cols, i + 1)
        ax.imshow(seen[i])
        ax.set_title(class_names[i].replace("___", "\n"), fontsize=6)
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "samples.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[saved] {path}")


def main():
    labels, class_names, info = count_labels()

    with open(os.path.join(OUT_DIR, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"[saved] {os.path.join(OUT_DIR, 'class_names.json')}  ({len(class_names)} lớp)")

    _, by_count = summarize(labels, class_names)
    plot_distribution(by_count)
    plot_samples(class_names)

    print("\nXong Bước 1. Hãy báo lại: tổng số ảnh, số lớp, tỉ lệ mất cân bằng,")
    print("và ảnh mẫu trong outputs/samples.png trông có hợp lý không.")


if __name__ == "__main__":
    main()
