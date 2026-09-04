"""
Bước 4 — Đánh giá model trên tập test.

Việc làm:
  1. Accuracy tổng + precision/recall/F1 theo TỪNG lớp (không chỉ accuracy
     tổng, vì dataset mất cân bằng 36x — 1 lớp hiếm bị đoán sai gần hết vẫn
     có thể không làm accuracy tổng giảm nhiều).
  2. Confusion matrix dạng heatmap: xem model hay nhầm bệnh nào với bệnh nào.
  3. In ra top 10 cặp nhầm lẫn nhiều nhất để đọc nhanh không cần nhìn ảnh.

Chạy:
    python scripts/step4_evaluate.py --model-path models/mobilenetv2_plantvillage.keras
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_pipeline import load_split_csv, make_dataset
from utils import load_class_names

OUT_DIR = Path("outputs")
SPLIT_DIR = OUT_DIR / "splits"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    class_names = load_class_names(OUT_DIR)
    split = load_split_csv(SPLIT_DIR, class_names)
    test_paths, test_labels = split["test"]
    print(f"Số ảnh test: {len(test_paths):,}")

    print(f"Load model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    # augment=False, shuffle=False: đánh giá phải dùng ảnh gốc, và giữ thứ
    # tự để khớp y_true với y_pred.
    test_ds = make_dataset(test_paths, test_labels, shuffle=False, augment=False)

    y_true = np.array(test_labels)
    print("Đang dự đoán trên tập test...")
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    overall_acc = float((y_pred == y_true).mean())
    print(f"\nOverall test accuracy: {overall_acc:.4f}")

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, zero_division=0
    )
    print("\n" + report)
    with open(OUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Overall test accuracy: {overall_acc:.4f}\n\n")
        f.write(report)
    print(f"[saved] {OUT_DIR / 'classification_report.txt'}")

    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype="float"), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"Confusion matrix (chuẩn hóa theo hàng) — test accuracy {overall_acc:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    path = OUT_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[saved] {path}")

    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    print("\nCác cặp nhầm lẫn nhiều nhất (Thực tế -> bị đoán nhầm thành):")
    for count, real, pred in pairs[:10]:
        print(f"  {real}  ->  {pred}  : {count} ảnh")

    print("\nXong Bước 4 (phần test set). Hãy báo lại:")
    print("- Overall test accuracy ở trên.")
    print("- outputs/classification_report.txt — để ý lớp nào F1 thấp nhất.")
    print("- outputs/confusion_matrix.png")
    print("- Danh sách cặp nhầm lẫn nhiều nhất ở trên.")


if __name__ == "__main__":
    main()
