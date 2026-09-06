"""
Bước 4b — Đánh giá model trên PlantDoc (ảnh lá chụp thực tế ngoài đồng).

Đây là con số quan trọng nhất về khả năng tổng quát hóa: model được đo
trên tập test PlantVillage (ảnh sạch) đạt ~97.4%, còn ở đây là ảnh đồng
ruộng thật, nhãn độc lập.

Cách lấy PlantDoc trên Colab:
    !kaggle datasets download -d nirmalsankalana/plantdoc-dataset -p plantdoc
    !unzip -q plantdoc/plantdoc-dataset.zip -d plantdoc
    !find plantdoc -maxdepth 3 -type d          # xem cấu trúc thật

Chạy:
    python scripts/step4b_eval_plantdoc.py \
        --model-path "/content/drive/MyDrive/plant-disease-detection/models/mobilenetv2_plantvillage.keras" \
        --data-dir plantdoc [--split auto|all|test]
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from data_pipeline import IMG_SIZE
from plantdoc_map import map_folder
from utils import IMG_EXTS, load_class_names

OUT_DIR = Path("outputs")


def find_class_dirs(root: Path, split: str):
    """Tìm mọi thư mục chứa ảnh trực tiếp. Nếu split != 'all' và có thư mục
    train/test thì chỉ lấy nhánh test."""
    img_dirs = []
    for p in root.rglob("*"):
        if p.is_dir() and any(f.suffix in IMG_EXTS for f in p.iterdir() if f.is_file()):
            img_dirs.append(p)

    if split != "all":
        test_dirs = [d for d in img_dirs if any(
            part.lower() in ("test", "testing", "val", "valid") for part in d.parts)]
        if test_dirs:
            print(f"[split] tìm thấy nhánh test → dùng {len(test_dirs)} thư mục test")
            return test_dirs
        print("[split] không thấy nhánh train/test riêng → đánh giá trên TOÀN BỘ ảnh")
    return img_dirs


def collect(root: Path, split: str, pv_names: list[str]):
    name_to_idx = {n: i for i, n in enumerate(pv_names)}
    paths, labels = [], []
    unmapped = Counter()
    per_pd_class = Counter()

    for d in find_class_dirs(root, split):
        pv = map_folder(d.name)
        if pv is None:
            n = sum(1 for f in d.iterdir() if f.suffix in IMG_EXTS)
            unmapped[d.name] += n
            continue
        for f in d.iterdir():
            if f.suffix in IMG_EXTS:
                paths.append(str(f))
                labels.append(name_to_idx[pv])
                per_pd_class[d.name] += 1

    return paths, labels, unmapped, per_pd_class


def batched_predict(model, paths, batch=64):
    probs_all = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        arr = np.stack([
            np.array(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype="float32")
            for p in chunk
        ])
        probs_all.append(model.predict(arr, verbose=0))
        print(f"\r  đã dự đoán {min(i + batch, len(paths))}/{len(paths)}", end="")
    print()
    return np.concatenate(probs_all, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default="plantdoc")
    parser.add_argument("--split", choices=["auto", "all", "test"], default="auto")
    args = parser.parse_args()

    pv_names = load_class_names(OUT_DIR)
    paths, labels, unmapped, per_pd_class = collect(
        Path(args.data_dir), "all" if args.split == "all" else "test", pv_names)

    print("\n================  PLANTDOC  ================")
    print(f"Ảnh map được về nhãn PlantVillage : {len(paths):,}")
    print(f"Số lớp PlantDoc dùng được         : {len(per_pd_class)}")
    if unmapped:
        print("\nCác lớp PlantDoc KHÔNG map được (bỏ qua) — báo lại nếu thấy sai:")
        for name, n in unmapped.most_common():
            print(f"  {name:45s} {n} ảnh")
    print("===========================================\n")

    if not paths:
        print("Không có ảnh nào map được. Chạy `!find plantdoc -maxdepth 3 -type d` "
              "và gửi mình danh sách thư mục để sửa bảng map.")
        return

    print(f"Load model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    probs = batched_predict(model, paths)
    y_true = np.array(labels)
    y_pred = probs.argmax(axis=1)
    top3 = np.argsort(probs, axis=1)[:, -3:]

    top1_acc = float((y_pred == y_true).mean())
    top3_acc = float(np.mean([t in row for t, row in zip(y_true, top3)]))

    print("\n================  KẾT QUẢ  ================")
    print(f"Top-1 accuracy (PlantDoc)  : {top1_acc:.3f}")
    print(f"Top-3 accuracy (PlantDoc)  : {top3_acc:.3f}")
    print(f"(để so sánh: test set PlantVillage ~0.974)")
    print("==========================================\n")

    present = sorted(set(labels))
    present_names = [pv_names[i] for i in present]
    print(classification_report(
        y_true, y_pred, labels=present, target_names=present_names,
        digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=present)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(max(8, len(present) * 0.45), max(7, len(present) * 0.4)))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(fraction=0.046)
    plt.xticks(range(len(present)), present_names, rotation=90, fontsize=6)
    plt.yticks(range(len(present)), present_names, fontsize=6)
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.title(f"PlantDoc confusion matrix — top-1 acc {top1_acc:.3f}")
    plt.tight_layout()
    p = OUT_DIR / "plantdoc_confusion.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"[saved] {p}")

    print("\nXong Bước 4b. Gửi mình: top-1/top-3 accuracy + classification report.")


if __name__ == "__main__":
    main()
