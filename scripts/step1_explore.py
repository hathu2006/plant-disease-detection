"""
Bước 1 — Khám phá dataset PlantVillage (tải qua Kaggle, không dùng TFDS).

Giả định: dataset đã được tải & giải nén vào thư mục "data/" (xem hướng dẫn
tải ở README hoặc phần chat). Script này tự tìm thư mục con tên "color"
(mỗi lớp cây/bệnh là 1 thư mục con chứa ảnh .jpg) rồi:
  - Thống kê số ảnh theo từng lớp.
  - Kiểm tra mất cân bằng lớp (class imbalance).
  - Lưu biểu đồ phân bố + lưới ảnh mẫu để hình dung dữ liệu.
  - Ghi outputs/class_names.json (dùng lại ở các bước sau).

Chạy:
    python scripts/step1_explore.py [--data-dir data]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # lưu file, không cần màn hình
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
OUT_DIR = "outputs"


def find_color_dir(root: Path) -> Path:
    """Tìm thư mục con tên 'color' chứa nhiều thư mục lớp nhất (đề phòng
    Kaggle giải nén ra cấu trúc lồng nhau kiểu data/plantvillage dataset/color/...).
    """
    candidates = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == "color"]
    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục 'color' trong {root}. "
            "Kiểm tra lại đã giải nén dataset Kaggle vào đúng chỗ chưa "
            "(chạy: !find data -maxdepth 4 -type d để xem cấu trúc thật)."
        )
    candidates.sort(key=lambda p: sum(1 for c in p.iterdir() if c.is_dir()), reverse=True)
    return candidates[0]


def scan_classes(color_dir: Path):
    class_dirs = sorted([p for p in color_dir.iterdir() if p.is_dir()])
    counts = {}
    for cd in class_dirs:
        n = sum(1 for f in cd.iterdir() if f.suffix in IMG_EXTS)
        counts[cd.name] = n
    return counts


def summarize(counts: dict):
    total = sum(counts.values())
    by_count = sorted(counts.items(), key=lambda r: r[1])
    least_name, least_n = by_count[0]
    most_name, most_n = by_count[-1]

    print("\n================  TỔNG QUAN  ================")
    print(f"Tổng số ảnh          : {total:,}")
    print(f"Số lớp               : {len(counts)}")
    print(f"Trung bình ảnh / lớp : {total / len(counts):.0f}")
    print(f"Lớp ÍT ảnh nhất      : {least_name}  = {least_n}")
    print(f"Lớp NHIỀU ảnh nhất   : {most_name}  = {most_n}")
    print(f"Tỉ lệ mất cân bằng   : {most_n / max(least_n, 1):.1f}x  (max / min)")
    print("============================================\n")

    print(f"{'lop':45s} {'so_anh':>8s}  {'ty_le_%':>8s}")
    print("-" * 66)
    for name, c in sorted(counts.items(), key=lambda r: r[0]):
        print(f"{name:45s} {c:8,d}  {100 * c / total:8.2f}")

    return by_count


def plot_distribution(by_count, out_dir: Path):
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
    path = out_dir / "class_distribution.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[saved] {path}")


def plot_samples(color_dir: Path, class_names: list, out_dir: Path):
    n = len(class_names)
    cols = 6
    rows_n = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 2.2, rows_n * 2.5))
    for i, cls in enumerate(class_names):
        cls_dir = color_dir / cls
        first_img = next((f for f in sorted(cls_dir.iterdir()) if f.suffix in IMG_EXTS), None)
        ax = plt.subplot(rows_n, cols, i + 1)
        if first_img is not None:
            with Image.open(first_img) as im:
                ax.imshow(im)
        ax.set_title(cls.replace("___", "\n"), fontsize=6)
        ax.axis("off")
    plt.tight_layout()
    path = out_dir / "samples.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[saved] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(exist_ok=True)

    root = Path(args.data_dir)
    color_dir = find_color_dir(root)
    print(f"Dùng thư mục ảnh: {color_dir}")

    counts = scan_classes(color_dir)
    class_names = sorted(counts.keys())

    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"[saved] {out_dir / 'class_names.json'}  ({len(class_names)} lớp)")

    by_count = summarize(counts)
    plot_distribution(by_count, out_dir)
    plot_samples(color_dir, class_names, out_dir)

    print("\nXong Bước 1. Hãy báo lại: tổng số ảnh, số lớp, tỉ lệ mất cân bằng,")
    print("và ảnh mẫu trong outputs/samples.png trông có hợp lý không.")


if __name__ == "__main__":
    main()
