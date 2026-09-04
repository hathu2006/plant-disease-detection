"""Hàm dùng chung cho các script data (step1, step2, ...)."""

import json
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


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


def list_files_labels(color_dir: Path, class_names: list[str]):
    """Trả về (paths, labels) — labels là chỉ số lớp theo thứ tự class_names."""
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    paths, labels = [], []
    for cls in class_names:
        cls_dir = color_dir / cls
        for f in cls_dir.iterdir():
            if f.suffix in IMG_EXTS:
                paths.append(str(f))
                labels.append(name_to_idx[cls])
    return paths, labels


def load_class_names(out_dir: Path = Path("outputs")) -> list[str]:
    with open(out_dir / "class_names.json", "r", encoding="utf-8") as f:
        return json.load(f)
