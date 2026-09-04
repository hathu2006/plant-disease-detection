"""
Dự đoán bệnh cây từ 1 hoặc nhiều ảnh bất kỳ (không cần thuộc PlantVillage).

Dùng lại ở:
  - Bước 4: test model trên ảnh ngoài dataset (đánh giá khả năng tổng quát hóa).
  - Bước 5: làm backend cho demo Gradio.

Chạy:
    python scripts/predict.py --model-path models/mobilenetv2_plantvillage.keras \
        anh1.jpg anh2.jpg
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from data_pipeline import IMG_SIZE
from utils import load_class_names

OUT_DIR = Path("outputs")


def load_and_resize(path: str) -> np.ndarray:
    """Đọc ảnh bất kỳ (jpg/png/...), convert RGB, resize về đúng kích thước
    model cần. KHÔNG tự chia thang màu ở đây — preprocess_input của
    MobileNetV2 đã nằm sẵn trong model (xem step3_train.py)."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype="float32")


def predict_topk(model, class_names: list[str], img_array: np.ndarray, k: int = 3):
    batch = np.expand_dims(img_array, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in top_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("images", nargs="+", help="đường dẫn tới 1 hoặc nhiều ảnh")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    class_names = load_class_names(OUT_DIR)
    print(f"Load model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    for path in args.images:
        img_array = load_and_resize(path)
        preds = predict_topk(model, class_names, img_array, k=args.topk)
        print(f"\n{path}")
        for name, prob in preds:
            bar = "#" * int(prob * 30)
            print(f"  {name:45s} {prob*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
