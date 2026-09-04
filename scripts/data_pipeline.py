"""
Pipeline dữ liệu dùng chung cho Bước 2 (kiểm tra) và Bước 3 (train).

Không tự ý copy ~2GB ảnh ra 3 thư mục train/val/test — chỉ giữ danh sách
đường dẫn + nhãn cho mỗi tập (lưu ở outputs/splits/*.csv), rồi build
tf.data.Dataset đọc thẳng từ đường dẫn gốc. Lý do: nhanh, đỡ tốn ổ đĩa, và
dễ tái lập lại đúng 1 split cho mọi lần chạy (cùng seed).

Quan trọng: bước resize/augment nằm ở đây, còn bước preprocess_input riêng
của MobileNetV2 (scale pixel về [-1, 1]) sẽ được gắn làm 1 layer NGAY TRONG
model ở Bước 3 — để khi export model đi demo/TFLite, ai dùng lại cũng không
thể quên bước tiền xử lý này.
"""

from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def stratified_split(paths, labels, val_size=0.1, test_size=0.1, seed=SEED):
    """Chia 80/10/10 mặc định, giữ nguyên tỉ lệ lớp ở mỗi tập (stratify)."""
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=(val_size + test_size),
        stratify=labels, random_state=seed,
    )
    rel_test = test_size / (val_size + test_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=rel_test,
        stratify=temp_labels, random_state=seed,
    )
    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def build_augmentation() -> tf.keras.Sequential:
    """Augmentation CHỈ áp dụng cho tập train (mô phỏng ảnh chụp thực tế:
    lệch góc, ánh sáng khác nhau, đôi khi bị lật)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomBrightness(0.15),
        ],
        name="augmentation",
    )


def _load_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img, label


def make_dataset(paths, labels, batch_size=BATCH_SIZE, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if augment:
        aug = build_augmentation()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def save_split_csv(split: dict, class_names: list[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (paths, labels) in split.items():
        with open(out_dir / f"{name}.csv", "w", encoding="utf-8") as f:
            f.write("path,label\n")
            for p, y in zip(paths, labels):
                f.write(f"{p},{class_names[y]}\n")


def load_split_csv(out_dir: Path, class_names: list[str]):
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    split = {}
    for name in ("train", "val", "test"):
        paths, labels = [], []
        with open(out_dir / f"{name}.csv", "r", encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                path, cls = line.rstrip("\n").rsplit(",", 1)
                paths.append(path)
                labels.append(name_to_idx[cls])
        split[name] = (paths, labels)
    return split
