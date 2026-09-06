"""
Bước 3b — Fine-tune model sang miền ảnh đồng ruộng (domain adaptation).

Lấy model đã train ở Bước 3, train tiếp trên PlantDoc train, NHƯNG trộn kèm
một phần PlantVillage train làm "mỏ neo" để model không quên hẳn miền ảnh
sạch (catastrophic forgetting). Augmentation mạnh hơn Bước 3 vì lần này
mục tiêu là chịu được điều kiện chụp thực tế.

PHẢI chạy tuần tự trong CÙNG 1 phiên Colab GPU:
    1. tải PlantVillage + PlantDoc, giải nén
    2. python scripts/step2_prepare_data.py      # tạo outputs/splits/*.csv (BẮT BUỘC)
    3. python scripts/step3b_finetune_plantdoc.py ...

Chạy:
    python scripts/step3b_finetune_plantdoc.py \
        --base-model "/content/drive/MyDrive/plant-disease-detection/models/mobilenetv2_plantvillage.keras" \
        --save-path  "/content/drive/MyDrive/plant-disease-detection/models/mobilenetv2_pv_plantdoc.keras" \
        --plantdoc-dir plantdoc [--pv-sample 6000] [--epochs 20] [--unfreeze-last 60]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_pipeline import AUTOTUNE, BATCH_SIZE, SEED, _load_image, load_split_csv
from plantdoc_map import map_folder
from utils import IMG_EXTS, load_class_names

OUT_DIR = Path("outputs")
SPLIT_DIR = OUT_DIR / "splits"


def list_plantdoc(root: Path, subset: str, pv_names):
    name_to_idx = {n: i for i, n in enumerate(pv_names)}
    base = root / subset
    if not base.is_dir():
        cands = [p for p in root.rglob(subset) if p.is_dir()]
        base = cands[0] if cands else root
    paths, labels = [], []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        pv = map_folder(d.name)
        if pv is None:
            continue
        for f in sorted(d.iterdir()):
            if f.suffix in IMG_EXTS:
                paths.append(str(f))
                labels.append(name_to_idx[pv])
    return paths, labels


def strong_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.25),
            tf.keras.layers.RandomZoom(0.25, 0.25),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomContrast(0.3),
            tf.keras.layers.RandomBrightness(0.3),
        ],
        name="strong_augmentation",
    )


def make_ds(paths, labels, augment, shuffle, batch=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE).batch(batch)
    if augment:
        aug = strong_augmentation()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--plantdoc-dir", default="plantdoc")
    parser.add_argument("--pv-sample", type=int, default=6000,
                        help="số ảnh PlantVillage train làm mỏ neo (0 = không trộn)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--unfreeze-last", type=int, default=60)
    args = parser.parse_args()

    print("GPU:", tf.config.list_physical_devices("GPU") or "KHÔNG có GPU!")
    pv_names = load_class_names(OUT_DIR)

    # --- PlantDoc train -> 85/15 thành pd_train / pd_val ---
    pd_paths, pd_labels = list_plantdoc(Path(args.plantdoc_dir), "train", pv_names)
    print(f"PlantDoc train: {len(pd_paths)} ảnh, {len(set(pd_labels))} lớp (đã map)")
    pd_tr_p, pd_va_p, pd_tr_y, pd_va_y = train_test_split(
        pd_paths, pd_labels, test_size=0.15, stratify=pd_labels, random_state=SEED)

    # --- PlantVillage anchor: lấy từ split train chính thức của step2 ---
    anchor_p, anchor_y = [], []
    if args.pv_sample > 0:
        pv_tr_p, pv_tr_y = load_split_csv(SPLIT_DIR, pv_names)["train"]
        if args.pv_sample < len(pv_tr_p):
            pv_tr_p, _, pv_tr_y, _ = train_test_split(
                pv_tr_p, pv_tr_y, train_size=args.pv_sample,
                stratify=pv_tr_y, random_state=SEED)
        anchor_p, anchor_y = list(pv_tr_p), list(pv_tr_y)
        print(f"PlantVillage anchor: {len(anchor_p)} ảnh")

    # --- oversample pd_train cho cân với anchor rồi ghép ---
    reps = max(1, round(len(anchor_p) / max(len(pd_tr_p), 1))) if anchor_p else 1
    train_p = list(pd_tr_p) * reps + anchor_p
    train_y = list(pd_tr_y) * reps + anchor_y
    print(f"Tập train fine-tune: {len(train_p)} ảnh (PlantDoc x{reps} + {len(anchor_p)} PV)")

    train_ds = make_ds(train_p, train_y, augment=True, shuffle=True)
    pdval_ds = make_ds(pd_va_p, pd_va_y, augment=False, shuffle=False)

    # --- load model, mở khóa top layers, recompile LR thấp ---
    model = tf.keras.models.load_model(args.base_model)
    try:
        base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    except StopIteration:
        base = next(l for l in model.layers if "mobilenet" in l.name.lower())
    base.trainable = True
    ft_at = max(len(base.layers) - args.unfreeze_last, 0)
    for layer in base.layers[:ft_at]:
        layer.trainable = False
    print(f"base_model {len(base.layers)} layer, mở khóa {args.unfreeze_last} layer cuối")

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.save_path, monitor="val_loss", save_best_only=True),
    ]
    hist = model.fit(train_ds, validation_data=pdval_ds, epochs=args.epochs, callbacks=cbs)
    model.save(args.save_path)
    print(f"\n[saved] {args.save_path}")

    h = hist.history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(h["accuracy"], label="train"); ax[0].plot(h["val_accuracy"], label="pd_val")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h["loss"], label="train"); ax[1].plot(h["val_loss"], label="pd_val")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "finetune_plantdoc_curves.png", dpi=120)
    plt.close()
    print(f"[saved] {OUT_DIR / 'finetune_plantdoc_curves.png'}")

    print("\nXong Bước 3b. Chạy lại đánh giá model MỚI trên cả 2 test set để so sánh:")
    print(f"  python scripts/step4_evaluate.py --model-path {args.save_path}")
    print(f"  python scripts/step4b_eval_plantdoc.py --model-path {args.save_path} --data-dir {args.plantdoc_dir}")


if __name__ == "__main__":
    main()
