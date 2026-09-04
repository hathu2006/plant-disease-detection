"""
Bước 3 — Transfer learning MobileNetV2 trên PlantVillage.

YÊU CẦU: bật GPU trước khi chạy — Runtime > Change runtime type > GPU (T4).
Kiểm tra: !nvidia-smi

2 pha huấn luyện:
  Pha 1 (feature extraction) — đóng băng toàn bộ MobileNetV2 (giữ nguyên
    trọng số ImageNet), chỉ train phần đầu phân loại mới gắn vào. Học
    nhanh vì phần lớn tham số không đổi.
  Pha 2 (fine-tuning) — mở khóa ~50 lớp cuối của MobileNetV2, train tiếp
    với learning rate rất nhỏ (1e-5) để các lớp đó "tinh chỉnh" theo đặc
    trưng riêng của ảnh lá cây, thay vì chỉ đặc trưng ImageNet chung chung.

EarlyStopping theo dõi val_loss ở cả 2 pha để dừng sớm nếu bắt đầu overfit
(train tốt lên nhưng val xấu đi).

LƯU Ý QUAN TRỌNG: Colab xóa hết file khi hết phiên/timeout. Muốn giữ model
sau khi train xong, mount Google Drive TRƯỚC khi chạy:
    from google.colab import drive
    drive.mount('/content/drive')
rồi chạy với:
    --save-dir "/content/drive/MyDrive/plant-disease-detection/models"

Chạy:
    python scripts/step3_train.py [--save-dir models] \
        [--epochs-head 8] [--epochs-finetune 8] [--unfreeze-last 50]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from data_pipeline import IMG_SIZE, load_split_csv, make_dataset
from utils import load_class_names

OUT_DIR = Path("outputs")
SPLIT_DIR = OUT_DIR / "splits"


def build_model(n_classes: int, base_trainable: bool = False):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet", pooling="avg"
    )
    base_model.trainable = base_trainable

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    # training=False luôn cố định, kể cả sau khi mở khóa fine-tune ở Pha 2:
    # giữ BatchNorm ở chế độ inference (dùng thống kê đã học từ ImageNet)
    # để không bị phá vỡ bởi vài epoch fine-tune với batch nhỏ.
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base_model


def plot_history(histories: list, out_dir: Path):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]
    switch_epoch = len(histories[0].history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="val")
    axes[0].axvline(switch_epoch - 0.5, color="gray", linestyle="--", label="bắt đầu fine-tune")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="val")
    axes[1].axvline(switch_epoch - 0.5, color="gray", linestyle="--", label="bắt đầu fine-tune")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plt.tight_layout()
    path = out_dir / "training_curves.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[saved] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--epochs-head", type=int, default=8)
    parser.add_argument("--epochs-finetune", type=int, default=8)
    parser.add_argument("--unfreeze-last", type=int, default=50,
                        help="số layer cuối của MobileNetV2 được mở khóa ở Pha 2")
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    print("GPU:", gpus or "KHÔNG có GPU — vào Runtime > Change runtime type > GPU trước khi train!")

    class_names = load_class_names(OUT_DIR)
    split = load_split_csv(SPLIT_DIR, class_names)

    with open(OUT_DIR / "class_weights.json") as f:
        class_weight = {int(k): v for k, v in json.load(f).items()}

    train_ds = make_dataset(*split["train"], shuffle=True, augment=True)
    val_ds = make_dataset(*split["val"], shuffle=False, augment=False)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "mobilenetv2_plantvillage.keras"

    # ---- Pha 1: train head, backbone đóng băng ----
    model, base_model = build_model(len(class_names), base_trainable=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks_head = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
    ]
    print("\n=== PHA 1: train head (backbone đóng băng) ===")
    hist1 = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs_head,
        class_weight=class_weight, callbacks=callbacks_head,
    )

    # ---- Pha 2: fine-tune, mở khóa top layers ----
    base_model.trainable = True
    fine_tune_at = max(len(base_model.layers) - args.unfreeze_last, 0)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    print(f"\nbase_model có {len(base_model.layers)} layer, "
          f"mở khóa {args.unfreeze_last} layer cuối (từ layer {fine_tune_at}).")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_ft = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
    ]
    print("\n=== PHA 2: fine-tune (mở khóa top layers) ===")
    hist2 = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
        class_weight=class_weight, callbacks=callbacks_ft,
    )

    plot_history([hist1, hist2], OUT_DIR)
    model.save(ckpt_path)
    print(f"\n[saved] model cuối cùng: {ckpt_path}")

    print("\nXong Bước 3. Hãy báo lại:")
    print("- outputs/training_curves.png (loss/accuracy 2 pha)")
    print("- val_accuracy / val_loss cuối cùng in ra ở log trên")
    print("- có dấu hiệu overfit không (train acc cao hẳn val acc, hoặc val_loss tăng lại)?")
    print("- ĐỪNG QUÊN tải/copy model ra Google Drive nếu chưa mount Drive từ đầu,")
    print("  không thì mất khi Colab ngắt phiên.")


if __name__ == "__main__":
    main()
