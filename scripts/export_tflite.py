"""
(Tùy chọn) Convert model Keras sang TFLite để nhúng vào app di động sau này.

Xuất 2 bản:
  - model_fp32.tflite : giữ nguyên độ chính xác (float32)
  - model_dynamic_int8.tflite : lượng tử hóa động trọng số về int8, nhẹ hơn
    ~4 lần, gần như không mất accuracy, chạy nhanh hơn trên CPU điện thoại.

Chạy:
    python scripts/export_tflite.py \
        --model-path models/mobilenetv2_pv_plantdoc.keras \
        --out-dir models
"""

import argparse
from pathlib import Path

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model_path)
    keras_mb = Path(args.model_path).stat().st_size / 1e6

    # --- FP32 ---
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    fp32 = conv.convert()
    fp32_path = out_dir / "model_fp32.tflite"
    fp32_path.write_bytes(fp32)

    # --- Dynamic range int8 ---
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    int8 = conv.convert()
    int8_path = out_dir / "model_dynamic_int8.tflite"
    int8_path.write_bytes(int8)

    print("\n================  KÍCH THƯỚC  ================")
    print(f"Keras gốc            : {keras_mb:6.2f} MB  ({args.model_path})")
    print(f"TFLite fp32          : {len(fp32) / 1e6:6.2f} MB  ({fp32_path})")
    print(f"TFLite dynamic int8  : {len(int8) / 1e6:6.2f} MB  ({int8_path})")
    print("=============================================")

    # kiểm tra nhanh bản int8 còn nạp + chạy được
    interp = tf.lite.Interpreter(model_content=int8)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"\nint8 input  {inp['shape']} {inp['dtype']}")
    print(f"int8 output {out['shape']} {out['dtype']}  (38 lớp)")
    print("\nXong. Báo mình 3 con số KÍCH THƯỚC ở trên để ghi vào README.")


if __name__ == "__main__":
    main()
