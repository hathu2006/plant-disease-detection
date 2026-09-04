# Plant Disease Detection — phân loại bệnh cây trồng qua ảnh lá

> Dự án cá nhân đang thực hiện. Mục tiêu: fine-tune một CNN nhẹ (MobileNetV2)
> trên dataset **PlantVillage** để phân loại ảnh lá cây thành 38 lớp
> (khỏe mạnh / các bệnh cụ thể), kèm demo web cho phép upload ảnh và nhận chẩn đoán.

## Vấn đề

Nông dân thường phát hiện bệnh cây muộn, khi triệu chứng đã lan rộng. Một công cụ
chẩn đoán nhanh từ ảnh lá (chụp bằng điện thoại) có thể giúp cảnh báo sớm.

## Cách tiếp cận

| Hạng mục | Lựa chọn | Lý do |
|---|---|---|
| Dataset | PlantVillage (qua `tensorflow_datasets`) | công khai, ~54k ảnh, 38 lớp, không cần Kaggle token |
| Model | Transfer learning từ **MobileNetV2** (pretrained ImageNet) | nhẹ, train nhanh trên Colab free, dễ convert sang TFLite |
| Framework | TensorFlow / Keras | có sẵn `keras.applications` + đường ra TFLite gọn |
| Demo | Gradio, deploy Hugging Face Spaces | có link công khai, viết ít code, hợp bài toán "1 ảnh → 1 dự đoán" |

## Các bước

- [x] **Bước 0** — chốt kiến trúc tổng thể
- [ ] **Bước 1** — tải & khám phá dữ liệu (`scripts/step1_explore.py`)
- [ ] **Bước 2** — tiền xử lý + data augmentation + chia train/val/test
- [ ] **Bước 3** — transfer learning MobileNetV2, theo dõi overfitting
- [ ] **Bước 4** — đánh giá: accuracy, precision/recall/F1 theo lớp, confusion matrix, test ảnh ngoài dataset
- [ ] **Bước 5** — demo Gradio + (tùy chọn) convert TFLite
- [ ] **Bước 6** — README case study đầy đủ + phần hạn chế

## Chạy Bước 1

Trên Google Colab (không cần GPU cho bước này):

```bash
pip install -q -r requirements.txt
python scripts/step1_explore.py
```

Sinh ra: `outputs/class_names.json`, `outputs/class_distribution.png`, `outputs/samples.png`.

## Hạn chế đã biết (sẽ nói kỹ ở Bước 6)

- PlantVillage chụp trong điều kiện kiểm soát (1 lá, nền đồng nhất, ánh sáng tốt) —
  model có thể kém chính xác với ảnh chụp thực tế ngoài đồng.
- Dataset thiên về cây trồng ở Mỹ/châu Âu (táo, nho, cà chua, khoai tây, ngô...),
  không bao gồm nhiều cây phổ biến ở Việt Nam như lúa.
- Đây là bài tập cá nhân, **không phải công cụ chẩn đoán để dùng thực tế**.
