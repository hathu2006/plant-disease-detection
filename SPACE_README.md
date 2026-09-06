---
title: Plant Disease Detection
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# Chẩn đoán bệnh cây trồng qua ảnh lá

Demo cho model MobileNetV2 (transfer learning trên PlantVillage, fine-tune trên
PlantDoc) phân loại ảnh lá thành 38 lớp bệnh/khỏe mạnh.

Mã nguồn & case study đầy đủ: https://github.com/hathu2006/plant-disease-detection

> Model huấn luyện chủ yếu trên ảnh lá "sạch" (1 lá, nền trơn). Độ chính xác
> giảm mạnh với ảnh chụp ngoài đồng (~62% top-1 trên bộ test PlantDoc). Gợi ý xử
> lý chỉ mang tính tham khảo, không thay tư vấn bảo vệ thực vật.

---

## Cách deploy (làm 1 lần)

1. Tạo Space mới tại https://huggingface.co/new-space → SDK **Gradio**, phần cứng
   **CPU basic (free)**.
2. Clone Space về máy:
   ```bash
   git clone https://huggingface.co/spaces/<user>/plant-disease-detection space
   cd space
   ```
3. Copy các file sau từ repo GitHub vào thư mục `space/`:
   - `app.py`
   - `requirements.txt`
   - `outputs/class_names.json`  (giữ nguyên đường dẫn `outputs/class_names.json`)
   - **file này** đổi tên thành `README.md` (phần YAML ở đầu là bắt buộc cho Space)
4. Copy model đã fine-tune vào `space/models/mobilenetv2_pv_plantdoc.keras`
   (Space tự bật Git LFS cho file > 10 MB).
5. Push:
   ```bash
   git add -A && git commit -m "deploy demo" && git push
   ```
6. Space tự build (~5-10 phút). Xong thì cập nhật link vào README của repo GitHub.
