"""
Bước 5 — Demo Gradio: upload ảnh lá cây -> top-3 bệnh dự đoán + % tin cậy
+ gợi ý xử lý ngắn + cảnh báo giới hạn.

Chạy local:
    MODEL_PATH=models/mobilenetv2_pv_plantdoc.keras python app.py

Deploy Hugging Face Spaces: xem hướng dẫn ở README ("Deploy demo").

LƯU Ý: gợi ý xử lý bên dưới là kiến thức phổ thông tham khảo, KHÔNG phải
tư vấn bảo vệ thực vật. Model train trên PlantVillage (ảnh 1 lá nền trơn),
độ chính xác giảm mạnh với ảnh chụp ngoài đồng (~62% trên PlantDoc).
"""

import inspect
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
CONF_WARN = 0.45
MODEL_PATH = os.environ.get("MODEL_PATH", "models/mobilenetv2_pv_plantdoc.keras")
CLASS_NAMES = json.loads(Path("outputs/class_names.json").read_text(encoding="utf-8"))

model = tf.keras.models.load_model(MODEL_PATH)

CROP_VI = {
    "Apple": "Táo",
    "Blueberry": "Việt quất",
    "Cherry_(including_sour)": "Anh đào",
    "Corn_(maize)": "Ngô",
    "Grape": "Nho",
    "Orange": "Cam",
    "Peach": "Đào",
    "Pepper,_bell": "Ớt chuông",
    "Potato": "Khoai tây",
    "Raspberry": "Mâm xôi",
    "Soybean": "Đậu tương",
    "Squash": "Bí",
    "Strawberry": "Dâu tây",
    "Tomato": "Cà chua",
}

# key = phần sau "___" trong tên lớp
ADVICE = {
    "healthy": "Lá trông khỏe mạnh. Tiếp tục theo dõi định kỳ, tưới vào gốc thay vì tưới lên lá, giữ vườn thông thoáng.",
    "Apple_scab": "Ghẻ táo (nấm Venturia). Thu gom và tiêu hủy lá rụng, tỉa cành cho thông thoáng, phun thuốc trừ nấm gốc đồng/lưu huỳnh khi chớm bệnh; ưu tiên giống kháng.",
    "Black_rot": "Thối đen (nấm). Cắt bỏ cành/quả/lá bệnh, dọn sạch tàn dư quanh gốc, phun thuốc trừ nấm phòng ngừa vào đầu mùa ẩm.",
    "Cedar_apple_rust": "Rỉ sắt táo-tuyết tùng. Loại bỏ cây tuyết tùng (ký chủ trung gian) gần vườn nếu được, phun thuốc trừ nấm từ khi nụ hé đến sau ra hoa.",
    "Powdery_mildew": "Phấn trắng (nấm). Tăng thông gió, tránh trồng dày, giảm bón đạm; phun lưu huỳnh, dầu neem hoặc thuốc đặc trị khi mới xuất hiện lớp phấn.",
    "Cercospora_leaf_spot Gray_leaf_spot": "Đốm xám lá ngô (nấm Cercospora). Luân canh cây trồng, cày vùi tàn dư, chọn giống kháng; phun thuốc trừ nấm nếu bệnh nặng trước trổ cờ.",
    "Common_rust_": "Rỉ sắt ngô (nấm Puccinia). Thường không cần xử lý trên giống kháng; nếu nhiễm sớm và nặng, phun thuốc trừ nấm.",
    "Northern_Leaf_Blight": "Cháy bìa lá phía Bắc (nấm Exserohilum). Dùng giống kháng, luân canh, vùi tàn dư; phun thuốc trừ nấm khi vết bệnh xuất hiện trên lá gần bắp.",
    "Esca_(Black_Measles)": "Esca / sởi đen nho (phức hợp nấm thân gỗ). Không có thuốc đặc trị; cắt bỏ phần thân bệnh, tránh tạo vết thương lớn khi tỉa, bảo vệ vết cắt.",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "Đốm lá nho (nấm). Tỉa tạo tán thông thoáng, dọn lá bệnh, phun thuốc trừ nấm gốc đồng theo lịch phòng ngừa.",
    "Haunglongbing_(Citrus_greening)": "Bệnh vàng lá gân xanh (vi khuẩn lan qua rầy chổng cánh). KHÔNG chữa được — nhổ bỏ và tiêu hủy cây bệnh để tránh lây, kiểm soát rầy, dùng cây giống sạch bệnh.",
    "Bacterial_spot": "Đốm vi khuẩn. Dùng hạt giống/cây con sạch bệnh, tránh làm việc khi lá ướt, luân canh; phun gốc đồng hạn chế lây lan (không diệt triệt để).",
    "Early_blight": "Đốm vòng / mốc sớm (nấm Alternaria). Ngắt bỏ lá gốc bị bệnh, phủ gốc, tưới gốc, luân canh 2-3 năm; phun thuốc trừ nấm định kỳ khi thời tiết ẩm.",
    "Late_blight": "Mốc sương (nấm trứng Phytophthora). Lây lan rất nhanh khi mát ẩm — nhổ bỏ cây bệnh ngay, phun thuốc trừ nấm phòng ngừa, tránh tưới lên lá, tiêu hủy củ/quả bệnh.",
    "Leaf_scorch": "Cháy lá dâu tây (nấm Diplocarpon). Dọn lá già/bệnh sau thu hoạch, trồng thưa, tưới nhỏ giọt; phun thuốc trừ nấm khi ra lá non nếu bệnh nặng.",
    "Septoria_leaf_spot": "Đốm lá Septoria (nấm). Ngắt lá bệnh sớm, tránh nước bắn lên lá, phủ gốc, luân canh; phun thuốc trừ nấm khi mới chớm.",
    "Spider_mites Two-spotted_spider_mite": "Nhện đỏ hai chấm (không phải bệnh — do nhện hại). Xịt mạnh nước rửa mặt dưới lá, tăng ẩm, thả thiên địch; dùng dầu khoáng/thuốc trừ nhện nếu mật độ cao.",
    "Target_Spot": "Đốm mắt cua (nấm Corynespora). Tỉa tạo thông thoáng, tránh tưới lên lá, luân canh; phun thuốc trừ nấm luân phiên hoạt chất khi bệnh xuất hiện.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Virus xoăn vàng lá cà chua (do bọ phấn truyền). KHÔNG chữa được — nhổ bỏ cây bệnh, dùng lưới chắn côn trùng và giống kháng, diệt bọ phấn.",
    "Tomato_mosaic_virus": "Virus khảm cà chua (lây qua tiếp xúc/hạt giống). KHÔNG chữa được — nhổ bỏ cây bệnh, khử trùng tay và dụng cụ, dùng hạt giống sạch và giống kháng.",
    "Leaf_Mold": "Mốc lá cà chua (nấm Passalora), phổ biến trong nhà kính ẩm. Giảm ẩm, tăng thông gió, tưới gốc, trồng thưa; phun thuốc trừ nấm nếu cần.",
}


def _pretty(class_name: str) -> str:
    crop, cond = class_name.split("___")
    return f"{CROP_VI.get(crop, crop)} — {cond.replace('_', ' ').strip()}"


def predict(img):
    if img is None:
        return {}, "Hãy tải lên một ảnh lá cây."

    arr = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype="float32")
    probs = model.predict(arr[None, ...], verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]

    labels = {_pretty(CLASS_NAMES[i]): float(probs[i]) for i in top}

    top1 = CLASS_NAMES[int(top[0])]
    p1 = float(probs[int(top[0])])
    crop, cond = top1.split("___")

    md = f"### {CROP_VI.get(crop, crop)} — {cond.replace('_', ' ').strip()}\n\n"
    md += f"**Gợi ý xử lý:** {ADVICE.get(cond, 'Chưa có gợi ý cho lớp này.')}\n\n"
    if p1 < CONF_WARN:
        md += ("> ⚠️ **Độ tin cậy thấp** — kết quả có thể sai. Thử chụp lại: "
               "chỉ 1 lá, nền đơn giản, đủ sáng, lá lấp đầy khung hình.\n\n")
    md += ("> ℹ️ Model huấn luyện trên PlantVillage (ảnh 1 lá, nền trơn). Với ảnh "
           "chụp ngoài đồng nhiều lá / nền phức tạp, độ chính xác giảm mạnh "
           "(~62% top-1 trên bộ test đồng ruộng PlantDoc). Gợi ý xử lý chỉ mang "
           "tính tham khảo, không thay tư vấn bảo vệ thực vật.")
    return labels, md


_examples_dir = Path("examples")
_examples = sorted(str(p) for p in _examples_dir.glob("*")) if _examples_dir.is_dir() else None

# gradio 4 dùng allow_flagging="never", gradio 5 đổi thành flagging_mode="never"
_flag_kw = {}
_params = inspect.signature(gr.Interface.__init__).parameters
if "flagging_mode" in _params:
    _flag_kw["flagging_mode"] = "never"
elif "allow_flagging" in _params:
    _flag_kw["allow_flagging"] = "never"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Ảnh lá cây"),
    outputs=[
        gr.Label(num_top_classes=3, label="Dự đoán (top-3)"),
        gr.Markdown(label="Chi tiết"),
    ],
    title="Chẩn đoán bệnh cây trồng qua ảnh lá",
    description=(
        "Upload ảnh một lá cây, model dự đoán loại cây + bệnh (38 lớp, dựa trên "
        "PlantVillage + fine-tune PlantDoc). **Chỉ hỗ trợ:** táo, việt quất, anh đào, "
        "ngô, nho, cam, đào, ớt chuông, khoai tây, mâm xôi, đậu tương, bí, dâu tây, "
        "cà chua. Không có lúa và nhiều cây trồng phổ biến ở Việt Nam. "
        "Kết quả tốt nhất với ảnh 1 lá, nền đơn giản, đủ sáng."
    ),
    examples=_examples,
    **_flag_kw,
)

if __name__ == "__main__":
    demo.launch()
