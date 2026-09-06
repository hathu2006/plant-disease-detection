"""
Ánh xạ tên lớp của PlantDoc  ->  38 lớp của PlantVillage (không gian nhãn
mà model đang dùng).

PlantDoc (~2,600 ảnh chụp đồng ruộng thật, 27-28 lớp) đặt tên lớp KHÁC
PlantVillage, ví dụ "Corn leaf blight" (PlantDoc) = "Corn_(maize)___Northern_Leaf_Blight"
(PlantVillage). File này giữ bảng quy đổi đó.

PlantDoc KHÔNG phủ hết 38 lớp — các lớp không có ảnh đồng ruộng tương ứng
(Apple Black_rot, Grape Esca, Orange Haunglongbing, Tomato Target_Spot...)
sẽ bị bỏ qua khi đánh giá, và script sẽ in ra rõ những lớp PlantDoc nào
không map được để mình chỉnh tiếp nếu cần.
"""


def _norm(name: str) -> str:
    """Chuẩn hóa tên thư mục PlantDoc để tra bảng: thường/bỏ gạch dưới/gộp khoảng trắng."""
    return " ".join(name.lower().replace("_", " ").split())


# key đã _norm  ->  tên lớp PlantVillage
_RAW_MAP = {
    "apple scab leaf": "Apple___Apple_scab",
    "apple leaf": "Apple___healthy",
    "apple rust leaf": "Apple___Cedar_apple_rust",
    "bell pepper leaf": "Pepper,_bell___healthy",
    "bell pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "blueberry leaf": "Blueberry___healthy",
    "cherry leaf": "Cherry_(including_sour)___healthy",
    "corn gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "corn rust leaf": "Corn_(maize)___Common_rust_",
    "peach leaf": "Peach___healthy",
    "potato leaf early blight": "Potato___Early_blight",
    "potato leaf late blight": "Potato___Late_blight",
    "potato leaf": "Potato___healthy",
    "raspberry leaf": "Raspberry___healthy",
    "soyabean leaf": "Soybean___healthy",
    "soybean leaf": "Soybean___healthy",
    "squash powdery mildew leaf": "Squash___Powdery_mildew",
    "strawberry leaf": "Strawberry___healthy",
    "tomato early blight leaf": "Tomato___Early_blight",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato leaf": "Tomato___healthy",
    "tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "tomato leaf late blight": "Tomato___Late_blight",
    "tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomato mold leaf": "Tomato___Leaf_Mold",
    "tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}

PLANTDOC_TO_PV = {_norm(k): v for k, v in _RAW_MAP.items()}


def map_folder(name: str) -> str | None:
    """Trả về tên lớp PlantVillage tương ứng, hoặc None nếu không map được."""
    return PLANTDOC_TO_PV.get(_norm(name))
