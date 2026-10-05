# AICycle Image Classification

# NOTE: ĐÂY LÀ BẢN FORK CÓ VÀI THAY ĐỔI NHỎ

> Framework huấn luyện Image Classification production-ready, xây dựng trên PyTorch Lightning,
> thiết kế theo hướng mở rộng và tích hợp MLOps — lấy cảm hứng từ kiến trúc của [Ultralytics](https://github.com/ultralytics/ultralytics).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.6-purple)](https://lightning.ai/)
[![MLflow](https://img.shields.io/badge/MLflow-3.1-blue)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Mục Lục

- [Giới Thiệu](#giới-thiệu)
- [Cài Đặt](#cài-đặt)
- [Quick Start](#quick-start)
- [Export Model](#export-model)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Hệ Thống Cấu Hình](#hệ-thống-cấu-hình)
- [Tính Năng](#tính-năng)
- [Mở Rộng Framework](#mở-rộng-framework)

---

## Giới Thiệu

**AICycle Image Classification** là một framework huấn luyện dạng modular, configuration-driven, xây dựng trên nền PyTorch Lightning. Framework này vận hành các pipeline phân loại ảnh sản xuất tại AICycle — từ chấm điểm chất lượng xe đến nhận dạng logo và hình dạng.

Framework tuân theo nguyên tắc **Separation of Concerns** nghiêm ngặt:

- **`classify/data/`** — data loading, preprocessing và augmentation
- **`classify/models/`** — backbone registry và Lightning module
- **`classify/engine/`** — training, inference và ONNX export
- **`classify/utils/`** — logging và config dùng chung
- **`cfg/`** — toàn bộ hyperparameter trong file YAML, không hard-code trong Python

Mọi workflow đều truy cập được qua **CLI** duy nhất (`main.py`) hoặc **Python API** (`Classifier`).

---

## Cài Đặt

### Cách 1 — pip (local / conda)

```bash
# 1. Clone repository
git clone https://github.com/aicycle/AICycle-Trainer.git
cd AICycle-Trainer/image_classification

# 2. Tạo và kích hoạt virtual environment (khuyến nghị)
conda create -n aicycle-cls python=3.10 -y
conda activate aicycle-cls

# 3. Cài đặt dependencies
pip install -r requirements.txt
```

### Cách 2 — Docker

```bash
# Build Docker image
cd image_classification/
bash docker/build.sh          # tạo image  lightning:latest

# Chạy container với GPU (chỉnh đường dẫn DATASET trong docker/run.sh trước)
bash docker/run.sh
```

Script `docker/run.sh` mount thư mục hiện tại vào container và cấp quyền truy cập toàn bộ GPU qua `--gpus all`.

> **Base image:** `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`

---

## Quick Start

### Python API — inference 5 dòng

```python
from classify import Classifier

clf = Classifier("cfg/car_quality.yaml")          # load task config
results = clf.predict(
    source="path/to/images/",                     # file hoặc thư mục
    ckpt_path="output/best.ckpt",
)
for r in results:
    print(r["file"], "→ class", r["class_index"], f"({r['score']:.2%})")
```

### Python API — training

```python
from classify import Classifier

clf = Classifier("cfg/car_quality.yaml")
clf.train()                                       # toàn bộ params đọc từ YAML

# Override từng key ngay trên dòng lệnh — không cần sửa YAML
clf.train(data_dir="/new/dataset", max_epochs=200, batch_size=32)
```

### CLI — train 1 dòng lệnh

```bash
python main.py train --cfg cfg/car_quality.yaml
```

### CLI — override params trực tiếp

```bash
# Truyền key=value sau tên sub-command để ghi đè YAML mà không cần sửa file
python main.py train --cfg cfg/car_quality.yaml \
    data_dir=/data/cars max_epochs=200 batch_size=32
```

### CLI — predict và export

```bash
# Dự đoán trên toàn bộ thư mục ảnh
python main.py predict \
    --cfg cfg/car_quality.yaml \
    --source /data/test_images \
    --ckpt_path output/best.ckpt

# Export sang ONNX
python main.py export \
    --cfg cfg/car_quality.yaml \
    --ckpt_path output/best.ckpt \
    --file_path model.onnx \
    --opset 12
```

---

## Export Model

Framework hỗ trợ 2 định dạng export, truy cập qua **Python API** hoặc **CLI**.

| Format | File | Phù hợp cho |
|---|---|---|
| **ONNX** | `.onnx` | ONNX Runtime, TensorRT, OpenCV DNN, Core ML |
| **TorchScript** | `.pt` | C++ server, mobile, môi trường không có Python |

### Python API — Export ONNX

```python
from classify import Classifier

clf = Classifier("cfg/car_quality.yaml")

# Export ONNX (mặc định)
clf.export(
    ckpt_path="output/best.ckpt",
    file_path="deploy/model.onnx",
    format="onnx",
    opset=12,           # ONNX opset version
)
```

### Python API — Export TorchScript

```python
from classify import Classifier

clf = Classifier("cfg/car_quality.yaml")

# Export TorchScript — không cần Python khi deploy
clf.export(
    ckpt_path="output/best.ckpt",
    file_path="deploy/model.pt",
    format="torchscript",
    optimize_torchscript=True,  # áp dụng torch.jit.optimize_for_inference
)
```

### Python API — Override resize khi export

Nếu cần export với kích thước input khác config:

```python
clf.export(
    ckpt_path="output/best.ckpt",
    file_path="deploy/model_512.onnx",
    format="onnx",
    resize=512,   # override cfg["resize"] chỉ cho lần export này
)
```

### Python API — Export trực tiếp từ hàm riêng

```python
from classify.engine.exporter import export_onnx, export_torchscript

# ONNX
export_onnx(
    ckpt_path="output/best.ckpt",
    file_path="deploy/model.onnx",
    resize=512,
    input_name="input",
    output_name="output",
    opset=12,
)

# TorchScript
export_torchscript(
    ckpt_path="output/best.ckpt",
    file_path="deploy/model.pt",
    resize=512,
    optimize=True,
)
```

### CLI — Export

```bash
# Export ONNX
python main.py export \
    --cfg cfg/car_quality.yaml \
    --ckpt_path output/best.ckpt \
    --file_path deploy/model.onnx \
    --opset 12

# Export TorchScript
python main.py export \
    --cfg cfg/car_quality.yaml \
    --ckpt_path output/best.ckpt \
    --file_path deploy/model.pt \
    format=torchscript
```

### Load và chạy TorchScript model (không cần class gốc)

```python
import torch

model = torch.jit.load("deploy/model.pt")
model.eval()

dummy = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    logits = model(dummy)   # shape: (1, num_classes)
```

---

## Cấu Trúc Dự Án

```
image_classification/
│
├── main.py                        # 🚀 Unified CLI & entry-point (train / predict / export)
├── requirements.txt               # Dependencies được ghim phiên bản
│
├── cfg/                           # ⚙️  Toàn bộ hyperparameter — không hard-code trong Python
│   ├── default.yaml               #    Giá trị mặc định (base layer, luôn được load)
│   └── car_quality.yaml           #    Override cho từng task (kế thừa từ default)
│
├── classify/                      # 📦 Core package
│   │
│   ├── __init__.py                #    High-level Classifier API (train / predict / export)
│   │
│   ├── data/                      # 🗃️  Data pipeline
│   │   ├── dataset.py             #    ClassificationDataset + ClassificationDataModule
│   │   ├── augmentation.py        #    Albumentations pipeline & Transformation wrapper
│   │   └── utils.py               #    letterbox(), create_dataset_dict()
│   │
│   ├── models/                    # 🧠 Backbone layer
│   │   ├── registry.py            #    @register_model factory — thêm backbone mới chỉ 5 dòng
│   │   └── lightning_module.py    #    ImageClassificationModule (LightningModule)
│   │
│   ├── engine/                    # ⚙️  Orchestration layer
│   │   ├── trainer.py             #    ClassificationTrainer (kết nối data + model + callbacks)
│   │   ├── predictor.py           #    Predictor (inference đơn ảnh & batch thư mục)
│   │   ├── exporter.py            #    export_onnx() — Lightning checkpoint → ONNX
│   │   └── callbacks.py           #    GenerateLabelFileCallback, LogArtifactsToMLflowCallback
│   │
│   └── utils/                     # 🔧 Tiện ích dùng chung
│       ├── config.py              #    get_cfg(), load_yaml(), merge_configs()
│       └── logging.py             #    get_logger() — log có timestamp, hỗ trợ ghi file
│
├── docker/                        # 🐳 Container setup
│   ├── Dockerfile
│   ├── build.sh
│   └── run.sh
│
└── scripts/                       # 🔨 Shell launcher cho từng task (wrapper mỏng qua main.py)
    ├── train_car_quality.sh
    ├── train_car_corner.sh
    ├── train_car_model.sh
    ├── train_car_shape.sh
    ├── train_car_logo.sh
    ├── predict.sh
    └── export_onnx.sh
```

---

## Hệ Thống Cấu Hình

Cấu hình được giải quyết theo **3 lớp** (lớp sau ghi đè lớp trước):

```
cfg/default.yaml          ← (1) Giá trị mặc định được ship sẵn
    ↓
cfg/<task>.yaml           ← (2) Override theo task  (cờ --cfg)
    ↓
key=value CLI overrides   ← (3) Override trực tiếp  (positional args)
```

**`cfg/default.yaml`** (trích đoạn):

```yaml
model_name: mobilenet_v3_small
num_classes: 2
optimizer: SGD
learning_rate: 0.005
max_epochs: 50
precision: "32"            # hoặc "16-mixed" / "bf16-mixed"
enable_mlflow: false
```

Để tạo config cho task mới, chỉ cần copy `default.yaml` và override các key cần thay đổi:

```bash
cp cfg/default.yaml cfg/my_task.yaml
# Chỉnh: data_dir, num_classes, model_name, ...
python main.py train --cfg cfg/my_task.yaml
```

---

## Tính Năng

### 🧠 Hỗ Trợ Mọi Backbone Torchvision — Không Cần Cấu Hình Thêm

Tất cả `torchvision.models` hoạt động ngay lập tức. Framework tự phát hiện classification head (`fc` / `classifier` / `head`) và thay thế đúng số output classes:

```yaml
model_name: efficientnet_v2_s   # hoặc resnet50, swin_t, vit_b_16, ...
num_classes: 4
```

### 🔌 Custom Model Registry (Factory Pattern)

Đăng ký backbone tùy chỉnh chỉ 5 dòng — **không cần sửa bất kỳ dòng code training nào**:

```python
from classify.models.registry import register_model
import torch.nn as nn

@register_model("my_lightweight_cnn")
def build(num_classes: int) -> nn.Module:
    ...
    return model
```

### ⚡ Mixed Precision Training

Bật bằng một dòng trong YAML — không cần chỉnh code:

```yaml
precision: "bf16-mixed"   # hoặc "16-mixed" cho GPU đời cũ hơn
```

### 📊 Tích Hợp MLflow

Bật MLflow tracking chỉ với 2 dòng trong task YAML:

```yaml
enable_mlflow: true
mlflow_tracking_uri: "https://mlflow.aicycle.ai"
```

Khi training kết thúc, framework tự động:
1. Export checkpoint tốt nhất sang ONNX
2. Upload checkpoints, ONNX model, CSV log và `labels.txt` lên MLflow artifacts

### 🏷️ Tự Động Sinh File Label

`labels.txt` được ghi ngay khi bắt đầu training, liệt kê tên các class theo thứ tự alphabet — khớp với integer index của model output — sẵn sàng đi kèm với ONNX model khi deploy.

### 📐 Letterboxing Giữ Tỷ Lệ Ảnh

Tất cả ảnh được resize về canvas vuông bằng letterboxing (không méo), với màu padding có thể cấu hình (`pad_color: 255` cho trắng, `0` cho đen).

### 🔁 Distributed Training (DDP)

Chuyển sang multi-GPU mà không cần chỉnh Python:

```bash
python main.py train --cfg cfg/car_quality.yaml strategy=ddp devices=4
```

### 🛡️ Checkpoint Loading An Toàn

`custom_load_state_dict()` load weights với khả năng chịu lỗi shape mismatch — layer nào không khớp sẽ được bỏ qua kèm warning thay vì crash, hỗ trợ fine-tuning an toàn khi thay đổi `num_classes`.

---

## Mở Rộng Framework

### Thêm loại dataset mới

1. Tạo `classify/data/my_dataset.py` implement `torch.utils.data.Dataset`.
2. Đăng ký hoặc swap vào `ClassificationDataModule.setup()`.
3. Không cần thay đổi bất kỳ code engine nào.

### Thêm training callback mới

```python
from lightning.pytorch.callbacks import Callback

class MyCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        ...
```

Thêm vào danh sách trong `ClassificationTrainer._build_callbacks()`.
