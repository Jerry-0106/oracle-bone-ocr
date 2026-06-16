FROM docker.m.daocloud.io/library/python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV OMP_NUM_THREADS=1

# Configure Debian mirror (Tsinghua) for China network
RUN sed -i 's|http://deb.debian.org/debian|http://mirrors.tuna.tsinghua.edu.cn/debian|g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|http://deb.debian.org/debian-security|http://mirrors.tuna.tsinghua.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources

# System dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libgomp1 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip with mirror and timeout
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip

# Install PyTorch with CUDA 12.4 (compatible with platform driver 12040 / CUDA 12.4)
# PyTorch 2.5.1 on PyPI = CUDA 12.4 build (no +cu suffix needed on standard pip)
# DO NOT upgrade to PyTorch 2.6+ (CUDA 12.6+) or 2.12+ (CUDA 13.0)
RUN pip install --no-cache-dir --default-timeout=300 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    torch==2.5.1 torchvision==0.20.1

# Install remaining dependencies
RUN pip install --no-cache-dir --default-timeout=300 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    opencv-python-headless \
    numpy \
    Pillow \
    ultralytics \
    tqdm

# Copy application code and models
COPY . /app
WORKDIR /app

RUN chmod +x run.sh

# ── Verify critical files exist ──
# Detector: YOLO11s @ 1280px (Phase 2, mAP50=0.84)
RUN test -f /app/checkpoints/yolo11s_det_1280.pt || \
    (echo "ERROR: Detector checkpoint missing!" && exit 1)

# Recognizer: ConvNeXt-Tiny + ArcFace (Phase 4, Top1=32.18%, 3483 classes)
RUN test -f /app/checkpoints/convnext_arcface_best.pt || \
    (echo "ERROR: Recognizer checkpoint missing!" && exit 1)

# ArcFace model definition (newly added for C3 fix)
RUN test -f /app/src/arcface_model.py || \
    (echo "ERROR: arcface_model.py missing!" && exit 1)

# Mappings (3483-class idx_to_class.json)
RUN test -f /app/mappings/idx_to_class.json || \
    (echo "ERROR: idx_to_class.json missing!" && exit 1)

# Inference script
RUN test -f /app/scripts/infer.py || \
    (echo "ERROR: infer.py missing!" && exit 1)

RUN echo "All critical files verified successfully"

ENTRYPOINT ["./run.sh"]
