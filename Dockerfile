FROM docker.m.daocloud.io/library/python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV OMP_NUM_THREADS=1

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, set mirror and timeout for reliability
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip

# Install PyTorch CPU-only (competition platform provides GPU separately if needed)
# Pinned versions to avoid resolution scan + timeout
RUN pip install --no-cache-dir --default-timeout=120 \
    torch==2.5.1+cpu torchvision==0.20.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --default-timeout=120 \
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

# Verify critical files exist
RUN ls -la /app/models/detector.pt /app/models/recognizer.pt && \
    ls -la /app/ocr_pipeline/mappings/ID_to_chinese.json && \
    echo "All critical files verified"

ENTRYPOINT ["./run.sh"]
