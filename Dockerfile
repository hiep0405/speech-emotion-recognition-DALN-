# ==============================================================================
# Dockerfile: SER Fine-tuning Wav2Vec 2.0 (Tối ưu hóa NVIDIA GeForce RTX 3060)
# Kiến trúc: NVIDIA Ampere (Compute Capability 8.6) | CUDA 12.1 | PyTorch 2.3+
# ==============================================================================

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Thiết lập biến môi trường hệ thống
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_CUDA_ARCH_LIST="8.6" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Cài đặt các thư viện hệ thống cần thiết (libsndfile cho đọc audio, ffmpeg, git, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc chính
WORKDIR /workspace

# Cài đặt Python packages với cơ chế cache layer
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . /workspace/

# Phân quyền thực thi cho các script khởi chạy
RUN chmod +x /workspace/entrypoint.sh /workspace/run.sh 2>/dev/null || true

# Mở cổng TensorBoard (6006) và GitHub Webhook (8000)
EXPOSE 6006 8000

# Thiết lập Entrypoint mặc định
ENTRYPOINT ["/bin/bash", "/workspace/entrypoint.sh"]
CMD ["train"]
