#!/bin/bash
# ==============================================================================
# Script Khởi Chạy 1 Lệnh Duy Nhất (Linux / macOS / WSL)
# Usage:
#   ./run.sh         -> Huấn luyện mô hình chuẩn (Fold 1)
#   ./run.sh test    -> Smoke test kiểm tra GPU và mô hình
#   ./run.sh all     -> Huấn luyện toàn bộ 5 Folds
#   ./run.sh auto    -> Chế độ tự động pull GitHub & train lại khi có commit mới
# ==============================================================================

set -e

echo "=========================================================================="
echo "    SPEECH EMOTION RECOGNITION (SER) - 1-CLICK DOCKER RUNNER"
echo "    Tối ưu hóa chuyên sâu: NVIDIA GeForce RTX 3060 (12GB / 6GB)"
echo "=========================================================================="

# 1. Kiểm tra Docker
if ! command -v docker &> /dev/null; then
    echo "[!] Lỗi: Máy tính chưa cài đặt Docker. Vui lòng cài Docker Desktop hoặc Docker Engine trước."
    exit 1
fi

# 2. Kiểm tra GPU NVIDIA
if command -v nvidia-smi &> /dev/null; then
    echo "[+] Phát hiện GPU NVIDIA:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "[!] Cảnh báo: Không tìm thấy nvidia-smi. Đang chạy ở chế độ CPU (khuyến nghị có GPU)."
fi

# 3. Tự động chuẩn bị file cấu hình .env
if [ ! -f .env ]; then
    echo "[+] Đang khởi tạo file .env từ .env.example..."
    cp .env.example .env
fi

# 4. Tự động tạo cấu trúc thư mục
mkdir -p checkpoints logs results extracted_finetuned_features data/wav

# 5. Phân tích tham số đầu vào
CMD_ARG=${1:-train}

case "$CMD_ARG" in
    test|smoke)
        echo "[*] Khởi chạy Smoke Test trong Docker..."
        docker compose run --rm ser-trainer test
        ;;
    all|5fold)
        echo "[*] Khởi chạy Huấn luyện toàn bộ 5 Folds trong Docker..."
        docker compose run --rm ser-trainer train-all
        ;;
    auto|watch)
        echo "[*] Khởi chạy Chế độ Auto-Sync GitHub & Retrain..."
        docker compose --profile auto up --build
        ;;
    *)
        echo "[*] Khởi chạy Huấn luyện & Máy chủ Giám sát TensorBoard..."
        docker compose up --build
        ;;
esac
