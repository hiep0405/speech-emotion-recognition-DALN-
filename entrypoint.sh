#!/bin/bash
set -e

echo "=========================================================================="
echo "    SPEECH EMOTION RECOGNITION (SER) - WAV2VEC 2.0 DOCKER RUNTIME"
echo "    Hardware Target: NVIDIA GeForce RTX 3060 (Ampere Architecture)"
echo "=========================================================================="

# Đảm bảo các thư mục cần thiết luôn tồn tại
mkdir -p /workspace/checkpoints /workspace/logs /workspace/results /workspace/extracted_finetuned_features /workspace/data

MODE=${1:-train}

case "$MODE" in
    test|smoke-test)
        echo "[*] Chạy kiểm tra nhanh hệ thống (Smoke Test)..."
        exec python quick_test.py
        ;;
    train)
        echo "[*] Bắt đầu huấn luyện mô hình Fold ${FOLD:-1} (${EPOCHS:-15} epochs)..."
        exec python train_cv.py --fold "${FOLD:-1}" --epochs "${EPOCHS:-15}"
        ;;
    train-all)
        echo "[*] Bắt đầu huấn luyện toàn bộ 5 Folds Cross Validation (${EPOCHS:-15} epochs)..."
        exec python train_cv.py --fold all --epochs "${EPOCHS:-15}"
        ;;
    watch|auto-sync)
        echo "[*] Khởi động chế độ Giám sát GitHub và Tự động Huấn luyện lại (Git Watcher)..."
        exec python git_watcher.py
        ;;
    export)
        shift
        echo "[*] Trích xuất đặc trưng mới và cập nhật đồ thị..."
        exec python export_and_extract_features.py "$@"
        ;;
    bash|sh)
        exec /bin/bash
        ;;
    *)
        echo "[*] Thực thi lệnh tùy biến: $@"
        exec "$@"
        ;;
esac
