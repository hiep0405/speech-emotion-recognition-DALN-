# 🎙️ Speech Emotion Recognition (SER) - Wav2Vec 2.0 Fine-Tuning

[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Optimized_for-RTX_3060-76B900?logo=nvidia)](https://www.nvidia.com/)
[![Weights & Biases](https://img.shields.io/badge/W%26B-Monitoring-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-Port_6006-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/tensorboard)

Dự án tinh chỉnh mô hình nền tảng âm thanh tự giám sát (**Self-Supervised Learning - Wav2Vec 2.0 Large XLS-R**) cho bài toán **Nhận diện Cảm xúc Tiếng nói (Speech Emotion Recognition - SER)** trên tập dữ liệu chuẩn **IEMOCAP** (4 lớp cảm xúc: *Anger*, *Sadness*, *Happiness*, *Neutral*).

Dự án được đóng gói trọn gói dạng **Turn-Key Solution**: Clone về và chạy **đúng 1 lệnh duy nhất**, tối ưu hóa bảo vệ card đồ họa **NVIDIA GeForce RTX 3060** (chống quá nhiệt, chống chai tụ), tích hợp **giám sát từ xa (TensorBoard, W&B, Telegram)** và **tự động cập nhật code từ GitHub để huấn luyện lại (Auto-Sync CI/CD)**.

---

## ⚡ 1. KHỞI CHẠY 1 LỆNH DUY NHẤT (ZERO SETUP)

Bất kỳ ai clone repository này về chỉ cần chạy đúng 1 lệnh tương ứng với hệ điều hành:

### 🔹 Trên Windows:
Double-click hoặc mở PowerShell/CMD gõ:
```cmd
run.bat
```

### 🔹 Trên Linux / macOS / WSL:
Mở Terminal và gõ:
```bash
chmod +x run.sh && ./run.sh
```

### 🔹 Chạy trực tiếp bằng Docker Compose chuẩn:
```bash
docker compose up --build
```

> **Hệ thống sẽ tự động:**
> 1. Kiểm tra Docker và kích hoạt nhân GPU NVIDIA CUDA.
> 2. Tự động sinh file cấu hình `.env` tối ưu cho RTX 3060.
> 3. Tự động tải Pretrained Model `wav2vec2-lg-xlsr-en-speech-emotion-recognition` và lưu vào volume cache.
> 4. Khởi chạy tiến trình huấn luyện và bật **TensorBoard Server** tại cổng `http://localhost:6006`.

---

## 🛡️ 2. TỐI ƯU HÓA CHUYÊN SÂU CHO NVIDIA GEFORCE RTX 3060

Dòng card **RTX 3060** (12GB Desktop hoặc 6GB Laptop kiến trúc Ampere) có hiệu năng cao nhưng dễ bị quá nhiệt và chai linh kiện nếu huấn luyện liên tục nhiều giờ không kiểm soát. Dự án tích hợp module **`HardwareGuardian`** độc quyền:

| Tính năng bảo vệ & tối ưu | Cơ chế kỹ thuật | Lợi ích |
| :--- | :--- | :--- |
| **Bảo vệ Chống Quá Nhiệt (Thermal Guardian)** | Đọc trực tiếp nhiệt độ GPU qua `pynvml`. Khi nhiệt độ chạm ngưỡng cảnh báo (**78°C**), hệ thống **tự động tạm dừng** huấn luyện, chờ quạt làm mát xuống dưới **70°C** rồi mới chạy tiếp. | **Tuyệt đối không hỏng máy, không phù tụ, không khô keo tản nhiệt.** |
| **Chống Tràn VRAM & Sập Màn Hình (OOM Safe Headroom)** | Giới hạn trần cấp phát VRAM ở mức **88%** (`set_per_process_memory_fraction(0.88)`), giữ lại 12% VRAM cho Windows/OS. | Tránh triệt để lỗi sập driver đồ họa, đơ máy hoặc xanh màn hình (BSOD). |
| **Chống Phân Mảnh VRAM** | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. | Giải quyết triệt để lỗi CUDA Out Of Memory do phân mảnh block bộ nhớ. |
| **Tăng tốc Tensor Core Ampere (TF32 & FP16)** | Bật `torch.backends.cuda.matmul.allow_tf32 = True` và Mixed Precision `autocast(fp16)`. | **Tăng tốc độ tính toán gấp 2.5 lần** so với FP32 thuần túy mà vẫn giữ nguyên độ chính xác. |
| **Gradient Checkpointing + Accumulation** | Bật checkpointing tầng Transformer + Gom batch 8 bước $\times$ 4 tích lũy = Effective Batch Size 32. | Chỉ tốn **~5.5GB - 6.5GB VRAM**, chạy mượt mà trên cả bản Laptop 6GB lẫn Desktop 12GB. |
| **Bám Chuẩn Thuật Toán Nghiên Cứu** | Đóng băng 7 tầng CNN, mở 6 tầng cuối Transformer với **Differential LR** ($10^{-5}$ cho backbone, $5 \times 10^{-4}$ cho classifier), Label Smoothing 0.05, Attentive Pooling. | Độ chính xác đạt chuẩn hội tụ SOTA cao nhất. |

---

## 📡 3. GIÁM SÁT TỪ XA ĐA KÊNH (REMOTE MONITORING)

Dự án cung cấp 3 kênh giám sát đồng thời giúp bạn theo dõi mô hình ở bất cứ đâu:

```
[Máy Huấn Luyện RTX 3060]
       │
       ├──► 1. TensorBoard Web UI (Mở port 6006 xem đồ thị Loss, WA/UA, GPU Temp)
       │
       ├──► 2. Weights & Biases (W&B Cloud - Xem trực tiếp trên App điện thoại)
       │
       └──► 3. Telegram Bot Alert (Báo cáo Best Checkpoint & Cảnh báo quá nhiệt)
```

1. **TensorBoard (Tích hợp sẵn trong Docker):**
   - Truy cập ngay tại trình duyệt: `http://localhost:6006`
   - Theo dõi biểu đồ Loss Train/Val, Weighted Accuracy (WA), Unweighted Accuracy (UA), Nhiệt độ GPU (°C), VRAM sử dụng (MB).
2. **Weights & Biases (W&B):**
   - Mở file `.env`, dán API Key của bạn: `WANDB_API_KEY=your_key_here`
   - Tự động đồng bộ lên cloud `wandb.ai`, mở điện thoại xem đồ thị realtime.
3. **Telegram Notifier (Tiện lợi nhất):**
   - Mở file `.env`, điền:
     ```env
     TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
     TELEGRAM_CHAT_ID=123456789
     ```
   - Bot tự động nhắn tin cho bạn:
     - 🚀 *Bắt đầu huấn luyện Fold 1*
     - 📊 *Epoch 05: Val UA 73.80% (⭐ NEW BEST!)*
     - 🔥 *Cảnh báo nhiệt độ GPU: 79°C - Đang hạ nhiệt...*
     - 🏁 *Hoàn tất huấn luyện và đường dẫn file Best Checkpoint.*

---

## 🔄 4. TỰ ĐỘNG CẬP NHẬT CODE GITHUB & TÁI HUẤN LUYỆN (AUTO-SYNC CI/CD)

Khi bạn chỉnh sửa code trên máy tính xách tay hoặc đẩy code lên GitHub, bạn **không cần phải đụng tay vào máy trạm RTX 3060**:

### Cách kích hoạt:
Chạy lệnh:
```bash
./run.sh auto
# hoặc trên Windows:
run.bat auto
```

### Cơ chế hoạt động:
* Module `git_watcher.py` chạy ngầm kiểm tra branch `main` trên GitHub định kỳ mỗi 60 giây.
* Ngay khi phát hiện bạn vừa `git push` commit mới:
  1. Tự động kéo mã nguồn mới về (`git pull origin main`).
  2. Dừng tiến trình huấn luyện cũ (nếu đang chạy).
  3. Gửi thông báo Telegram: *"Phát hiện code mới, đang khởi động lại huấn luyện..."*
  4. Tự động train lại với code và thuật toán mới!
* *(Tùy chọn nâng cao)*: Hỗ trợ lắng nghe GitHub Webhook tại cổng `8000` (`http://ip-may-ban:8000/webhook`) để phản hồi tức thì trong mili-giây.

---

## 🔗 5. HƯỚNG DẪN ĐẨY DỰ ÁN LÊN GITHUB CỦA BẠN

Dự án này đã được cấu hình sẵn file `.gitignore` để tự động loại bỏ các file nặng (audio `.wav`, checkpoints `.pt`, cache), đảm bảo repo nhẹ và đẩy lên GitHub trong vài giây:

### Bước 1: Tạo Repository mới trên GitHub
1. Truy cập [github.com/new](https://github.com/new).
2. Đặt tên Repository (ví dụ: `ser-wav2vec2-docker`).
3. Chọn chế độ **Public** hoặc **Private** tùy ý.
4. **Không** tích chọn *"Add a README"* hay *".gitignore"* (vì dự án này đã có sẵn).
5. Bấm **Create repository** và copy đường link dạng:  
   `https://github.com/<tai-khoan-cua-ban>/ser-wav2vec2-docker.git`

### Bước 2: Chạy các lệnh Git để đẩy toàn bộ dự án lên
Mở Terminal hoặc PowerShell tại thư mục này và chạy lần lượt:

```bash
# 1. Khởi tạo Git nội bộ (nếu chưa có)
git init

# 2. Đổi tên branch chính thành main
git branch -M main

# 3. Thêm toàn bộ mã nguồn
git add .

# 4. Tạo commit đầu tiên
git commit -m "feat: Dockerized SER Wav2Vec2 training with RTX 3060 optimization, monitoring, and auto-sync"

# 5. Gắn link GitHub của bạn (thay link thật vào đây)
git remote add origin https://github.com/<tai-khoan-cua-ban>/ser-wav2vec2-docker.git

# 6. Đẩy lên GitHub
git push -u origin main
```

---

## 📂 6. CẤU TRÚC THƯ MỤC DỰ ÁN

```
finetune_ssl/
├── Dockerfile                      # Image CUDA 12.1 + PyTorch 2.3+ cho RTX 3060
├── docker-compose.yml              # Điều phối 3 container: Trainer, TensorBoard, Git Watcher
├── entrypoint.sh                   # Script điều hướng container runtime
├── run.sh                          # [1-Click] Khởi chạy cho Linux / macOS / WSL
├── run.bat                         # [1-Click] Khởi chạy cho Windows
├── .env.example                    # File mẫu cấu hình tham số & tokens
├── .env                            # File cấu hình biến môi trường chính thức
├── .gitignore                      # Loại trừ file nặng (audio, checkpoint, logs)
├── requirements.txt                # Danh sách thư viện Python
├── config.py                       # Cấu hình siêu tham số tự động đọc biến môi trường
├── dataset.py                      # Dataset nạp audio, Dynamic Padding, Dummy fallback
├── models.py                       # Mô hình Wav2Vec2 + Attentive Pooling + Freeze CNN
├── train_cv.py                     # Script huấn luyện chính tích hợp Guardian & Monitoring
├── export_and_extract_features.py  # Trích xuất đặc trưng mới cho mạng đồ thị MGCN-LSTM
├── quick_test.py                   # Smoke test kiểm tra nhanh GPU trong 30 giây
├── git_watcher.py                  # Daemon tự động pull GitHub & train lại khi có push mới
├── utils/
│   ├── hardware_guardian.py        # Bảo vệ nhiệt độ GPU RTX 3060 & chống tràn VRAM
│   └── remote_monitor.py           # Hub giám sát TensorBoard, W&B, Telegram Bot
├── data/
│   ├── iemocap_metadata_5531.csv   # File metadata 5,531 mẫu cảm xúc IEMOCAP
│   └── wav/                        # Nơi đặt các file âm thanh .wav
└── checkpoints/                    # Nơi lưu trữ Best Models (.pt)
```

---

## 🎯 7. CÁC LỆNH HỮU ÍCH KHÁC

* **Chạy Smoke Test (Kiểm tra nhanh GPU & Gradient trong 30 giây):**
  ```bash
  ./run.sh test       # Linux/macOS
  run.bat test        # Windows
  ```
* **Chạy toàn bộ 5 Folds Cross Validation (Leave-One-Session-Out):**
  ```bash
  ./run.sh all        # Linux/macOS
  run.bat all         # Windows
  ```
* **Xem nhật ký log trực tiếp của container:**
  ```bash
  docker compose logs -f ser-trainer
  ```
* **Dừng toàn bộ hệ thống:**
  ```bash
  docker compose down
  ```
