import os
import sys
import torch

# Tự động nạp file .env nếu có
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class FinetuneConfig:
    """
    Cấu hình Siêu tham số & Đường dẫn cho quy trình Fine-tune Wav2Vec 2.0 trên IEMOCAP.
    Tương thích hoàn toàn với Docker, GitHub CI/CD, và card đồ họa NVIDIA GeForce RTX 3060.
    Tự động đọc biến môi trường từ hệ thống hoặc file .env.
    """
    # --------------------------------------------------------------------------
    # 1. Đường dẫn hệ thống linh hoạt (Portable Paths)
    # --------------------------------------------------------------------------
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(MODULE_DIR)
    ROOT_DIR = os.path.dirname(PROJECT_DIR)
    
    # Metadata file: Ưu tiên file nội bộ trong repo, fallback nếu chạy từ đồ án gốc
    LOCAL_METADATA = os.path.join(MODULE_DIR, "data", "iemocap_metadata_5531.csv")
    FALLBACK_METADATA = os.path.join(ROOT_DIR, "temporal_multigraph_ser", "data", "iemocap_metadata_5531.csv")
    METADATA_CSV = os.environ.get("METADATA_CSV", LOCAL_METADATA if os.path.exists(LOCAL_METADATA) else FALLBACK_METADATA)
    
    # Thư mục chứa audio IEMOCAP (Tự động đọc biến môi trường hoặc tìm thư mục data/wav)
    DEFAULT_AUDIO_DIR = os.environ.get("AUDIO_ROOT_DIR")
    if not DEFAULT_AUDIO_DIR:
        candidate_1 = os.path.join(MODULE_DIR, "data", "wav")
        candidate_2 = os.path.join(ROOT_DIR, "data gốc IEMOCAP", "IEMOCAP_full_release")
        DEFAULT_AUDIO_DIR = candidate_1 if os.path.exists(candidate_1) else candidate_2
    AUDIO_ROOT_DIR = DEFAULT_AUDIO_DIR
    
    # Thư mục lưu checkpoint, logs, và feature trích xuất
    CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", os.path.join(MODULE_DIR, "checkpoints"))
    LOGS_DIR = os.environ.get("LOGS_DIR", os.path.join(MODULE_DIR, "logs"))
    RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(MODULE_DIR, "results"))
    EXPORTED_FEAT_DIR = os.environ.get("EXPORTED_FEAT_DIR", os.path.join(MODULE_DIR, "extracted_finetuned_features"))

    # --------------------------------------------------------------------------
    # 2. Backbone Pretrained Model
    # --------------------------------------------------------------------------
    PRETRAINED_MODEL_NAME = os.environ.get(
        "PRETRAINED_MODEL_NAME", 
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )
    
    FEATURE_DIM = 1024          # Kích thước vector hidden của wav2vec2-large
    NUM_CLASSES = 4             # 4 nhãn: ang(0), sad(1), hap(2), neu(3)
    EMO_MAP = {0: "ang", 1: "sad", 2: "hap", 3: "neu"}

    # --------------------------------------------------------------------------
    # 3. Xử lý Âm thanh (Audio Preprocessing)
    # --------------------------------------------------------------------------
    SAMPLE_RATE = 16000         # 16kHz chuẩn cho toàn bộ mô hình SSL
    MAX_AUDIO_SEC = float(os.environ.get("MAX_AUDIO_SEC", 6.0)) # Cắt tối đa 6.0 giây để chống bùng nổ OOM
    MIN_AUDIO_SEC = 0.5
    MAX_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_SEC)  # 96,000 samples

    # --------------------------------------------------------------------------
    # 4. Chiến lược Đóng băng & Cập nhật Trọng số (Freezing Strategy)
    # --------------------------------------------------------------------------
    FREEZE_CNN_ENCODER = True   # BẮT BUỘC: Giữ cố định 7 tầng CNN để bảo vệ feature trích xuất
    # Partial Fine-tune: Chỉ mở 6 tầng cuối trong 24 tầng Transformer
    UNFREEZE_TOP_TRANSFORMER_LAYERS = int(os.environ.get("UNFREEZE_LAYERS", 6))

    # --------------------------------------------------------------------------
    # 5. Siêu tham số Huấn luyện Tối ưu RTX 3060 (12GB / 6GB)
    # --------------------------------------------------------------------------
    # Batch size 8 + Gradient Accumulation 4 -> Effective Batch Size = 32
    # Tiêu tốn ~5.5GB - 6.5GB VRAM, an toàn tuyệt đối cho cả RTX 3060 Laptop và Desktop
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
    GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 4))
    NUM_EPOCHS = int(os.environ.get("EPOCHS", 15))
    
    # Differential Learning Rate (Tốc độ học phân tầng)
    LR_BACKBONE = float(os.environ.get("LR_BACKBONE", 1e-5)) # Cực nhỏ cho Transformer tránh quên kiến thức
    LR_HEAD = float(os.environ.get("LR_HEAD", 5e-4))         # Lớn hơn cho Classifier Head
    WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01))
    WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.1))
    
    DROPOUT_RATE = float(os.environ.get("DROPOUT_RATE", 0.2))
    LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", 0.05))

    # --------------------------------------------------------------------------
    # 6. Quản lý Phần cứng & Bảo Vệ Card Đồ Họa (Hardware Guardian)
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = os.environ.get("USE_AMP", "true").lower() in ("true", "1")
    GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "true").lower() in ("true", "1")
    
    # Ngưỡng nhiệt độ an toàn cho RTX 3060 (°C)
    GPU_MAX_TEMP = int(os.environ.get("GPU_MAX_TEMP", 78))        # Tạm dừng nếu GPU chạm ngưỡng này
    GPU_COOLDOWN_TEMP = int(os.environ.get("GPU_COOLDOWN_TEMP", 70)) # Chờ quạt hạ nhiệt xuống ngưỡng này
    GPU_MEM_FRACTION = float(os.environ.get("GPU_MEM_FRACTION", 0.88)) # Giữ lại 12% VRAM tránh treo OS
    
    # Số luồng DataLoader (Trong Docker/Linux dùng 4 luồng; trên Windows dùng 0)
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4 if sys.platform != "win32" else 0))
    SEED = int(os.environ.get("SEED", 42))

    # --------------------------------------------------------------------------
    # 7. Cấu hình Giám sát Từ xa & Auto-sync
    # --------------------------------------------------------------------------
    WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ser-wav2vec2-rtx3060")
    SYNC_INTERVAL = int(os.environ.get("SYNC_INTERVAL", 60)) # Chu kỳ kiểm tra GitHub (giây)
    GIT_BRANCH = os.environ.get("GIT_BRANCH", "main")

# Tự động tạo thư mục cần thiết
for d in [FinetuneConfig.CHECKPOINT_DIR, FinetuneConfig.LOGS_DIR, 
          FinetuneConfig.RESULTS_DIR, FinetuneConfig.EXPORTED_FEAT_DIR]:
    os.makedirs(d, exist_ok=True)
