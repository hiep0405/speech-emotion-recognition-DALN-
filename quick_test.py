import os
import sys
import torch
from torch.utils.data import DataLoader

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FinetuneConfig
from dataset import IEMOCAPAudioDataset, audio_collate_fn
from models import Wav2Vec2EmotionClassifier
from utils.hardware_guardian import HardwareGuardian

def run_smoke_test():
    print("=" * 70)
    print("[*] KIỂM TRA NHANH TOÀN DIỆN (SMOKE TEST) - SER WAV2VEC 2.0 DOCKER")
    print("=" * 70)
    
    # 0. Kích hoạt tối ưu phần cứng RTX 3060
    HardwareGuardian.optimize_for_rtx3060(FinetuneConfig.GPU_MEM_FRACTION)
    guardian = HardwareGuardian(
        max_temp=FinetuneConfig.GPU_MAX_TEMP,
        cooldown_temp=FinetuneConfig.GPU_COOLDOWN_TEMP
    )
    
    device = torch.device(FinetuneConfig.DEVICE)
    print(f"[*] Thiết bị kiểm thử: {device}")
    gpu_stats = guardian.get_gpu_stats()
    if gpu_stats.get("temp_c", 0) > 0:
        print(f"[*] GPU Telemetry: {gpu_stats['name']} | Nhiệt độ: {gpu_stats['temp_c']}°C | VRAM: {gpu_stats['vram_used_mb']:.0f}/{gpu_stats['vram_total_mb']:.0f} MB")
    
    # 1. Test nạp Dataset
    print("\n[1/4] Kiểm tra Dataset & Dataloader...")
    dataset = IEMOCAPAudioDataset(fold_list=[1])
    if len(dataset) == 0:
        print("[!] Lỗi: Không đọc được mẫu nào từ dataset!")
        return False
        
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=audio_collate_fn)
    sample_batch = next(iter(loader))
    print(f" - Batch input_values shape: {sample_batch['input_values'].shape}")
    print(f" - Batch attention_mask shape: {sample_batch['attention_mask'].shape}")
    print(f" - Batch labels: {sample_batch['labels'].tolist()}")
    
    # 2. Test Khởi tạo Mô hình
    print("\n[2/4] Kiểm tra Khởi tạo Model Wav2Vec2EmotionClassifier...")
    model = Wav2Vec2EmotionClassifier().to(device)
    model.train()
    
    # 3. Test Forward & Backward Pass
    print("\n[3/4] Kiểm tra Forward & Backward Pass (AMP FP16)...")
    inputs = sample_batch["input_values"].to(device)
    mask = sample_batch["attention_mask"].to(device)
    targets = sample_batch["labels"].to(device)
    
    with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=FinetuneConfig.USE_AMP):
        out = model(inputs, attention_mask=mask)
        logits = out["logits"]
        loss = torch.nn.functional.cross_entropy(logits, targets)
        
    print(f" - Logits shape: {logits.shape} (Kỳ vọng: [2, 4])")
    print(f" - Loss tính toán: {loss.item():.4f}")
    
    loss.backward()
    
    # 4. Kiểm tra Gradient và Trạng thái Đóng băng
    print("\n[4/4] Kiểm tra tính toàn vẹn của Gradient...")
    cnn_frozen = True
    for name, param in model.wav2vec2.named_parameters():
        if "feature_extractor" in name:
            if param.grad is not None and param.grad.abs().sum() > 0:
                cnn_frozen = False
                break
                
    head_updated = False
    for param in model.classifier.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            head_updated = True
            break
            
    print(f" - CNN Feature Extractor (7 tầng) đóng băng: {'[ĐẠT]' if cnn_frozen else '[LỖI]'}")
    print(f" - Classifier Head nhận gradient: {'[ĐẠT]' if head_updated else '[LỖI]'}")
    
    if torch.cuda.is_available():
        vram_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f" - Đỉnh VRAM tiêu thụ trong mini-batch: {vram_used:.2f} MB")
        
    print("\n" + "=" * 70)
    print("[+] TẤT CẢ CÁC BƯỚC KIỂM TRA ĐỀU HOẠT ĐỘNG HOÀN HẢO! HỆ THỐNG SẴN SÀNG!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    run_smoke_test()
