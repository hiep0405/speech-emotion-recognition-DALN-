import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import recall_score, accuracy_score, classification_report, confusion_matrix

# Thiết lập encoding UTF-8
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
from utils.remote_monitor import RemoteMonitor

def set_seed(seed=FinetuneConfig.SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def compute_metrics(y_true, y_pred):
    """Tính toán Weighted Accuracy (WA) và Unweighted Accuracy (UA) chuẩn SER."""
    wa = accuracy_score(y_true, y_pred) * 100.0
    ua = recall_score(y_true, y_pred, average="macro") * 100.0
    return wa, ua

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input_values"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)
            
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=FinetuneConfig.USE_AMP):
                outputs = model(inputs, attention_mask=mask)
                loss = criterion(outputs["logits"], targets)
                
            total_loss += loss.item() * len(targets)
            preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
    val_loss = total_loss / len(val_loader.dataset)
    val_wa, val_ua = compute_metrics(all_targets, all_preds)
    return val_loss, val_wa, val_ua, all_targets, all_preds

def train_single_fold(test_fold, epochs=FinetuneConfig.NUM_EPOCHS):
    # 0. Kích hoạt tối ưu phần cứng cho RTX 3060 và bảo vệ nhiệt độ
    HardwareGuardian.optimize_for_rtx3060(FinetuneConfig.GPU_MEM_FRACTION)
    guardian = HardwareGuardian(
        max_temp=FinetuneConfig.GPU_MAX_TEMP,
        cooldown_temp=FinetuneConfig.GPU_COOLDOWN_TEMP
    )
    
    device = torch.device(FinetuneConfig.DEVICE)
    print("=" * 70)
    print(f"[*] BẮT ĐẦU HUẤN LUYỆN FOLD {test_fold} (Leave Session {test_fold} Out)")
    print(f"[*] Thiết bị: {device} | Mixed Precision (FP16): {FinetuneConfig.USE_AMP}")
    print(f"[*] Batch size: {FinetuneConfig.BATCH_SIZE} | Accumulation: {FinetuneConfig.GRAD_ACCUM_STEPS} (Effective: {FinetuneConfig.BATCH_SIZE * FinetuneConfig.GRAD_ACCUM_STEPS})")
    print("=" * 70)
    
    # Khởi tạo Hub giám sát từ xa (TensorBoard, W&B, Telegram)
    fold_log_dir = os.path.join(FinetuneConfig.LOGS_DIR, f"fold_{test_fold}")
    monitor = RemoteMonitor(
        log_dir=fold_log_dir,
        project_name=FinetuneConfig.WANDB_PROJECT,
        run_name=f"wav2vec2_ser_fold_{test_fold}",
        config={
            "fold": test_fold,
            "epochs": epochs,
            "batch_size": FinetuneConfig.BATCH_SIZE,
            "grad_accum": FinetuneConfig.GRAD_ACCUM_STEPS,
            "lr_backbone": FinetuneConfig.LR_BACKBONE,
            "lr_head": FinetuneConfig.LR_HEAD,
            "unfreeze_layers": FinetuneConfig.UNFREEZE_TOP_TRANSFORMER_LAYERS,
            "device": str(device)
        }
    )
    
    all_folds = [1, 2, 3, 4, 5]
    train_folds = [f for f in all_folds if f != test_fold]
    val_folds = [test_fold]
    
    # 1. Chuẩn bị Dataloader
    train_dataset = IEMOCAPAudioDataset(fold_list=train_folds)
    val_dataset = IEMOCAPAudioDataset(fold_list=val_folds)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=FinetuneConfig.BATCH_SIZE,
        shuffle=True,
        collate_fn=audio_collate_fn,
        num_workers=FinetuneConfig.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=FinetuneConfig.BATCH_SIZE,
        shuffle=False,
        collate_fn=audio_collate_fn,
        num_workers=FinetuneConfig.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )
    
    # 2. Khởi tạo Mô hình
    model = Wav2Vec2EmotionClassifier().to(device)
    
    # 3. Phân tách nhóm tham số để áp dụng Differential Learning Rate
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "wav2vec2" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": FinetuneConfig.LR_BACKBONE, "weight_decay": FinetuneConfig.WEIGHT_DECAY},
        {"params": head_params, "lr": FinetuneConfig.LR_HEAD, "weight_decay": FinetuneConfig.WEIGHT_DECAY}
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    total_steps = (len(train_loader) // FinetuneConfig.GRAD_ACCUM_STEPS) * epochs
    warmup_steps = int(total_steps * FinetuneConfig.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=FinetuneConfig.LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=(FinetuneConfig.USE_AMP and device.type == "cuda"))
    
    best_val_ua = 0.0
    best_checkpoint_path = os.path.join(FinetuneConfig.CHECKPOINT_DIR, f"best_wav2vec2_ser_fold{test_fold}.pt")
    
    # 4. Vòng lặp huấn luyện chính
    for epoch in range(1, epochs + 1):
        # Kiểm tra nhiệt độ phần cứng trước mỗi epoch
        guardian.check_and_cool(notify_callback=monitor.send_telegram)
        
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            inputs = batch["input_values"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)
            
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=FinetuneConfig.USE_AMP):
                outputs = model(inputs, attention_mask=mask)
                loss = criterion(outputs["logits"], targets)
                # Chia loss theo gradient accumulation steps
                loss = loss / FinetuneConfig.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            
            train_loss += loss.item() * FinetuneConfig.GRAD_ACCUM_STEPS * len(targets)
            preds = torch.argmax(outputs["logits"], dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targets.cpu().numpy())
            
            # Cập nhật gradient sau mỗi GRAD_ACCUM_STEPS
            if (step + 1) % FinetuneConfig.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
        train_loss = train_loss / len(train_dataset)
        train_wa, train_ua = compute_metrics(train_targets, train_preds)
        
        # Đánh giá trên Validation Set
        val_loss, val_wa, val_ua, val_targets, val_preds = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        
        # Lấy số liệu GPU hiện tại
        gpu_stats = guardian.get_gpu_stats()
        
        # Ghi log kết quả
        is_best = val_ua > best_val_ua
        if is_best:
            best_val_ua = val_ua
            torch.save({
                "epoch": epoch,
                "fold": test_fold,
                "model_state_dict": model.state_dict(),
                "val_ua": val_ua,
                "val_wa": val_wa,
                "config": vars(FinetuneConfig)
            }, best_checkpoint_path)
            mark = "(*) BEST"
        else:
            mark = ""
            
        temp_str = f"| GPU: {gpu_stats.get('temp_c', 0)}°C, VRAM: {gpu_stats.get('vram_used_mb', 0):.0f}MB" if gpu_stats else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} WA: {train_wa:.2f}% UA: {train_ua:.2f}% | "
              f"Val Loss: {val_loss:.4f} WA: {val_wa:.2f}% UA: {val_ua:.2f}% {mark} {temp_str}")
        
        # Đẩy dữ liệu lên TensorBoard, W&B, Telegram
        lr_dict = {
            "backbone": optimizer.param_groups[0]["lr"],
            "head": optimizer.param_groups[1]["lr"]
        }
        monitor.log_epoch(
            epoch=epoch,
            train_loss=train_loss, train_wa=train_wa, train_ua=train_ua,
            val_loss=val_loss, val_wa=val_wa, val_ua=val_ua,
            is_best=is_best, best_ua=best_val_ua,
            gpu_stats=gpu_stats, lr_dict=lr_dict
        )
              
    monitor.log_finish(best_val_ua, best_checkpoint_path)
    print(f"\n[+] Hoàn tất Fold {test_fold}! Best Val UA: {best_val_ua:.2f}% | Đã lưu: {best_checkpoint_path}\n")
    return best_val_ua

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec 2.0 trên IEMOCAP (Tối ưu RTX 3060 & Docker)")
    parser.add_argument("--fold", type=str, default="1", help="Fold kiểm thử (1-5 hoặc 'all')")
    parser.add_argument("--epochs", type=int, default=FinetuneConfig.NUM_EPOCHS)
    args = parser.parse_args()
    
    set_seed()
    
    if args.fold.lower() == "all":
        results = {}
        for f in [1, 2, 3, 4, 5]:
            results[f] = train_single_fold(f, epochs=args.epochs)
        print("=" * 70)
        print("[*] TỔNG KẾT 5-FOLD CROSS VALIDATION:")
        for f, ua in results.items():
            print(f" - Fold {f}: UA = {ua:.2f}%")
        print(f" => TRUNG BÌNH 5 FOLDS: UA = {np.mean(list(results.values())):.2f}%")
        print("=" * 70)
    else:
        fold_num = int(args.fold)
        train_single_fold(fold_num, epochs=args.epochs)

if __name__ == "__main__":
    main()
