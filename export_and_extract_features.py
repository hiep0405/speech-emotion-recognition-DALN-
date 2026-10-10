import os
import sys
import csv
import argparse
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Thiết lập UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FinetuneConfig
from models import Wav2Vec2EmotionClassifier

# Import graph builder từ Đồ án liên ngành
try:
    from graph_builder import build_temporal_graph
    GRAPH_BUILDER_AVAILABLE = True
except ImportError:
    GRAPH_BUILDER_AVAILABLE = False

def extract_features(checkpoint_path, output_feat_dir, output_graph_dir=None, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[*] Đang nạp Checkpoint Fine-tuned: {checkpoint_path}")
    
    # 1. Khởi tạo mô hình và nạp trọng số
    model = Wav2Vec2EmotionClassifier().to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[+] Đã nạp thành công checkpoint từ epoch {ckpt.get('epoch', '?')} (Val UA: {ckpt.get('val_ua', 0):.2f}%)")
    else:
        print(f"[!] CẢNH BÁO: Không tìm thấy checkpoint tại {checkpoint_path}. Sử dụng trọng số backbone gốc.")
        
    model.eval()
    
    os.makedirs(output_feat_dir, exist_ok=True)
    if output_graph_dir:
        os.makedirs(output_graph_dir, exist_ok=True)
        
    # 2. Đọc metadata
    print(f"[*] Đọc danh sách mẫu từ: {FinetuneConfig.METADATA_CSV}")
    records = []
    with open(FinetuneConfig.METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
            
    print(f"[*] Bắt đầu trích xuất đặc trưng cho {len(records)} mẫu IEMOCAP...")
    
    with torch.no_grad():
        for r in tqdm(records, desc="Extracting Fine-tuned Features", dynamic_ncols=True):
            utt_id = r["utterance_id"]
            wav_path = r["wav_path"]
            label = int(r["label"])
            fold = int(r["fold"])
            
            feat_save_path = os.path.join(output_feat_dir, f"{utt_id}.pt")
            
            # Đọc audio
            try:
                audio, sr = sf.read(wav_path, dtype="float32")
            except Exception as e:
                print(f"[!] Lỗi đọc {wav_path}: {e}")
                continue
                
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
                
            # Chuẩn hóa biên độ
            if np.std(audio) > 1e-6:
                audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-7)
                
            input_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)
            
            # Trích xuất representation tầng cuối
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
                outputs = model.wav2vec2(input_values=input_tensor)
                # (1, T_frames, 1024) -> (T_frames, 1024)
                feat = outputs.last_hidden_state.squeeze(0).float().cpu()
                
            # Lưu đặc trưng .pt
            torch.save(feat, feat_save_path)
            
            # Nếu có graph builder, dựng sẵn đồ thị Temporal Multigraph
            if output_graph_dir and GRAPH_BUILDER_AVAILABLE:
                graph_save_path = os.path.join(output_graph_dir, f"{utt_id}.pt")
                graph_data = build_temporal_graph(feat)
                graph_data.y = torch.tensor([label], dtype=torch.long)
                graph_data.fold = fold
                graph_data.utterance_id = utt_id
                torch.save(graph_data, graph_save_path)
                
    print(f"[+] Hoàn tất! Toàn bộ đặc trưng mới đã được lưu tại: {output_feat_dir}")
    if output_graph_dir:
        print(f"[+] Toàn bộ đồ thị cập nhật đã được lưu tại: {output_graph_dir}")

def main():
    parser = argparse.ArgumentParser(description="Trích xuất đặc trưng từ mô hình Wav2Vec 2.0 đã Fine-tune")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(FinetuneConfig.CHECKPOINT_DIR, "best_wav2vec2_ser_fold1.pt"))
    parser.add_argument("--output_feat_dir", type=str, default=FinetuneConfig.EXPORTED_FEAT_DIR)
    parser.add_argument("--build_graphs", action="store_true", help="Tự động dựng cache graphs cho MGCN-LSTM")
    args = parser.parse_args()
    
    graph_dir = os.path.join(FinetuneConfig.PROJECT_DIR, "data", "cached_graphs_finetuned") if args.build_graphs else None
    extract_features(args.checkpoint, args.output_feat_dir, graph_dir)

if __name__ == "__main__":
    main()
