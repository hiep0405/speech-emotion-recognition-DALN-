import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from config import FinetuneConfig

class AttentivePooling(nn.Module):
    """
    Tầng Attentive Pooling giúp mô hình tự học trọng số tập trung vào
    các đoạn âm thanh chứa nhiều sắc thái cảm xúc nhất.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        # x: (B, T, D)
        scores = self.attn(x)  # (B, T, 1)
        if mask is not None:
            # Mask out padding frames (mask: B, T)
            # Dùng -1e4 thay vì -1e9 để tránh overflow khi dùng FP16 (torch.Half)
            mask_expanded = mask.unsqueeze(-1)
            scores = scores.masked_fill(~mask_expanded, -1e4)
            
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        pooled = torch.sum(x * weights, dim=1)  # (B, D)
        return pooled, weights


class Wav2Vec2EmotionClassifier(nn.Module):
    """
    Mô hình Fine-tuning Wav2Vec 2.0 cho bài toán Speech Emotion Recognition (SER)
    Hỗ trợ Freeze CNN Encoder, Partial Unfreeze Transformer và Attentive Pooling.
    """
    def __init__(self, model_name=FinetuneConfig.PRETRAINED_MODEL_NAME, num_classes=FinetuneConfig.NUM_CLASSES):
        super().__init__()
        print(f"[*] Khởi tạo Backbone SSL từ: {model_name}...")
        self.config = AutoConfig.from_pretrained(model_name)
        self.wav2vec2 = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_dim = self.config.hidden_size  # 1024 cho Wav2Vec2 Large
        
        # 1. Bật Gradient Checkpointing nếu được cấu hình
        if FinetuneConfig.GRADIENT_CHECKPOINTING and hasattr(self.wav2vec2, "gradient_checkpointing_enable"):
            self.wav2vec2.gradient_checkpointing_enable()
            print("[+] Đã kích hoạt Gradient Checkpointing để tiết kiệm VRAM.")

        # 2. Đóng băng CNN Feature Extractor (7 convolutional layers)
        if FinetuneConfig.FREEZE_CNN_ENCODER and hasattr(self.wav2vec2, "freeze_feature_encoder"):
            self.wav2vec2.freeze_feature_encoder()
            print("[+] Đã đóng băng hoàn toàn CNN Feature Extractor (Bảo toàn trọng số âm học cơ sở).")
            
        # 3. Partial Unfreezing (Chỉ mở K tầng Transformer cuối nếu cấu hình)
        if FinetuneConfig.UNFREEZE_TOP_TRANSFORMER_LAYERS is not None:
            self._apply_partial_unfreeze(FinetuneConfig.UNFREEZE_TOP_TRANSFORMER_LAYERS)

        # 4. Pooling Layer & Classification Head
        self.pooler = AttentivePooling(hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(FinetuneConfig.DROPOUT_RATE),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(FinetuneConfig.DROPOUT_RATE),
            nn.Linear(256, num_classes)
        )

    def _apply_partial_unfreeze(self, num_top_layers):
        """Đóng băng các tầng Transformer dưới, chỉ cho phép huấn luyện K tầng Transformer trên cùng."""
        if hasattr(self.wav2vec2, "encoder") and hasattr(self.wav2vec2.encoder, "layers"):
            total_layers = len(self.wav2vec2.encoder.layers)
            freeze_until = max(0, total_layers - num_top_layers)
            
            for i, layer in enumerate(self.wav2vec2.encoder.layers):
                if i < freeze_until:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
                        
            print(f"[+] Áp dụng Partial Fine-tuning: Đóng băng {freeze_until}/{total_layers} tầng đầu, "
                  f"chỉ mở {num_top_layers} tầng Transformer trên cùng.")

    def forward(self, input_values, attention_mask=None):
        """
        Forward Pass:
        input_values: (B, num_samples) - Raw waveform audio
        attention_mask: (B, num_samples) - 1 cho sample thật, 0 cho padding
        """
        # Trích xuất đặc trưng qua Backbone Wav2Vec 2.0
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        # last_hidden_state: (B, T_frames, hidden_dim)
        frame_features = outputs.last_hidden_state
        
        # Tạo mask cho các frame sau khi qua CNN Downsampling
        downsampled_mask = None
        if attention_mask is not None and hasattr(self.wav2vec2, "_get_feature_vector_attention_mask"):
            downsampled_mask = self.wav2vec2._get_feature_vector_attention_mask(
                frame_features.shape[1], attention_mask
            ).bool()

        # Attentive Pooling qua các frame
        utterance_vec, attn_weights = self.pooler(frame_features, mask=downsampled_mask)
        
        # Dự đoán phân loại cảm xúc
        logits = self.classifier(utterance_vec)
        
        return {
            "logits": logits,
            "utterance_vec": utterance_vec,
            "frame_features": frame_features,
            "attn_weights": attn_weights
        }

    def extract_frame_features(self, input_values, attention_mask=None):
        """
        Dùng cho việc xuất đặc trưng sau fine-tune để cấp sang mạng MGCN-LSTM
        Trả về frame_features: (B, T_frames, 1024)
        """
        with torch.no_grad():
            outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
            return outputs.last_hidden_state
