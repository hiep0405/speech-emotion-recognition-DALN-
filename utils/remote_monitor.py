import os
import requests
import json
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class RemoteMonitor:
    """
    Hệ thống Giám Sát Từ Xa Toàn Diện (Remote Monitoring & Telemetry Hub):
    1. TensorBoard: Tự động ghi log đồ thị, mở port 6006 xem realtime.
    2. Weights & Biases (W&B): Đồng bộ hóa trực tuyến qua Cloud, theo dõi qua điện thoại mọi lúc mọi nơi.
    3. Telegram Bot Notifier: Bắn thông báo trực tiếp về điện thoại khi đạt mốc mới, có Best Checkpoint,
       hoặc khi GPU quá nhiệt.
    """
    def __init__(self, log_dir, project_name="ser-wav2vec2-finetune", run_name="fold1", config=None):
        self.log_dir = log_dir
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}

        # 1. Khởi tạo TensorBoard
        self.tb_writer = None
        if TENSORBOARD_AVAILABLE:
            try:
                os.makedirs(log_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=log_dir)
                print(f"[RemoteMonitor] ✅ TensorBoard đã kích hoạt tại: {log_dir} (Port 6006)")
            except Exception as e:
                print(f"[RemoteMonitor] Không thể khởi tạo TensorBoard: {e}")

        # 2. Khởi tạo Weights & Biases (W&B)
        self.wandb_run = None
        self.wandb_key = os.environ.get("WANDB_API_KEY", "").strip()
        if self.wandb_key and WANDB_AVAILABLE:
            try:
                wandb.login(key=self.wandb_key, relogin=False)
                self.wandb_run = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", self.project_name),
                    name=self.run_name,
                    config=self.config
                )
                print(f"[RemoteMonitor] ✅ Weights & Biases đã đồng bộ trực tuyến: {self.wandb_run.url}")
            except Exception as e:
                print(f"[RemoteMonitor] Không thể đăng nhập W&B: {e}")
        else:
            if not self.wandb_key:
                print("[RemoteMonitor] ℹ️ WANDB_API_KEY chưa cấu hình (Bỏ qua đồng bộ W&B Cloud).")

        # 3. Khởi tạo Telegram Bot
        self.telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if self.telegram_token and self.telegram_chat_id:
            print("[RemoteMonitor] ✅ Telegram Bot Alert đã kích hoạt.")
            self.send_telegram(f"🚀 **BẮT ĐẦU HUẤN LUYỆN SER WAV2VEC 2.0**\n"
                              f"• Run: `{self.run_name}`\n"
                              f"• Device: `{self.config.get('DEVICE', 'cuda')}`\n"
                              f"• Batch Size: `{self.config.get('BATCH_SIZE', 8)}`")

    def send_telegram(self, message):
        """Gửi thông báo tin nhắn tức thì về Telegram của người dùng."""
        if not (self.telegram_token and self.telegram_chat_id):
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception:
            pass

    def log_epoch(self, epoch, train_loss, train_wa, train_ua, val_loss, val_wa, val_ua, 
                  is_best=False, best_ua=0.0, gpu_stats=None, lr_dict=None):
        """Ghi nhận dữ liệu sau mỗi Epoch lên tất cả các kênh giám sát."""
        # 1. Log lên TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("Loss/Train", train_loss, epoch)
            self.tb_writer.add_scalar("Loss/Val", val_loss, epoch)
            self.tb_writer.add_scalar("Accuracy/Train_WA", train_wa, epoch)
            self.tb_writer.add_scalar("Accuracy/Train_UA", train_ua, epoch)
            self.tb_writer.add_scalar("Accuracy/Val_WA", val_wa, epoch)
            self.tb_writer.add_scalar("Accuracy/Val_UA", val_ua, epoch)
            
            if gpu_stats:
                if "temp_c" in gpu_stats:
                    self.tb_writer.add_scalar("GPU/Temperature_C", gpu_stats["temp_c"], epoch)
                if "vram_used_mb" in gpu_stats:
                    self.tb_writer.add_scalar("GPU/VRAM_Used_MB", gpu_stats["vram_used_mb"], epoch)

            if lr_dict:
                for k, v in lr_dict.items():
                    self.tb_writer.add_scalar(f"LR/{k}", v, epoch)

            self.tb_writer.flush()

        # 2. Log lên Weights & Biases
        if self.wandb_run:
            log_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_wa": train_wa,
                "train_ua": train_ua,
                "val_wa": val_wa,
                "val_ua": val_ua,
            }
            if gpu_stats:
                log_data.update({f"gpu_{k}": v for k, v in gpu_stats.items() if isinstance(v, (int, float))})
            if lr_dict:
                log_data.update(lr_dict)
            self.wandb_run.log(log_data)

        # 3. Bắn Telegram nếu đạt Best hoặc mỗi 5 epochs
        if is_best or (epoch % 5 == 0):
            star = " ⭐ NEW BEST!" if is_best else ""
            temp_str = f" | Temp: {gpu_stats.get('temp_c', 0)}°C" if gpu_stats else ""
            msg = (f"📊 **Epoch [{epoch:02d}]{star}**\n"
                   f"• Val UA: **{val_ua:.2f}%** (Best: {best_ua:.2f}%)\n"
                   f"• Val WA: {val_wa:.2f}% | Val Loss: {val_loss:.4f}\n"
                   f"• Train UA: {train_ua:.2f}% | Train Loss: {train_loss:.4f}{temp_str}")
            self.send_telegram(msg)

    def log_finish(self, best_ua, checkpoint_path):
        """Gửi thông báo khi hoàn tất toàn bộ quá trình huấn luyện."""
        msg = (f"🏁 **HUẤN LUYỆN HOÀN TẤT!**\n"
               f"• Best Unweighted Accuracy (UA): **{best_ua:.2f}%**\n"
               f"• Checkpoint: `{checkpoint_path}`")
        print("\n" + msg)
        self.send_telegram(msg)
        if self.wandb_run:
            self.wandb_run.finish()
        if self.tb_writer:
            self.tb_writer.close()
