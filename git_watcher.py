import os
import sys
import time
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from config import FinetuneConfig

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def send_alert(message):
    """Gửi thông báo Telegram về tiến trình cập nhật code."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id and REQUESTS_AVAILABLE:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
        except Exception:
            pass


class GitWatcher:
    """
    Trình Giám Sát Tự Động Cập Nhật Code & Tái Huấn Luyện (CI/CD Auto-Sync & Retrainer):
    - Tự động theo dõi repository GitHub theo chu kỳ (mặc định 60s) hoặc qua Webhook.
    - Khi bạn 'git push' code mới từ máy khác lên GitHub, máy này sẽ tự động:
      1. Kéo mã nguồn mới nhất về ('git pull').
      2. Ngắt tiến trình huấn luyện cũ (nếu đang chạy).
      3. Khởi động lại quá trình huấn luyện với thuật toán/code mới!
    """
    def __init__(self, branch="main", interval=60, fold="1", epochs=15):
        self.branch = os.environ.get("GIT_BRANCH", branch)
        self.interval = int(os.environ.get("SYNC_INTERVAL", interval))
        self.fold = os.environ.get("FOLD", str(fold))
        self.epochs = int(os.environ.get("EPOCHS", str(epochs)))
        self.training_process = None
        self.running = True

    def run_command(self, cmd):
        """Thực thi lệnh shell và lấy output."""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return None

    def start_training(self):
        """Khởi động tiến trình huấn luyện train_cv.py."""
        if self.training_process and self.training_process.poll() is None:
            print("[GitWatcher] 🛑 Đang dừng tiến trình huấn luyện cũ...")
            self.training_process.terminate()
            try:
                self.training_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.training_process.kill()

        cmd = [sys.executable, "train_cv.py", "--fold", self.fold, "--epochs", str(self.epochs)]
        print(f"[GitWatcher] 🚀 KHỞI ĐỘNG HUẤN LUYỆN MỚI: {' '.join(cmd)}")
        send_alert(f"🚀 **AUTO-TRAIN KÍCH HOẠT!**\n• Bắt đầu huấn luyện với code mới nhất trên branch `{self.branch}` (Fold {self.fold}, {self.epochs} epochs).")
        
        # Chạy subprocess không chặn luồng giám sát
        self.training_process = subprocess.Popen(cmd)

    def check_for_updates(self):
        """Kiểm tra xem GitHub có commit mới không."""
        # 1. Fetch từ remote
        fetch_res = self.run_command(f"git fetch origin {self.branch}")
        if fetch_res is None:
            # Nếu chưa cấu hình remote origin thì bỏ qua
            return False

        # 2. So sánh SHA commit hiện tại và origin
        local_hash = self.run_command(f"git rev-parse HEAD")
        remote_hash = self.run_command(f"git rev-parse origin/{self.branch}")

        if not local_hash or not remote_hash:
            return False

        if local_hash != remote_hash:
            commit_info = self.run_command(f"git log -1 --pretty=format:'%h - %s (%an)' origin/{self.branch}")
            print("\n" + "=" * 70)
            print(f"[GitWatcher] 🔔 PHÁT HIỆN CODE MỚI TRÊN GITHUB!")
            print(f" - Commit: {commit_info}")
            print(f" - Bắt đầu pull code và restart training...")
            print("=" * 70)

            send_alert(f"🔄 **PHÁT HIỆN COMMIT MỚI TRÊN GITHUB**\n• `{commit_info}`\n• Đang pull code và chuẩn bị train lại...")

            # 3. Pull code mới
            pull_res = self.run_command(f"git pull origin {self.branch}")
            print(f"[GitWatcher] Git pull output:\n{pull_res}")

            # 4. Kích hoạt huấn luyện lại
            self.start_training()
            return True

        return False

    def loop(self):
        """Vòng lặp giám sát định kỳ."""
        print("=" * 70)
        print(f"[*] AUTO-SYNC GIT WATCHER ĐANG CHẠY")
        print(f" - Branch theo dõi: origin/{self.branch}")
        print(f" - Chu kỳ kiểm tra: {self.interval} giây")
        print(f" - Chế độ: Tự động pull & tái huấn luyện khi có push mới")
        print("=" * 70)

        # Chạy train lần đầu tiên khi khởi động
        self.start_training()

        while self.running:
            time.sleep(self.interval)
            try:
                self.check_for_updates()
            except Exception as e:
                print(f"[GitWatcher] Lỗi khi kiểm tra GitHub: {e}")


class WebhookHandler(BaseHTTPRequestHandler):
    """Webhook server nhận tín hiệu trực tiếp từ GitHub Webhook (POST /webhook)."""
    watcher_instance = None

    def do_POST(self):
        if self.path == "/webhook":
            print("[GitWatcher] ⚡ Nhận được tín hiệu Webhook từ GitHub!")
            if WebhookHandler.watcher_instance:
                WebhookHandler.watcher_instance.check_for_updates()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Tắt log HTTP rườm rà


def run_webhook_server(watcher, port=8000):
    WebhookHandler.watcher_instance = watcher
    server = HTTPServer(("0.0.0.0", port), WebhookHandler)
    print(f"[GitWatcher] 🌐 Webhook server lắng nghe tại port {port} (POST /webhook)")
    server.serve_forever()


if __name__ == "__main__":
    watcher = GitWatcher()
    
    # Khởi động Webhook server ở luồng phụ (nếu muốn dùng Webhook ngoài việc polling)
    webhook_thread = threading.Thread(target=run_webhook_server, args=(watcher,), daemon=True)
    webhook_thread.start()

    # Chạy vòng lặp polling
    try:
        watcher.loop()
    except KeyboardInterrupt:
        print("\n[GitWatcher] Đã dừng trình giám sát.")
        if watcher.training_process:
            watcher.training_process.terminate()
