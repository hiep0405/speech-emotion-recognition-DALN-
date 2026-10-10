# 🚀 HƯỚNG DẪN CHI TIẾT ĐẨY DỰ ÁN LÊN GITHUB

Tài liệu này hướng dẫn bạn từng bước cách liên kết và đẩy (push) toàn bộ dự án này lên GitHub.

---

## 📌 BƯỚC 1: LẤY LINK REPOSITORY TRÊN GITHUB

1. Đăng nhập vào tài khoản GitHub của bạn tại: [https://github.com](https://github.com)
2. Nhấn vào dấu `+` ở góc trên cùng bên phải và chọn **New repository** (hoặc truy cập trực tiếp: [https://github.com/new](https://github.com/new)).
3. Điền các thông tin:
   - **Repository name:** `speech-emotion-recognition-wav2vec2` (hoặc tên bất kỳ bạn thích).
   - **Description:** `Speech Emotion Recognition fine-tuning Wav2Vec 2.0 with Docker, RTX 3060 optimization, and remote monitoring`.
   - **Public / Private:** Chọn tùy ý theo nhu cầu của bạn.
   - ⚠️ **LƯU Ý QUAN TRỌNG:** **KHÔNG** tích chọn các ô *"Add a README file"*, *"Add .gitignore"*, hay *"Choose a license"*, vì dự án trong máy bạn đã có sẵn toàn bộ các file này rồi!
4. Nhấn nút xanh **Create repository**.
5. Sau khi tạo xong, GitHub sẽ hiển thị đường link repository của bạn ở dạng:
   - **HTTPS:** `https://github.com/<tai-khoan-cua-ban>/<ten-repo>.git`
   - **SSH:** `git@github.com:<tai-khoan-cua-ban>/<ten-repo>.git`
   *(Hãy copy đường link này!)*

---

## 📌 BƯỚC 2: CHẠY CÁC LỆNH PUSH TỪ MÁY TÍNH CỦA BẠN

Mở **PowerShell** hoặc **Command Prompt (CMD)** hoặc **Git Bash** tại thư mục dự án:
`c:\Users\HIEPHIEP\Downloads\speech_emo_recognition\SER-ĐỒ ÁN LIÊN NGÀNH\finetune_ssl`

Sau đó chạy lần lượt các lệnh sau:

### Lệnh 1: Khởi tạo Git repository trong thư mục này
```powershell
git init
```

### Lệnh 2: Đặt tên nhánh chính là `main`
```powershell
git branch -M main
```

### Lệnh 3: Thêm tất cả các file cấu hình và mã nguồn vào staging
*(File `.gitignore` sẽ tự động loại bỏ các file nặng như `.wav`, checkpoints `.pt`, logs để repo nhẹ nhất)*
```powershell
git add .
```

### Lệnh 4: Tạo commit lưu trạng thái
```powershell
git commit -m "feat: complete dockerized SER project with RTX 3060 optimization, remote monitor, and auto-sync"
```

### Lệnh 5: Gắn link GitHub bạn vừa copy ở Bước 1 vào `origin`
*(Thay thế `<LINK_GITHUB_CUA_BAN>` bằng link thật của bạn)*:
```powershell
git remote add origin <LINK_GITHUB_CUA_BAN>
```
*Ví dụ:*
```powershell
git remote add origin https://github.com/hiep0405/speech-emotion-recognition-wav2vec2.git
```

*(Mẹo: Nếu trước đó đã từng gán remote và báo lỗi `remote origin already exists`, bạn chỉ cần chạy: `git remote set-url origin <LINK_GITHUB_CUA_BAN>`)*

### Lệnh 6: Đẩy toàn bộ dự án lên GitHub
```powershell
git push -u origin main
```

---

## 📌 BƯỚC 3: NGƯỜI KHÁC CLONE VỀ VÀ CHẠY 1 LỆNH NHƯ THẾ NÀO?

Bất kỳ người nào (hoặc máy chủ / máy bạn bè của bạn) muốn chạy mô hình này chỉ cần:

```bash
# 1. Clone về máy
git clone <LINK_GITHUB_CUA_BAN>
cd speech-emotion-recognition-wav2vec2

# 2. Chạy đúng 1 lệnh duy nhất!
# Trên Windows:
run.bat

# Hoặc trên Linux/WSL:
./run.sh
```

Không cần cài Python, không cần cài CUDA, không cần cài PyTorch, Docker sẽ tự động dựng toàn bộ môi trường và kích hoạt card đồ họa RTX 3060 để chạy ngay lập tức!
