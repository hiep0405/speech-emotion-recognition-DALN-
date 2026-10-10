@echo off
chcp 65001 > nul
echo ==========================================================================
echo     ĐANG ĐẨY DỰ ÁN LÊN GITHUB: hiep0405/speech-emotion-recognition-DALN-
echo ==========================================================================
cd /d "%~dp0"
git push -u origin main --force
echo.
if %errorlevel% equ 0 (
    echo [OK] ĐÃ ĐẨY LÊN GITHUB THÀNH CÔNG!
) else (
    echo [!] Nếu gặp lỗi xác thực, vui lòng đăng nhập vào trình duyệt khi Git yêu cầu.
)
pause
