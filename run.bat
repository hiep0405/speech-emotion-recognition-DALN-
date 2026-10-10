@echo off
REM ==============================================================================
REM Script Khoi Chay 1 Lenh Duy Nhat (Windows CMD / PowerShell)
REM Usage:
REM   run.bat         -> Huan luyen chuan (Fold 1) + Mo TensorBoard
REM   run.bat test    -> Smoke test kiem tra GPU va mo hinh
REM   run.bat all     -> Huan luyen toan bo 5 Folds
REM   run.bat auto    -> Che do tu dong pull GitHub & train lai khi co commit moi
REM ==============================================================================

chcp 65001 > nul
echo ==========================================================================
echo     SPEECH EMOTION RECOGNITION (SER) - 1-CLICK DOCKER RUNNER (WINDOWS)
echo     Toi uu hoa chuyen sau: NVIDIA GeForce RTX 3060 (12GB / 6GB)
echo ==========================================================================

REM 1. Kiem tra Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Loi: May tinh chua cai dat Docker hoac Docker Desktop chua duoc bat.
    echo Vui long mo Docker Desktop truoc khi chay script nay.
    pause
    exit /b 1
)

REM 2. Kiem tra GPU NVIDIA
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [+] Phat hien GPU NVIDIA thanh cong.
) else (
    echo [!] Canh bao: Khong tim thay nvidia-smi. Hay dam bao da bat WSL 2 GPU Support trong Docker Desktop.
)

REM 3. Tu dong tao file .env neu chua co
if not exist .env (
    echo [+] Khoi tao file .env tu .env.example...
    copy .env.example .env > nul
)

REM 4. Tu dong tao thu muc
if not exist checkpoints mkdir checkpoints
if not exist logs mkdir logs
if not exist results mkdir results
if not exist extracted_finetuned_features mkdir extracted_finetuned_features
if not exist data\wav mkdir data\wav

REM 5. Xu ly tham so
set CMD_ARG=%1
if "%CMD_ARG%"=="" set CMD_ARG=train

if "%CMD_ARG%"=="test" (
    echo [*] Khoi chay Smoke Test trong Docker...
    docker compose run --rm ser-trainer test
    goto end
)

if "%CMD_ARG%"=="all" (
    echo [*] Khoi chay Huan luyen 5 Folds trong Docker...
    docker compose run --rm ser-trainer train-all
    goto end
)

if "%CMD_ARG%"=="auto" (
    echo [*] Khoi chay Che do Auto-Sync GitHub ^& Retrain...
    docker compose --profile auto up --build
    goto end
)

echo [*] Khoi chay Huan luyen ^& May chu Giam sat TensorBoard (Port 6006)...
docker compose up --build

:end
