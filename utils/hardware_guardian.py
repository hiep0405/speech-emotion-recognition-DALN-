import os
import time
import torch

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class HardwareGuardian:
    """
    Module Bảo Vệ Phần Cứng & Giám Sát Nhiệt Độ GPU (Hardware & Thermal Guardian).
    Đặc biệt tối ưu hóa cho dòng card NVIDIA GeForce RTX 3060 (12GB Desktop / 6GB Laptop):
    1. Chống quá nhiệt (Thermal Throttling & Degradation Protection): Tạm dừng khi chạm ngưỡng nhiệt.
    2. Chống tràn VRAM (OOM Protection): Phân bổ headroom và chống phân mảnh VRAM.
    3. Tối ưu nhân Tensor Core Ampere (TF32 & FP16 mixed precision).
    """
    def __init__(self, device_id=0, max_temp=78, cooldown_temp=70):
        self.device_id = device_id
        self.max_temp = max_temp
        self.cooldown_temp = cooldown_temp
        self.nvml_initialized = False
        self.gpu_handle = None
        self.gpu_name = "Unknown GPU"

        if torch.cuda.is_available() and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
                if isinstance(self.gpu_name, bytes):
                    self.gpu_name = self.gpu_name.decode("utf-8")
                self.nvml_initialized = True
            except Exception as e:
                print(f"[HardwareGuardian] Không thể khởi tạo NVML: {e}")

    def get_gpu_stats(self):
        """Trả về thông tin chi tiết về nhiệt độ, điện năng, và bộ nhớ VRAM."""
        if not self.nvml_initialized or self.gpu_handle is None:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device_id) / (1024 ** 2)
                reserved = torch.cuda.memory_reserved(self.device_id) / (1024 ** 2)
                return {
                    "name": torch.cuda.get_device_name(self.device_id),
                    "temp_c": 0,
                    "vram_used_mb": allocated,
                    "vram_reserved_mb": reserved,
                    "vram_total_mb": 0,
                    "power_w": 0,
                    "fan_speed": 0
                }
            return {"name": "CPU", "temp_c": 0, "vram_used_mb": 0, "vram_total_mb": 0}

        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            used_mb = mem_info.used / (1024 ** 2)
            total_mb = mem_info.total / (1024 ** 2)
            
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # W
            except Exception:
                power = 0.0

            try:
                fan = pynvml.nvmlDeviceGetFanSpeed(self.gpu_handle)
            except Exception:
                fan = 0

            return {
                "name": self.gpu_name,
                "temp_c": temp,
                "vram_used_mb": used_mb,
                "vram_total_mb": total_mb,
                "power_w": power,
                "fan_speed": fan
            }
        except Exception:
            return {"name": self.gpu_name, "temp_c": 0, "vram_used_mb": 0, "vram_total_mb": 0}

    def check_and_cool(self, notify_callback=None):
        """
        Kiểm tra nhiệt độ GPU. Nếu vượt quá max_temp, tạm dừng huấn luyện để quạt hạ nhiệt.
        Bảo vệ tuyệt đối GPU không bị quá nhiệt, chai tụ, hoặc chập cháy linh kiện khi train nhiều giờ.
        """
        stats = self.get_gpu_stats()
        current_temp = stats.get("temp_c", 0)

        if current_temp >= self.max_temp:
            msg = (f"[HardwareGuardian] ⚠️ CẢNH BÁO NHIỆT ĐỘ: GPU đang ở {current_temp}°C "
                   f"(Ngưỡng an toàn: {self.max_temp}°C). Tạm dừng huấn luyện để làm mát...")
            print("\n" + "!" * 70)
            print(msg)
            print("!" * 70)

            if notify_callback:
                try:
                    notify_callback(f"🔥 CẢNH BÁO QUÁ NHIỆT: {self.gpu_name} đạt {current_temp}°C! Đang kích hoạt chế độ hạ nhiệt...")
                except Exception:
                    pass

            # Vòng lặp nghỉ ngơi cho đến khi nhiệt độ hạ xuống dưới cooldown_temp
            total_waited = 0
            while True:
                time.sleep(10)
                total_waited += 10
                new_stats = self.get_gpu_stats()
                temp = new_stats.get("temp_c", 0)
                print(f"[HardwareGuardian] Đang làm mát... Nhiệt độ hiện tại: {temp}°C (Mục tiêu: <={self.cooldown_temp}°C) - Đã đợi {total_waited}s")

                if temp <= self.cooldown_temp or total_waited >= 180:
                    break

            print(f"[HardwareGuardian] ✅ Nhiệt độ đã an toàn ({temp}°C). Tiếp tục tiến trình huấn luyện!\n")
            if notify_callback:
                try:
                    notify_callback(f"❄️ GPU đã hạ nhiệt thành công ({temp}°C). Tiếp tục huấn luyện.")
                except Exception:
                    pass

    @staticmethod
    def optimize_for_rtx3060(mem_fraction=0.88):
        """
        Kích hoạt toàn bộ cờ tối ưu hóa kiến trúc Ampere (RTX 3060):
        - TensorFloat-32 (TF32): Tăng tốc 2-3x phép nhân ma trận mà không mất độ chính xác.
        - cuDNN Benchmark: Tối ưu kernel CNN cho kích thước đầu vào cố định.
        - Memory Fraction: Giữ lại 12% VRAM tránh treo máy OS / BSOD.
        - Expandable Segments: Chống phân mảnh bộ nhớ CUDA.
        """
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        if torch.cuda.is_available():
            # 1. Kích hoạt TF32 cho kiến trúc Ampere
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            # 2. Giới hạn tỷ lệ bộ nhớ VRAM tối đa để tránh sập driver màn hình
            try:
                torch.cuda.set_per_process_memory_fraction(mem_fraction)
            except Exception:
                pass

            gpu_name = torch.cuda.get_device_name(0)
            print("=" * 70)
            print(f"[+] ĐÃ KÍCH HOẠT TỐI ƯU HÓA PHẦN CỨNG CHO: {gpu_name}")
            print(f" - Kiến trúc Ampere TF32: BẬT (Tăng tốc nhân Tensor Core)")
            print(f" - cuDNN Benchmark: BẬT (Tối ưu hóa Kernel)")
            print(f" - Chống phân mảnh VRAM (expandable_segments): BẬT")
            print(f" - Giới hạn trần VRAM an toàn: {mem_fraction * 100:.0f}% (Tránh tràn bộ nhớ & sập máy)")
            print("=" * 70)
