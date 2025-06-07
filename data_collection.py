import psutil
import platform
from datetime import datetime
import GPUtil

def get_gpu_metrics():
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return {
                'GPU_Usage (%)': gpus[0].load * 100,
                'Total_GPU_Power (TFLOPS)': gpus[0].memoryTotal / 1000
            }
    except:
        pass
    return {'GPU_Usage (%)': 0, 'Total_GPU_Power (TFLOPS)': 0}

def collect_all_features(node_id):
    now = datetime.now()
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_io_counters()
    
    # Determine patterns
    usage_pattern = 'Idle' if cpu < 20 else 'Periodic Peaks' if cpu < 70 else 'Constant Load'
    hour = now.hour
    part_of_day = (
        'Night' if 22 <= hour <= 6 else
        'Morning' if hour <= 12 else
        'Afternoon' if hour <= 18 else
        'Evening'
    )

    return {
        # Core metrics
        'Active_Hours': 8,
        'Start_Hour': hour,
        'CPU_Usage (%)': cpu,
        'Memory_Usage (%)': mem.percent,
        'Disk_IO (MB/s)': (disk.read_bytes + disk.write_bytes) / (1024**2),
        'Total_RAM (GB)': mem.total / (1024**3),
        'Total_CPU_Power (GHz)': psutil.cpu_freq().max / 1000,
        'Total_Storage (GB)': psutil.disk_usage('/').total / (1024**3),
        'Day_of_Week': now.weekday(),
        'Is_Weekend': int(now.weekday() >= 5),
        'Month': now.month,
        
        # Usage patterns
        'Usage_Pattern_Constant Load': int(usage_pattern == 'Constant Load'),
        'Usage_Pattern_Idle': int(usage_pattern == 'Idle'),
        'Usage_Pattern_Periodic Peaks': int(usage_pattern == 'Periodic Peaks'),
        
        # OS features
        'Operating_System_Linux': int(platform.system() == 'Linux'),
        'Operating_System_Windows': int(platform.system() == 'Windows'),
        'Operating_System_macOS': int(platform.system() == 'Darwin'),
        
        # Time features
        'Part_of_Day_Night': int(part_of_day == 'Night'),
        'Part_of_Day_Morning': int(part_of_day == 'Morning'),
        'Part_of_Day_Afternoon': int(part_of_day == 'Afternoon'),
        'Part_of_Day_Evening': int(part_of_day == 'Evening'),
        
        # GPU metrics
        **get_gpu_metrics()
    }