import requests
import time
import psutil
import socket
import logging
import threading
import random
import GPUtil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResourceClient:
    def __init__(self, coordinator_url):
        self.coordinator = coordinator_url.rstrip('/')
        self.node_id = f"{socket.gethostname()}-{psutil.Process().pid}"
        self.running_processes = {}
        
    def get_gpu_metrics(self):
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return {
                    'gpu_usage': gpus[0].load * 100,
                    'gpu_total': gpus[0].memoryTotal
                }
        except:
            return {'gpu_usage': 0, 'gpu_total': 0}
        
    def get_available_resources(self):
        cpu_percent = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        gpu = self.get_gpu_metrics()
        
        total_cores = psutil.cpu_count()
        total_ram = mem.total / (1024 ** 3)
        total_disk = disk.total / (1024 ** 3)
        total_gpu = gpu.get('gpu_total', 0)
        
        available_cores = max(0.1, total_cores * (1 - cpu_percent/100))
        available_ram = max(0.1, total_ram * (1 - mem.percent/100))
        available_disk = max(1, total_disk * (1 - disk.percent/100))
        available_gpu = max(5, 100 - gpu.get('gpu_usage', 0))
        
        return {
            'cpu_cores': available_cores,
            'ram_gb': available_ram,
            'disk_gb': available_disk,
            'gpu_percent': available_gpu,
            'original_resources': {
                'cpu_cores': total_cores,
                'ram_gb': total_ram,
                'disk_gb': total_disk,
                'gpu_percent': 100
            },
            'usage_metrics': {
                'cpu_percent': cpu_percent,
                'mem_percent': mem.percent,
                'disk_percent': disk.percent,
                'gpu_percent': gpu.get('gpu_usage', 0)
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def run_process(self, process):
        pid = process['pid']
        allocated = process['allocated_resources']
        requested = process['original_requirements']
        
        cpu_efficiency = allocated['cpu'] / requested['cpu']
        ram_efficiency = allocated['ram'] / requested['ram']
        efficiency = min(cpu_efficiency, ram_efficiency)
        
        logger.info(f"Starting PID {pid} with {efficiency*100:.1f}% resources")
        local_progress = 0
        
        while local_progress < 100:
            time.sleep(1 * (1/efficiency if efficiency > 0 else 1))
            local_progress = min(100, local_progress + random.randint(5, 15) * efficiency)
            
            try:
                response = requests.post(
                    f"{self.coordinator}/update_progress",
                    json={
                        'pid': pid,
                        'progress': local_progress,
                        'node_complete': local_progress >= 100
                    },
                    timeout=3
                )
                
                if local_progress >= 100:
                    logger.info(f"Process {pid} completed on this node")
                    break
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Progress update failed: {e}")
        
        logger.info(f"Process {pid} fully completed")
    
    def run(self):
        logger.info(f"Starting client node {self.node_id}")
        
        while True:
            try:
                resources = self.get_available_resources()
                requests.post(
                    f"{self.coordinator}/register",
                    json={
                        'node_id': self.node_id,
                        'available_resources': {
                            'cpu_cores': resources['cpu_cores'],
                            'ram_gb': resources['ram_gb'],
                            'disk_gb': resources['disk_gb'],
                            'gpu_percent': resources['gpu_percent']
                        },
                        'original_resources': resources['original_resources'],
                        'usage_metrics': resources['usage_metrics']
                    },
                    timeout=5
                )
                
                response = requests.get(
                    f"{self.coordinator}/get_allocations",
                    params={'node_id': self.node_id},
                    timeout=5
                )
                assigned_processes = response.json().get('processes', [])
                
                current_pids = {p['pid'] for p in assigned_processes}
                existing_pids = set(self.running_processes.keys())
                
                for pid in current_pids - existing_pids:
                    process = next(p for p in assigned_processes if p['pid'] == pid)
                    thread = threading.Thread(
                        target=self.run_process,
                        args=(process,),
                        daemon=True
                    )
                    thread.start()
                    self.running_processes[pid] = thread
                
                time.sleep(15)
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Connection error: {e}")
                time.sleep(30)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python client.py <coordinator_url>")
        sys.exit(1)
    
    client = ResourceClient(sys.argv[1])
    client.run()