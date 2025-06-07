from flask import Flask, request, jsonify, render_template
import threading
import time
from collections import defaultdict
import random
import logging
from datetime import datetime
import os
from prediction import ResourcePredictor
import numpy as np

app = Flask(__name__)

# Custom template filters
def datetime_format(value, format="%Y-%m-%d %H:%M:%S"):
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value).strftime(format)
        elif isinstance(value, datetime):
            return value.strftime(format)
        return str(value)
    except:
        return "N/A"

def duration_seconds(start_time, end_time=None):
    try:
        if not isinstance(start_time, datetime):
            return 0
        end = end_time if isinstance(end_time, datetime) else datetime.now()
        return (end - start_time).total_seconds()
    except:
        return 0

app.jinja_env.filters['datetime'] = datetime_format
app.jinja_env.filters['duration'] = duration_seconds
app.jinja_env.globals.update(min=min, max=max, zip=zip, enumerate=enumerate)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coordinator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Shared state
connected_nodes = {}
pending_processes = []
allocations = defaultdict(list)
completed_processes = []
lock = threading.Lock()
next_pid = 1
predictor = ResourcePredictor("models/")

class DummyProcess:
    def __init__(self, requirements):
        global next_pid
        self.pid = next_pid
        next_pid += 1
        self.original_requirements = requirements.copy()
        self.allocated_resources = {'cpu': 0, 'ram': 0, 'disk': 0, 'gpu': 0}
        self.status = "pending"
        self.assigned_to = []
        self.created_at = datetime.now()
        self.progress = 0
        self.completed_nodes = set()
        self.min_allocation = {
            'cpu': max(0.1, requirements['cpu'] * 0.1),
            'ram': max(0.1, requirements['ram'] * 0.1),
            'disk': max(0, requirements['disk'] * 0.1),
            'gpu': max(0, requirements['gpu'] * 0.1)
        }

def predict_node_performance(node_resources):
    """Predict performance metrics for a node using all models"""
    features = {
        'CPU_Usage (%)': 100 - (node_resources['resources']['cpu_cores'] / node_resources['original_resources']['cpu_cores'] * 100),
        'Memory_Usage (%)': 100 - (node_resources['resources']['ram_gb'] / node_resources['original_resources']['ram_gb'] * 100),
        'Disk_IO (MB/s)': 0,  # Placeholder
        'GPU_Usage (%)': 100 - node_resources['resources']['gpu_percent'],
        'Total_RAM (GB)': node_resources['original_resources']['ram_gb'],
        'Total_CPU_Power (GHz)': node_resources['original_resources']['cpu_cores'] * 2.5,
        'Total_Storage (GB)': node_resources['original_resources']['disk_gb'],
        'Total_GPU_Power (TFLOPS)': 10.0,
        'Active_Hours': 8,
        'Start_Hour': datetime.now().hour,
        'Day_of_Week': datetime.now().weekday(),
        'Is_Weekend': int(datetime.now().weekday() >= 5),
        'Month': datetime.now().month,
        'Usage_Pattern_Constant Load': 0,
        'Usage_Pattern_Idle': 1 if node_resources['resources']['cpu_cores'] > node_resources['original_resources']['cpu_cores'] * 0.8 else 0,
        'Usage_Pattern_Periodic Peaks': 0,
        'Operating_System_Linux': 1,
        'Operating_System_Windows': 0
    }
    
    prediction_result = predictor.predict_resources(features)
    
    performance = {
        'throughput': min(100, max(10, prediction_result['cpu_cores'] * 10)),
        'latency': max(1, (100 - prediction_result['cpu_cores']) / 10),
        'stability': min(100, max(50, prediction_result['ram_gb'] * 10)),
        'model_predictions': prediction_result['all_predictions'],
        'weighted_prediction': {
            'cpu_cores': prediction_result['cpu_cores'],
            'ram_gb': prediction_result['ram_gb'],
            'disk_gb': prediction_result['disk_gb'],
            'gpu_percent': prediction_result['gpu_percent']
        }
    }
    
    return performance

@app.route('/')
def dashboard():
    with lock:
        node_predictions = {}
        model_metrics = defaultdict(list)
        
        for node_id, node in connected_nodes.items():
            pred = predict_node_performance(node)
            node_predictions[node_id] = pred
            
            for model_name, model_pred in pred['model_predictions'].items():
                model_metrics[model_name].append({
                    'node': str(node_id),
                    'cpu': float(model_pred['cpu_cores']),
                    'ram': float(model_pred['ram_gb']),
                    'confidence': float(model_pred['confidence'])
                })

        
        return render_template('dashboard.html',
            nodes=connected_nodes,
            pending=pending_processes,
            allocated=allocations,
            completed=completed_processes,
            now=datetime.now(),
            predictions=node_predictions,
            model_metrics=dict(model_metrics)
        )

@app.route('/register', methods=['POST'])
def register_node():
    data = request.json
    with lock:
        connected_nodes[data['node_id']] = {
            'resources': data['available_resources'],
            'original_resources': data['original_resources'],
            'usage_metrics': data.get('usage_metrics', {}),
            'last_seen': time.time(),
            'ip': request.remote_addr
        }
        logger.info(f"Node registered: {data['node_id']}")
    return jsonify({"status": "registered"})

@app.route('/submit_process', methods=['POST'])
def submit_process():
    with lock:
        process = DummyProcess(request.json)
        pending_processes.append(process)
        logger.info(f"Process submitted: PID {process.pid}")
    return jsonify({"status": "queued", "pid": process.pid})

@app.route('/get_allocations', methods=['GET'])
def get_allocations():
    node_id = request.args.get('node_id')
    with lock:
        node_processes = []
        for process in allocations.get(node_id, []):
            node_processes.append({
                "pid": process.pid,
                "original_requirements": process.original_requirements,
                "allocated_resources": process.allocated_resources,
                "assigned_to": process.assigned_to
            })
        return jsonify({"processes": node_processes})

@app.route('/update_progress', methods=['POST'])
def update_progress():
    data = request.json
    with lock:
        target_process = None
        for node_id in allocations:
            for process in allocations[node_id]:
                if process.pid == data['pid']:
                    target_process = process
                    break
            if target_process:
                break

        if not target_process:
            return jsonify({"status": "process not found"}), 404

        if 'progress' in data and data['progress'] > target_process.progress:
            target_process.progress = data['progress']

        if data.get('node_complete', False):
            reporting_node = None
            for node_id in allocations:
                if target_process in allocations[node_id]:
                    reporting_node = node_id
                    break
            
            if reporting_node:
                target_process.completed_nodes.add(reporting_node)
                
                if reporting_node in connected_nodes:
                    node = connected_nodes[reporting_node]
                    total_nodes = len(target_process.assigned_to)
                    node['resources']['cpu_cores'] += target_process.allocated_resources['cpu'] / total_nodes
                    node['resources']['ram_gb'] += target_process.allocated_resources['ram'] / total_nodes
                
                if target_process in allocations[reporting_node]:
                    allocations[reporting_node].remove(target_process)

                if len(target_process.completed_nodes) == len(target_process.assigned_to):
                    target_process.status = "completed"
                    completed_processes.append(target_process)
                    logger.info(f"Process {target_process.pid} fully completed")

        return jsonify({"status": "updated"})

def try_allocate_more_resources(process):
    remaining_cpu = process.original_requirements['cpu'] - process.allocated_resources['cpu']
    remaining_ram = process.original_requirements['ram'] - process.allocated_resources['ram']
    
    if remaining_cpu <= 0 and remaining_ram <= 0:
        process.status = "fully_allocated"
        return True

    sorted_nodes = sorted(
        connected_nodes.items(),
        key=lambda x: (x[1]['resources']['cpu_cores'], x[1]['resources']['ram_gb']),
        reverse=True
    )

    for node_id, node in sorted_nodes:
        if node_id in process.assigned_to:
            continue

        alloc_cpu = min(node['resources']['cpu_cores'], remaining_cpu)
        alloc_ram = min(node['resources']['ram_gb'], remaining_ram)

        if alloc_cpu > 0 or alloc_ram > 0:
            process.assigned_to.append(node_id)
            process.allocated_resources['cpu'] += alloc_cpu
            process.allocated_resources['ram'] += alloc_ram
            node['resources']['cpu_cores'] -= alloc_cpu
            node['resources']['ram_gb'] -= alloc_ram
            
            logger.info(f"Added resources to PID {process.pid} from {node_id}")
            
            if (process.allocated_resources['cpu'] >= process.original_requirements['cpu'] and
                process.allocated_resources['ram'] >= process.original_requirements['ram']):
                process.status = "fully_allocated"
                return True
    return False

def allocate_new_processes():
    pending_processes.sort(
        key=lambda p: max(p.min_allocation['cpu'], p.min_allocation['ram'])
    )

    for process in pending_processes[:]:
        allocated_nodes = []
        remaining_cpu = process.original_requirements['cpu']
        remaining_ram = process.original_requirements['ram']

        sorted_nodes = sorted(
            connected_nodes.items(),
            key=lambda x: (x[1]['resources']['cpu_cores'], x[1]['resources']['ram_gb']),
            reverse=True
        )

        for node_id, node in sorted_nodes:
            if remaining_cpu <= 0 and remaining_ram <= 0:
                break

            alloc_cpu = min(node['resources']['cpu_cores'], remaining_cpu)
            alloc_ram = min(node['resources']['ram_gb'], remaining_ram)

            if ((process.allocated_resources['cpu'] + alloc_cpu >= process.min_allocation['cpu']) and
                (process.allocated_resources['ram'] + alloc_ram >= process.min_allocation['ram'])):
                
                allocated_nodes.append(node_id)
                process.allocated_resources['cpu'] += alloc_cpu
                process.allocated_resources['ram'] += alloc_ram
                remaining_cpu -= alloc_cpu
                remaining_ram -= alloc_ram

                node['resources']['cpu_cores'] -= alloc_cpu
                node['resources']['ram_gb'] -= alloc_ram

        if (process.allocated_resources['cpu'] >= process.min_allocation['cpu'] and
            process.allocated_resources['ram'] >= process.min_allocation['ram']):
            
            process.assigned_to = allocated_nodes
            process.status = "partially_allocated" if (remaining_cpu > 0 or remaining_ram > 0) else "fully_allocated"
            
            for node_id in allocated_nodes:
                allocations[node_id].append(process)
            
            pending_processes.remove(process)
            logger.info(
                f"Allocated PID {process.pid} to {len(allocated_nodes)} nodes | "
                f"CPU: {process.allocated_resources['cpu']:.1f}/{process.original_requirements['cpu']:.1f} cores | "
                f"RAM: {process.allocated_resources['ram']:.1f}/{process.original_requirements['ram']:.1f} GB"
            )

def allocate_resources():
    while True:
        time.sleep(5)
        with lock:
            for process in [p for p in pending_processes if p.status == "partially_allocated"]:
                try_allocate_more_resources(process)
            
            if pending_processes and connected_nodes:
                allocate_new_processes()

def cleanup_nodes():
    while True:
        time.sleep(60)
        with lock:
            stale = [nid for nid, node in connected_nodes.items() 
                    if time.time() - node['last_seen'] > 120]
            for nid in stale:
                logger.info(f"Removing stale node: {nid}")
                del connected_nodes[nid]

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    threading.Thread(target=allocate_resources, daemon=True).start()
    threading.Thread(target=cleanup_nodes, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True)