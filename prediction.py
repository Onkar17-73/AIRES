import joblib
import numpy as np
import logging
from typing import Dict, Any, List
from keras.models import load_model
import tensorflow as tf
from collections import defaultdict
from tcn import TCN
from keras.saving import register_keras_serializable

class ResourcePredictor:
    def __init__(self, model_path: str):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.model_names = []
        self.feature_order = [
            'Active_Hours', 'Start_Hour', 'CPU_Usage (%)', 'Memory_Usage (%)',
            'Disk_IO (MB/s)', 'GPU_Usage (%)', 'Total_RAM (GB)', 'Total_CPU_Power (GHz)',
            'Total_Storage (GB)', 'Total_GPU_Power (TFLOPS)', 'Day_of_Week', 'Is_Weekend',
            'Month', 'Usage_Pattern_Constant Load', 'Usage_Pattern_Idle',
            'Usage_Pattern_Periodic Peaks', 'Operating_System_Linux',
            'Operating_System_Windows'
        ]
        
        try:
            # Load non-Keras models first
            self.models = {
                'XGBoost': joblib.load(f"{model_path}/multi_target_resource_predictor.pkl"),
                'CatBoost': joblib.load(f"{model_path}/multioutput_cat_model.pkl"),
                'LightGBM': joblib.load(f"{model_path}/multioutput_lgb_model.pkl")
            }
            
            # Custom MAE function for Keras models
            def mae(y_true, y_pred):
                from keras import backend as K
                return K.mean(K.abs(y_pred - y_true))
            
            # Load Keras models with custom objects
            self.models['LSTM'] = load_model(
                f"{model_path}/deep_lstm_resource_predictor.h5",
                custom_objects={'mae': mae}
            )
            self.models['TCN'] = load_model(
                f"{model_path}/tcn_multi_target_model.keras",
                custom_objects={'mae': mae}
            )
            
            self.model_names = list(self.models.keys())
            self.logger.info("All models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

    def _prepare_input(self, features: Dict[str, Any]) -> np.ndarray:
        return np.array([[features[f] for f in self.feature_order]])

    def _predict_with_model(self, model_name: str, input_array: np.ndarray) -> Dict[str, Any]:
        try:
            model = self.models[model_name]
            
            if model_name in ['LSTM', 'TCN']:
                input_array = input_array.reshape(1, 1, len(self.feature_order))
                prediction = model.predict(input_array)[0]
                confidence = float(np.min(model.predict_proba(input_array)[0])) * 100
            else:
                prediction = model.predict(input_array)[0]
                try:
                    confidence = float(np.max(model.predict_proba(input_array)[0])) * 100
                except AttributeError:
                    confidence = 85.0
            
            return {
                'cpu_cores': max(0.1, prediction[0]),
                'ram_gb': max(0.1, prediction[1]),
                'disk_gb': max(1, prediction[2]),
                'gpu_percent': max(5, 100 - prediction[3]),
                'confidence': confidence
            }
        except Exception as e:
            self.logger.error(f"Prediction failed with {model_name}: {e}")
            return None

    def predict_resources(self, features: Dict[str, Any]) -> Dict[str, Any]:
        input_array = self._prepare_input(features)
        all_predictions = {}
        
        for model_name in self.model_names:
            prediction = self._predict_with_model(model_name, input_array)
            if prediction:
                all_predictions[model_name] = prediction
        
        if not all_predictions:
            total_ram = features['Total_RAM (GB)']
            total_cpu = features['Total_CPU_Power (GHz)']
            fallback = {
                'cpu_cores': max(0.1, total_cpu * (1 - features['CPU_Usage (%)']/100)),
                'ram_gb': max(0.1, total_ram * (1 - features['Memory_Usage (%)']/100)),
                'disk_gb': max(1, features['Total_Storage (GB)'] * 0.8),
                'gpu_percent': max(5, 100 - features['GPU_Usage (%)']),
                'confidence': 70.0
            }
            all_predictions['Fallback'] = fallback
        
        weighted = defaultdict(float)
        total_weight = 0.0
        
        for model_name, pred in all_predictions.items():
            weight = pred['confidence'] / 100.0
            weighted['cpu_cores'] += pred['cpu_cores'] * weight
            weighted['ram_gb'] += pred['ram_gb'] * weight
            weighted['disk_gb'] += pred['disk_gb'] * weight
            weighted['gpu_percent'] += pred['gpu_percent'] * weight
            total_weight += weight
        
        averaged_prediction = {
            'cpu_cores': weighted['cpu_cores'] / total_weight,
            'ram_gb': weighted['ram_gb'] / total_weight,
            'disk_gb': weighted['disk_gb'] / total_weight,
            'gpu_percent': weighted['gpu_percent'] / total_weight,
            'all_predictions': all_predictions
        }
        
        return averaged_prediction