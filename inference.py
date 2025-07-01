"""
Configurable EMG Model Inference System
=======================================
Allows user to specify:
- Number of additional classes (1-5, thinking always included)
- Number of channels (1-4)
- Auto-adapts inference pipeline to match configuration
- Perfect feature alignment with training

Features:
- User configurable classes and channels
- Automatic thinking class inclusion
- Adaptive feature enhancement pipeline
- Real-time continuous prediction
- Comprehensive evaluation metrics
"""

import os
import csv
import sys
import json
import time
import random
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
import serial
import joblib
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import threading
import queue
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ConfigurableEMGInference:
    def __init__(self, models_dir="adaptive_models", serial_port='/dev/cu.usbserial-110', baud_rate=115200):
        """
        Configurable EMG Subvocal Detection Inference System
        
        User specifies:
        - Number of additional classes (1-5, thinking always included)
        - Number of channels (1-4)
        """
        self.models_dir = models_dir
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.available_models = {}
        self.selected_model = None
        self.model_data = None
        self.ser = None
        
        # User-configurable parameters (will be set by user)
        self.num_classes = None
        self.num_channels = None
        self.class_names = []
        self.feature_names = []
        self.total_features = None
        self.window_size = 500
        
        # Testing parameters
        self.test_duration = 60
        
        # Data collection and enhancement (dynamic based on channels)
        self.emg_buffer = deque(maxlen=1000)
        self.enhanced_buffer = deque(maxlen=1000)
        self.predictions = []
        self.actual_labels = []
        self.prediction_times = []
        self.is_collecting = False
        self.data_queue = queue.Queue()
        
        # Feature computation history (dynamic based on channels)
        self.channel_histories = {}
        
        # GUI elements
        self.root = None
        self.command_label = None
        self.prediction_label = None
        self.accuracy_label = None
        self.status_label = None
        self.progress_var = None
        self.time_label = None
        self.confidence_label = None
        
        # Statistics
        self.correct_predictions = 0
        self.total_predictions = 0
        self.start_time = None
        self.last_prediction_time = 0
    
    def setup_user_configuration(self):
        """Interactive setup for inference configuration"""
        print("🎯 EMG Subvocal Detection Configuration")
        print("=" * 50)
        print("Note: 'thinking' class is automatically included as the rest state")
        
        # Get number of additional classes (thinking is always included)
        while True:
            try:
                num_additional_classes = int(input("How many additional subvocal classes? (1-5, 'thinking' is always included): "))
                if 1 <= num_additional_classes <= 5:
                    break
                else:
                    print("❌ Please enter a number between 1 and 5")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Get class names (thinking is always included)
        print(f"\n'thinking' class is automatically included as the rest state.")
        print(f"Enter names for {num_additional_classes} additional subvocal classes:")
        self.class_names = []
        for i in range(num_additional_classes):
            while True:
                class_name = input(f"Subvocal class {i+1} name (e.g., 'hello', 'yes', 'no'): ").strip().lower()
                if class_name and class_name not in self.class_names and class_name != 'thinking':
                    self.class_names.append(class_name)
                    break
                elif class_name == 'thinking':
                    print("❌ 'thinking' is already included automatically")
                elif class_name in self.class_names:
                    print("❌ Class name already used, please choose a different name")
                else:
                    print("❌ Please enter a valid class name")
        
        # Add thinking class to the list (always last for consistency)
        self.class_names.append('thinking')
        self.num_classes = len(self.class_names)
        
        # Get number of channels
        while True:
            try:
                self.num_channels = int(input(f"\nHow many EMG channels? (1-4): "))
                if 1 <= self.num_channels <= 4:
                    break
                else:
                    print("❌ Please enter a number between 1 and 4")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Setup feature names and channel histories based on configuration
        self.setup_feature_configuration()
        
        # Show configuration summary
        self.show_configuration_summary()
        
        return True
    
    def setup_feature_configuration(self):
        """Setup feature names and channel histories based on user configuration"""
        # Generate feature names based on number of channels
        self.feature_names = []
        for ch in range(1, self.num_channels + 1):
            ch_features = [
                f'ch{ch}_filtered', f'ch{ch}_envelope', 
                f'ch{ch}_filtered_rms', f'ch{ch}_filtered_movavg',
                f'ch{ch}_envelope_rms', f'ch{ch}_envelope_movavg'
            ]
            self.feature_names.extend(ch_features)
        
        self.total_features = len(self.feature_names)
        
        # Setup channel histories
        self.channel_histories = {}
        for ch in range(1, self.num_channels + 1):
            self.channel_histories[f'ch{ch}_filtered'] = deque(maxlen=100)
            self.channel_histories[f'ch{ch}_envelope'] = deque(maxlen=100)
    
    def show_configuration_summary(self):
        """Display the configuration summary"""
        print("\n" + "=" * 50)
        print("📋 SUBVOCAL DETECTION CONFIGURATION")
        print("=" * 50)
        
        active_classes = [name for name in self.class_names if name != 'thinking']
        
        print(f"🎯 Classes: {', '.join([name.upper() for name in self.class_names])}")
        print(f"   • Subvocal classes: {', '.join([name.upper() for name in active_classes])}")
        print(f"   • Rest state: THINKING (automatic)")
        print(f"📡 Channels: {self.num_channels}")
        print(f"🔧 Features: {self.total_features} total ({self.total_features//self.num_channels} per channel)")
        print(f"📊 Expected ESP32 input: {self.num_channels * 2} raw values")
        
        esp32_format = ','.join([f'ch{i+1}_filtered,ch{i+1}_envelope' for i in range(self.num_channels)])
        print(f"📡 Expected ESP32 format: {esp32_format}")
        
        print(f"\n⚙️  Inference Pipeline:")
        print(f"   ESP32 → {self.num_channels * 2} raw values")
        print(f"   Python → {self.total_features} enhanced features")
        print(f"   Model → {self.num_classes} class prediction")
        
        confirm = input(f"\nProceed with this configuration? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Configuration cancelled. Restarting setup...")
            return self.setup_user_configuration()
    
    def compute_rms(self, signal, window_size=20):
        """Compute RMS exactly like data collection"""
        if len(signal) < window_size:
            return np.sqrt(np.mean(np.square(signal))) if len(signal) > 0 else 0
        return np.sqrt(np.mean(np.square(signal[-window_size:])))
    
    def moving_average(self, signal, window_size=20):
        """Compute moving average exactly like data collection"""
        if len(signal) < window_size:
            return np.mean(signal) if len(signal) > 0 else 0
        return np.mean(signal[-window_size:])
    
    def enhance_raw_features(self, raw_emg_data):
        """
        Convert raw ESP32 values → enhanced features for configured number of channels
        
        Input: [ch1_filtered, ch1_envelope, ch2_filtered, ch2_envelope, ...] (2*num_channels values)
        Output: All enhanced features matching configuration
        """
        if len(raw_emg_data) != self.num_channels * 2:
            raise ValueError(f"Expected {self.num_channels * 2} raw values, got {len(raw_emg_data)}")
        
        enhanced_features = []
        
        # Process each channel
        for ch in range(1, self.num_channels + 1):
            # Extract raw values for this channel
            ch_filtered = raw_emg_data[(ch-1) * 2]
            ch_envelope = raw_emg_data[(ch-1) * 2 + 1]
            
            # Add to history buffers
            self.channel_histories[f'ch{ch}_filtered'].append(ch_filtered)
            self.channel_histories[f'ch{ch}_envelope'].append(ch_envelope)
            
            # Compute enhanced features exactly like data collection
            ch_filtered_rms = self.compute_rms(list(self.channel_histories[f'ch{ch}_filtered']))
            ch_filtered_movavg = self.moving_average(list(self.channel_histories[f'ch{ch}_filtered']))
            ch_envelope_rms = self.compute_rms(list(self.channel_histories[f'ch{ch}_envelope']))
            ch_envelope_movavg = self.moving_average(list(self.channel_histories[f'ch{ch}_envelope']))
            
            # Add all 6 features for this channel (same order as data collection)
            enhanced_features.extend([
                ch_filtered,        # ch{n}_filtered
                ch_envelope,        # ch{n}_envelope
                ch_filtered_rms,    # ch{n}_filtered_rms
                ch_filtered_movavg, # ch{n}_filtered_movavg
                ch_envelope_rms,    # ch{n}_envelope_rms
                ch_envelope_movavg  # ch{n}_envelope_movavg
            ])
        
        return enhanced_features
    
    def extract_statistical_features(self, window_data):
        """
        Extract statistical features from enhanced data for configured number of channels
        
        Input: (window_size, total_features) window data
        Output: (1, total_features * 10) statistical features
        """
        features = []
        
        # For each enhanced feature
        for i in range(window_data.shape[1]):
            channel_data = window_data[:, i]
            feature_name = self.feature_names[i]
            
            # Extract same 10 statistical features as training
            channel_features = [
                np.mean(channel_data),                           # mean
                np.std(channel_data),                            # std
                np.min(channel_data),                            # min
                np.max(channel_data),                            # max
                np.median(channel_data),                         # median
                np.percentile(channel_data, 25),                 # p25
                np.percentile(channel_data, 75),                 # p75
                np.sum(channel_data**2),                         # energy
                np.sum(channel_data**2) / len(channel_data),     # power
                # Zero crossings (only for filtered channels)
                np.sum(np.diff(np.sign(channel_data)) != 0) if 'filtered' in feature_name else 0
            ]
            
            features.extend(channel_features)
        
        # Should give us total_features × 10 statistical features
        result = np.array(features).reshape(1, -1)
        return result
    
    def discover_models(self):
        """Discover available trained models with configuration matching"""
        print("🔍 Discovering compatible models...")
        
        if not os.path.exists(self.models_dir):
            print(f"❌ Models directory not found: {self.models_dir}")
            return False
        
        self.available_models = {}
        compatible_count = 0
        
        for item in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, item)
            if os.path.isdir(model_path):
                summary_path = os.path.join(model_path, 'model_summary.json')
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, 'r') as f:
                            summary = json.load(f)
                        
                        # Extract configuration from model summary
                        config = summary.get('adaptive_configuration', {})
                        model_classes = config.get('num_classes', 'Unknown')
                        model_channels = config.get('num_channels', 'Unknown')
                        
                        # Check compatibility
                        is_compatible = (model_classes == self.num_classes and 
                                       model_channels == self.num_channels)
                        
                        self.available_models[item] = {
                            'name': item,
                            'type': summary['model_type'],
                            'accuracy': summary['accuracy'],
                            'timestamp': summary['timestamp'],
                            'path': model_path,
                            'summary': summary,
                            'num_classes': model_classes,
                            'num_channels': model_channels,
                            'class_names': config.get('class_names', []),
                            'feature_names': config.get('feature_names', []),
                            'compatible': is_compatible
                        }
                        
                        if is_compatible:
                            compatible_count += 1
                        
                    except Exception as e:
                        print(f"⚠️  Could not read model {item}: {e}")
        
        print(f"✅ Found {len(self.available_models)} total models")
        print(f"✅ Found {compatible_count} compatible models (matching {self.num_classes} classes, {self.num_channels} channels)")
        
        if compatible_count == 0:
            print(f"❌ No compatible models found for your configuration!")
            print(f"   Looking for: {self.num_classes} classes, {self.num_channels} channels")
            print(f"   Available models have different configurations.")
            return False
        
        return True
    
    def setup_threshold_model(self):
        """Setup hardcoded threshold-based model"""
        print("🎯 Threshold-Based Model Configuration")
        print("=" * 50)
        print("This creates a simple threshold-based classifier using signal amplitude.")
        print("You can set thresholds for different features to classify subvocal actions.")
        
        # Show available features for threshold
        print(f"\nAvailable features for thresholding:")
        for i, feature in enumerate(self.feature_names, 1):
            print(f"  {i}. {feature}")
        
        # Select feature for thresholding
        while True:
            try:
                feature_idx = int(input(f"\nSelect feature for threshold (1-{len(self.feature_names)}): ")) - 1
                if 0 <= feature_idx < len(self.feature_names):
                    selected_feature = self.feature_names[feature_idx]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(self.feature_names)}")
            except ValueError:
                print("❌ Please enter a valid number")
        
        print(f"\n✅ Selected feature: {selected_feature}")
        
        # Get threshold values for each class
        active_classes = [name for name in self.class_names if name != 'thinking']
        
        print(f"\nFor {len(self.class_names)}-class classification:")
        print(f"Classes: {', '.join(self.class_names)}")
        print(f"\nThreshold Logic:")
        print(f"- Below threshold → {self.class_names[-1]} (thinking/rest)")
        print(f"- Above threshold → Active classes based on amplitude levels")
        
        thresholds = {}
        
        if len(active_classes) == 1:
            # Binary classification: thinking vs one active class
            while True:
                try:
                    threshold = float(input(f"\nEnter threshold value (signal above = {active_classes[0]}, below = thinking): "))
                    thresholds = {
                        'feature': selected_feature,
                        'binary_threshold': threshold,
                        'high_class': active_classes[0],
                        'low_class': 'thinking'
                    }
                    break
                except ValueError:
                    print("❌ Please enter a valid number")
        
        else:
            # Multi-class: need multiple thresholds
            print(f"\nFor multi-class, enter thresholds in ascending order:")
            sorted_classes = ['thinking'] + sorted(active_classes)
            
            threshold_values = []
            for i in range(len(active_classes)):
                while True:
                    try:
                        if i == 0:
                            threshold = float(input(f"Threshold {i+1} (above = active classes, below = thinking): "))
                        else:
                            threshold = float(input(f"Threshold {i+1} (higher amplitude classes): "))
                        
                        if not threshold_values or threshold > threshold_values[-1]:
                            threshold_values.append(threshold)
                            break
                        else:
                            print(f"❌ Threshold must be higher than previous ({threshold_values[-1]})")
                    except ValueError:
                        print("❌ Please enter a valid number")
            
            thresholds = {
                'feature': selected_feature,
                'multi_thresholds': threshold_values,
                'classes': sorted_classes
            }
        
        # Create threshold model data
        self.model_data = {
            'model_type': 'threshold',
            'thresholds': thresholds,
            'feature_names': self.feature_names,
            'window_size': self.window_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
        
        # Show configuration summary
        print(f"\n📋 Threshold Model Configuration:")
        print(f"Feature: {selected_feature}")
        if 'binary_threshold' in thresholds:
            print(f"Threshold: {thresholds['binary_threshold']}")
            print(f"Logic: < {thresholds['binary_threshold']} = {thresholds['low_class']}")
            print(f"       >= {thresholds['binary_threshold']} = {thresholds['high_class']}")
        else:
            print(f"Thresholds: {thresholds['multi_thresholds']}")
            for i, (thresh, cls) in enumerate(zip(thresholds['multi_thresholds'], thresholds['classes'][1:])):
                if i == 0:
                    print(f"< {thresh} = thinking")
                print(f">= {thresh} = {cls}")
        
        return True
    
    def predict_threshold_model(self, window_data):
        """Make prediction using threshold-based model"""
        try:
            thresholds = self.model_data['thresholds']
            feature_name = thresholds['feature']
            
            # Find feature index
            feature_idx = self.feature_names.index(feature_name)
            
            # Get feature data from window
            feature_data = window_data[:, feature_idx]
            
            # Calculate amplitude metric (you can use different metrics)
            amplitude_metrics = {
                'mean': np.mean(np.abs(feature_data)),
                'max': np.max(np.abs(feature_data)),
                'rms': np.sqrt(np.mean(feature_data**2)),
                'std': np.std(feature_data)
            }
            
            # Use RMS as default amplitude measure
            amplitude = amplitude_metrics['rms']
            
            # Apply threshold logic
            if 'binary_threshold' in thresholds:
                # Binary classification
                if amplitude >= thresholds['binary_threshold']:
                    predicted_label = thresholds['high_class']
                    confidence = min(0.99, 0.5 + (amplitude - thresholds['binary_threshold']) / thresholds['binary_threshold'])
                else:
                    predicted_label = thresholds['low_class']
                    confidence = min(0.99, 0.5 + (thresholds['binary_threshold'] - amplitude) / thresholds['binary_threshold'])
            
            else:
                # Multi-class classification
                threshold_values = thresholds['multi_thresholds']
                classes = thresholds['classes']
                
                predicted_label = classes[0]  # Default to thinking
                for i, thresh in enumerate(threshold_values):
                    if amplitude >= thresh:
                        predicted_label = classes[i + 1]
                
                # Calculate confidence based on distance from thresholds
                confidence = 0.7  # Default confidence for threshold models
            
            return predicted_label, confidence, amplitude_metrics
            
        except Exception as e:
            print(f"Threshold prediction error: {e}")
            return None, None, None
    
    def select_model_interactive(self):
        """Interactive model selection with threshold option"""
        # Filter to only show compatible models
        compatible_models = {name: info for name, info in self.available_models.items() 
                           if info['compatible']}
        
        print(f"\n📋 Model Selection ({self.num_classes} classes, {self.num_channels} channels):")
        print("=" * 80)
        
        model_list = list(compatible_models.items())
        
        # Show trained models
        if model_list:
            print("🤖 Trained Models:")
            for i, (model_name, model_info) in enumerate(model_list, 1):
                classes_str = ', '.join(model_info['class_names']) if model_info['class_names'] else 'Unknown'
                print(f"  {i}. {model_name}")
                print(f"     Type: {model_info['type'].title()}")
                print(f"     Accuracy: {model_info['accuracy']:.3f}")
                print(f"     Classes: {classes_str}")
                print(f"     Created: {model_info['timestamp'][:19]}")
                print()
        else:
            print("❌ No compatible trained models found")
        
        # Add threshold option
        threshold_option = len(model_list) + 1
        print(f"🎯 Hardcoded Models:")
        print(f"  {threshold_option}. Threshold-Based Model (Hardcoded)")
        print(f"     Type: Simple amplitude threshold")
        print(f"     Setup: Configure threshold values manually")
        print(f"     Use: Good for clear amplitude differences")
        print()
        
        total_options = threshold_option
        
        while True:
            try:
                choice = input(f"Select model (1-{total_options}): ").strip()
                choice_idx = int(choice) - 1
                
                if choice_idx == threshold_option - 1:
                    # User selected threshold model
                    self.selected_model = "Threshold_Model"
                    print(f"✅ Selected: Threshold-Based Model")
                    
                    # Setup threshold model
                    if self.setup_threshold_model():
                        return True
                    else:
                        continue
                
                elif 0 <= choice_idx < len(model_list):
                    # User selected trained model
                    selected_name = model_list[choice_idx][0]
                    self.selected_model = selected_name
                    
                    print(f"✅ Selected: {selected_name}")
                    print(f"   Configuration: {self.num_classes} classes, {self.num_channels} channels")
                    
                    return True
                else:
                    print(f"❌ Invalid choice. Please enter 1-{total_options}")
                    
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Cancelled by user")
                return False
    
    def load_selected_model(self):
        """Load the selected model with threshold support"""
        if not self.selected_model:
            print("❌ No model selected")
            return False
        
        # Handle threshold model
        if self.selected_model == "Threshold_Model":
            print(f"✅ Threshold model configured and ready")
            return True
        
        # Handle trained models
        print(f"📦 Loading model: {self.selected_model}")
        
        model_info = self.available_models[self.selected_model]
        model_path = model_info['path']
        
        try:
            if model_info['type'] == 'classical':
                # Load classical model - fix the filename construction
                model_name_prefix = self.selected_model.split('_')[0].lower()  # e.g., "random" from "Random_Forest_2class_2ch"
                
                # Try different possible filenames
                possible_files = [
                    os.path.join(model_path, 'model', f'{model_name_prefix}_forest.joblib'),
                    os.path.join(model_path, 'model', f'{self.selected_model.lower()}.joblib'),
                    os.path.join(model_path, 'model', f'{model_name_prefix}.joblib'),
                    os.path.join(model_path, 'model', f'random_forest.joblib'),
                    os.path.join(model_path, 'model', f'gradient_boosting.joblib'),
                    os.path.join(model_path, 'model', f'svm.joblib'),
                    os.path.join(model_path, 'model', f'logistic_regression.joblib')
                ]
                
                # Find the actual file
                model_file = None
                for possible_file in possible_files:
                    if os.path.exists(possible_file):
                        model_file = possible_file
                        break
                
                if model_file is None:
                    # List actual files in the model directory to help debug
                    model_dir = os.path.join(model_path, 'model')
                    if os.path.exists(model_dir):
                        actual_files = os.listdir(model_dir)
                        print(f"❌ Model file not found. Available files: {actual_files}")
                        print(f"   Tried: {[os.path.basename(f) for f in possible_files]}")
                    else:
                        print(f"❌ Model directory not found: {model_dir}")
                    return False
                
                print(f"📦 Loading from: {model_file}")
                self.model_data = joblib.load(model_file)
                
                # Get configuration from model
                self.window_size = self.model_data['window_size']
                
                print(f"✅ Loaded classical model: {self.selected_model}")
                print(f"   Window size: {self.window_size}")
                print(f"   Expected features: {self.total_features * 10} (statistical)")
                print(f"   Classes: {self.model_data['label_encoder'].classes_}")
                
            elif model_info['type'] == 'deep_learning':
                # Load LSTM model
                model_file = os.path.join(model_path, 'model', 'lstm_model.h5')
                preprocessing_file = os.path.join(model_path, 'model', 'preprocessing_params.json')
                
                model = tf.keras.models.load_model(model_file)
                
                with open(preprocessing_file, 'r') as f:
                    preprocessing_params = json.load(f)
                
                self.model_data = {
                    'model': model,
                    'preprocessing_params': preprocessing_params,
                    'model_type': 'deep_learning'
                }
                
                self.window_size = preprocessing_params['window_size']
                
                print(f"✅ Loaded LSTM model: {self.selected_model}")
                print(f"   Window size: {self.window_size}")
                print(f"   Expected input: ({self.window_size}, {self.total_features}) - temporal")
                print(f"   Classes: {preprocessing_params['label_encoder_classes']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def setup_test_parameters(self):
        """Setup test parameters"""
        print(f"\n⚙️  Test Configuration:")
        print("-" * 40)
        
        while True:
            try:
                duration = input("Test duration in seconds (30-120, default 60): ").strip()
                if not duration:
                    self.test_duration = 60
                    break
                
                duration = int(duration)
                if 30 <= duration <= 120:
                    self.test_duration = duration
                    break
                else:
                    print("❌ Duration must be between 30-120 seconds")
                    
            except ValueError:
                print("❌ Please enter a valid number")
        
        print(f"✅ Test duration: {self.test_duration} seconds")
        
        print(f"\n📋 Complete Test Configuration:")
        print(f"   Model: {self.selected_model}")
        print(f"   Duration: {self.test_duration} seconds")
        print(f"   Window size: {self.window_size} samples")
        print(f"   Classes: {', '.join([name.upper() for name in self.class_names])}")
        print(f"   Channels: {self.num_channels}")
        print(f"   ESP32 sends: {self.num_channels * 2} raw features")
        print(f"   Python enhances to: {self.total_features} features")
        print(f"   Prediction mode: Continuous (every 200ms)")
        
        esp32_format = ','.join([f'ch{i+1}_filtered,ch{i+1}_envelope' for i in range(self.num_channels)])
        confirm = input(f"\nESP32 format: {esp32_format}. Continue? (y/n): ").strip().lower()
        return confirm in ['y', 'yes']
    
    def connect_serial(self):
        """Connect to EMG device with configuration validation"""
        print(f"🔌 Connecting to {self.serial_port}...")
        
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            time.sleep(2)
            
            # Test the connection and validate format
            print("Testing configurable data format and feature enhancement...")
            for i in range(5):
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line and ',' in line:
                        parts = line.split(',')
                        print(f"  Sample {i+1}: {line} ({len(parts)} values)")
                        
                        expected_values = self.num_channels * 2
                        if len(parts) == expected_values:
                            print(f"  ✅ Correct ESP32 format: {expected_values} values for {self.num_channels} channels")
                            # Test enhancement
                            try:
                                raw_values = [float(parts[j]) for j in range(expected_values)]
                                enhanced = self.enhance_raw_features(raw_values)
                                print(f"  ✅ Enhanced to: {len(enhanced)} features")
                                print(f"  📊 Feature preview: {[f'{x:.2f}' for x in enhanced[:6]]}...")
                            except Exception as e:
                                print(f"  ⚠️  Enhancement test failed: {e}")
                        else:
                            print(f"  ⚠️  Expected {expected_values} values, got {len(parts)}")
                time.sleep(0.2)
            
            print("✅ Serial connection established and validated")
            return True
            
        except Exception as e:
            print(f"❌ Serial connection failed: {e}")
            return False
    
    def create_gui(self):
        """Create configurable testing GUI"""
        self.root = tk.Tk()
        self.root.title(f"Subvocal Detection - {self.selected_model}")
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)
        
        # Main container
        main_frame = tk.Frame(self.root, bg="black")
        main_frame.pack(fill="both", expand=True)
        
        # Top section - Model info
        info_frame = tk.Frame(main_frame, bg="black")
        info_frame.pack(fill="x", padx=20, pady=10)
        
        active_classes = [name for name in self.class_names if name != 'thinking']
        classes_str = '/'.join([name.upper() for name in active_classes]) + '/THINKING'
        model_info_text = (f"Model: {self.selected_model} | Duration: {self.test_duration}s | "
                          f"Classes: {classes_str} | Channels: {self.num_channels} | "
                          f"Features: {self.num_channels*2}→{self.total_features}")
        
        model_info_label = tk.Label(info_frame, text=model_info_text,
                                   font=("Arial", 14), fg="yellow", bg="black")
        model_info_label.pack()
        
        # Exit button
        exit_btn = tk.Button(info_frame, text="Exit (ESC)", command=self.stop_test,
                           bg="red", fg="white", font=("Arial", 12))
        exit_btn.pack(side="right", padx=10)
        
        # Center section - Command display
        center_frame = tk.Frame(main_frame, bg="black")
        center_frame.pack(fill="both", expand=True)
        
        # Current state display
        tk.Label(center_frame, text="CURRENT STATE", font=("Arial", 16),
                fg="gray", bg="black").pack(pady=(20, 5))
        
        self.command_label = tk.Label(center_frame, text="READY",
                                     font=("Arial", 70), fg="white", bg="black")
        self.command_label.pack(pady=10)
        
        # Prediction display
        prediction_frame = tk.Frame(center_frame, bg="black")
        prediction_frame.pack(pady=15)
        
        tk.Label(prediction_frame, text="MODEL PREDICTION:",
                font=("Arial", 18), fg="cyan", bg="black").pack()
        
        self.prediction_label = tk.Label(prediction_frame, text="---",
                                        font=("Arial", 50), fg="lime", bg="black")
        self.prediction_label.pack()
        
        # Confidence display
        self.confidence_label = tk.Label(prediction_frame, text="Confidence: ---%",
                                       font=("Arial", 14), fg="yellow", bg="black")
        self.confidence_label.pack(pady=5)
        
        # Bottom section - Status and progress
        bottom_frame = tk.Frame(main_frame, bg="black")
        bottom_frame.pack(fill="x", padx=20, pady=15)
        
        # Accuracy display
        accuracy_frame = tk.Frame(bottom_frame, bg="black")
        accuracy_frame.pack(pady=8)
        
        self.accuracy_label = tk.Label(accuracy_frame, text="Accuracy: 0.0% (0/0)",
                                      font=("Arial", 20), fg="white", bg="black")
        self.accuracy_label.pack()
        
        # Correctness indicator
        self.status_label = tk.Label(accuracy_frame, text="Starting...",
                                    font=("Arial", 16), fg="cyan", bg="black")
        self.status_label.pack(pady=5)
        
        # Progress bar
        progress_frame = tk.Frame(bottom_frame, bg="black")
        progress_frame.pack(fill="x", pady=8)
        
        tk.Label(progress_frame, text="Progress:",
                font=("Arial", 14), fg="white", bg="black").pack()
        
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                     maximum=100, length=800)
        progress_bar.pack(pady=3)
        
        # Time remaining
        self.time_label = tk.Label(progress_frame, text="Time remaining: --:--",
                                  font=("Arial", 14), fg="white", bg="black")
        self.time_label.pack()
        
        # Key bindings
        self.root.bind('<Escape>', lambda e: self.stop_test())
        self.root.bind('<space>', lambda e: self.stop_test())
        self.root.focus_set()
        
        return True
    
    def predict_from_buffer(self):
        """Make prediction from current enhanced EMG buffer with threshold support"""
        if len(self.enhanced_buffer) < self.window_size:
            return None, None
        
        # Get latest window (enhanced data)
        window_data = np.array(list(self.enhanced_buffer)[-self.window_size:])
        # Shape should be (window_size, total_features)
        
        try:
            # Handle threshold model
            if self.selected_model == "Threshold_Model":
                predicted_label, confidence, amplitude_metrics = self.predict_threshold_model(window_data)
                if predicted_label is None:
                    return None, None
                
                # Store amplitude info for debugging
                self.last_amplitude_metrics = amplitude_metrics
                
                return predicted_label, confidence
            
            # Handle trained models
            if self.available_models[self.selected_model]['type'] == 'deep_learning':
                # LSTM prediction - use enhanced temporal data
                model = self.model_data['model']
                preprocessing_params = self.model_data['preprocessing_params']
                
                # Normalize data
                X_mean = np.array(preprocessing_params['X_mean'])
                X_std = np.array(preprocessing_params['X_std'])
                window_normalized = (window_data - X_mean) / X_std
                
                # Reshape for LSTM: (1, window_size, n_features)
                window_reshaped = window_normalized.reshape(1, self.window_size, self.total_features)
                
                # Predict
                prediction_probs = model.predict(window_reshaped, verbose=0)[0]
                prediction_idx = np.argmax(prediction_probs)
                confidence = prediction_probs[prediction_idx]
                
                # Decode label
                label_classes = preprocessing_params['label_encoder_classes']
                predicted_label = label_classes[prediction_idx]
                
            else:
                # Classical model prediction - extract statistical features
                model = self.model_data['model']
                scaler = self.model_data.get('scaler', None)
                label_encoder = self.model_data['label_encoder']
                
                # Extract statistical features
                features = self.extract_statistical_features(window_data)
                
                # Scale if needed
                if scaler:
                    features = scaler.transform(features)
                
                # Predict
                prediction_idx = model.predict(features)[0]
                
                if hasattr(model, 'predict_proba'):
                    prediction_probs = model.predict_proba(features)[0]
                    confidence = prediction_probs[prediction_idx]
                else:
                    confidence = 1.0
                
                # Decode label
                predicted_label = label_encoder.inverse_transform([prediction_idx])[0]
            
            return predicted_label, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            print(traceback.format_exc())
            return None, None
    
    def data_collection_thread(self):
        """Background thread for configurable EMG data collection"""
        while self.is_collecting:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line and ',' in line:
                        parts = line.split(',')
                        expected_values = self.num_channels * 2
                        
                        if len(parts) >= expected_values:
                            try:
                                # Parse the raw EMG values from ESP32
                                raw_emg_values = [float(parts[i]) for i in range(expected_values)]
                                
                                # Enhance to full feature set
                                enhanced_values = self.enhance_raw_features(raw_emg_values)
                                
                                # Store both raw and enhanced
                                self.emg_buffer.append(raw_emg_values)
                                self.enhanced_buffer.append(enhanced_values)
                                
                                # Put enhanced data in queue for main thread
                                self.data_queue.put(('data', enhanced_values))
                                
                            except ValueError as e:
                                pass  # Skip bad data
                        
            except Exception as e:
                self.data_queue.put(('error', str(e)))
            
            time.sleep(0.002)  # ~500Hz
    
    def run_test(self):
        """Run configurable testing session"""
        print(f"\n🚀 Starting Subvocal Detection Test")
        print("=" * 60)
        
        # Connect to serial
        if not self.connect_serial():
            return False
        
        # Create GUI
        if not self.create_gui():
            return False
        
        # Initialize test variables
        self.predictions = []
        self.actual_labels = []
        self.prediction_times = []
        self.correct_predictions = 0
        self.total_predictions = 0
        self.start_time = time.time()
        self.is_collecting = True
        self.last_prediction_time = 0
        
        # Start data collection thread
        collection_thread = threading.Thread(target=self.data_collection_thread)
        collection_thread.daemon = True
        collection_thread.start()
        
        print(f"📝 Subvocal detection test started!")
        print(f"🎯 Model will predict: {', '.join([name.upper() for name in self.class_names])}")
        print(f"📡 Using {self.num_channels} channels with {self.total_features} features")
        print(f"🔄 Continuous prediction every 200ms")
        print(f"🎮 Press ESC or SPACE to stop early")
        
        # Start GUI update loop
        self.update_gui_loop()
        
        # Start GUI main loop
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"GUI error: {e}")
        
        # Cleanup
        self.is_collecting = False
        if self.ser:
            self.ser.close()
        
        return True
    
    def update_gui_loop(self):
        """Configurable GUI update loop with continuous prediction"""
        if not self.is_collecting:
            return
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Check if test is complete
        if elapsed >= self.test_duration:
            self.finish_test()
            return
        
        # Update progress and time
        progress = (elapsed / self.test_duration) * 100
        self.progress_var.set(progress)
        remaining = self.test_duration - elapsed
        mins, secs = int(remaining // 60), int(remaining % 60)
        self.time_label.config(text=f"Time remaining: {mins:02d}:{secs:02d}")
        
        # Initialize configurable state machine if needed
        if not hasattr(self, 'current_command_state'):
            self.current_command_state = 'thinking'
            self.last_state_change = current_time
            self.current_command = None
            self.thinking_duration = random.uniform(2.0, 5.0)  # 2-5 seconds for subvocal
            self.thinking_label = 'thinking' if 'thinking' in self.class_names else self.class_names[-1]
        
        # Configurable command state machine
        if self.current_command_state == 'thinking':
            display_thinking = "REST" if self.thinking_label == 'thinking' else self.thinking_label.upper()
            self.command_label.config(text=display_thinking, fg="yellow")
            
            if current_time - self.last_state_change > self.thinking_duration:
                # Switch to command - exclude thinking/rest state
                available_commands = [name for name in self.class_names if name not in ['thinking', 'rest']]
                if available_commands:
                    self.current_command = random.choice(available_commands)
                    self.current_command_state = 'command'
                    self.last_state_change = current_time
                    self.command_label.config(text=f"SAY: {self.current_command.upper()}", fg="lime")
                    print(f"💭 Subvocal Command: {self.current_command.upper()}")
                
        elif self.current_command_state == 'command':
            if current_time - self.last_state_change > 2.0:  # 2 seconds for subvocal pronunciation
                # Switch back to thinking
                self.current_command_state = 'thinking'
                self.last_state_change = current_time
                self.thinking_duration = random.uniform(2.0, 5.0)  # Random 2-5 seconds
                self.current_command = None
        
        # CONTINUOUS PREDICTION (every 200ms)
        if current_time - self.last_prediction_time > 0.2:
            predicted_label, confidence = self.predict_from_buffer()
            
            if predicted_label:
                # Update prediction display
                self.prediction_label.config(text=predicted_label.upper())
                
                # Update confidence display
                conf_text = f"Confidence: {confidence*100:.1f}%"
                self.confidence_label.config(text=conf_text)
                
                # Determine expected label based on current state
                if self.current_command_state == 'thinking':
                    expected_label = self.thinking_label
                else:
                    expected_label = self.current_command
                
                # Check correctness
                is_correct = predicted_label == expected_label
                
                # Update status
                if is_correct:
                    self.status_label.config(text="✅ CORRECT", fg="green")
                    self.correct_predictions += 1
                else:
                    self.status_label.config(text="❌ INCORRECT", fg="red")
                
                self.total_predictions += 1
                
                # Store results
                self.predictions.append(predicted_label)
                self.actual_labels.append(expected_label)
                self.prediction_times.append(current_time)
                
                # Update accuracy display
                accuracy = (self.correct_predictions / self.total_predictions) * 100
                self.accuracy_label.config(
                    text=f"Accuracy: {accuracy:.1f}% ({self.correct_predictions}/{self.total_predictions})"
                )
                
                # Debug output (every 25 predictions)
                if self.total_predictions % 25 == 0:
                    print(f"Expected: {expected_label.upper()} | Predicted: {predicted_label.upper()} | "
                          f"Correct: {'✅' if is_correct else '❌'} | Confidence: {confidence:.3f} | "
                          f"Accuracy: {accuracy:.1f}%")
            
            self.last_prediction_time = current_time
        
        # Process data queue
        try:
            while not self.data_queue.empty():
                msg_type, data = self.data_queue.get_nowait()
                if msg_type == 'error':
                    print(f"Data collection error: {data}")
        except:
            pass
        
        # Schedule next update
        if self.root:
            self.root.after(100, self.update_gui_loop)
    
    def stop_test(self):
        """Stop the test manually"""
        self.is_collecting = False
        self.finish_test()
    
    def finish_test(self):
        """Finish the test and show configurable results"""
        self.is_collecting = False
        
        if self.root:
            self.root.destroy()
        
        # Calculate final results
        if self.total_predictions > 0:
            final_accuracy = (self.correct_predictions / self.total_predictions) * 100
            
            print(f"\n🎉 Subvocal Detection Test Complete!")
            print("=" * 60)
            print(f"Model: {self.selected_model}")
            print(f"Configuration: {self.num_classes} classes, {self.num_channels} channels")
            print(f"Classes: {', '.join([name.upper() for name in self.class_names])}")
            print(f"Duration: {self.test_duration}s")
            print(f"Total Predictions: {self.total_predictions}")
            print(f"Correct Predictions: {self.correct_predictions}")
            print(f"Overall Accuracy: {final_accuracy:.2f}%")
            
            if len(self.predictions) == len(self.actual_labels):
                # Per-class analysis
                print(f"\n📊 Per-Class Analysis:")
                for label in self.class_names:
                    # Actual occurrences
                    actual_count = sum(1 for al in self.actual_labels if al == label)
                    
                    # Predicted occurrences  
                    pred_count = sum(1 for pl in self.predictions if pl == label)
                    
                    # Correct predictions for this class
                    correct_for_class = sum(1 for al, pl in zip(self.actual_labels, self.predictions) 
                                          if al == label and pl == label)
                    
                    # Class accuracy and precision
                    class_accuracy = (correct_for_class / max(1, actual_count)) * 100
                    precision = (correct_for_class / max(1, pred_count)) * 100
                    
                    display_name = "THINKING (REST)" if label == "thinking" else label.upper()
                    print(f"   • {display_name}:")
                    print(f"     - Actual: {actual_count} times ({actual_count/self.total_predictions*100:.1f}%)")
                    print(f"     - Predicted: {pred_count} times ({pred_count/self.total_predictions*100:.1f}%)")
                    print(f"     - Correct: {correct_for_class} times")
                    print(f"     - Recall: {class_accuracy:.1f}%")
                    print(f"     - Precision: {precision:.1f}%")
                
                # Detailed metrics
                print(f"\n📈 Detailed Classification Report:")
                try:
                    from sklearn.metrics import classification_report
                    target_names = [name.upper() if name != 'thinking' else 'THINKING' for name in self.class_names]
                    print(classification_report(self.actual_labels, self.predictions, 
                                               target_names=target_names, zero_division=0))
                except Exception as e:
                    print(f"Could not generate classification report: {e}")
                
                # Confusion matrix
                print(f"\n🎯 Confusion Matrix:")
                try:
                    cm = confusion_matrix(self.actual_labels, self.predictions, labels=self.class_names)
                    print("        Predicted:")
                    headers = [name[:5].upper() for name in self.class_names]
                    print("        " + "  ".join([f"{h:>5}" for h in headers]))
                    for i, true_label in enumerate(self.class_names):
                        display_label = "THINK" if true_label == "thinking" else true_label[:5].upper()
                        print(f"True {display_label:>5}: {cm[i]}")
                except Exception as e:
                    print(f"Could not generate confusion matrix: {e}")
                
                # Save results
                self.save_results()
            else:
                print("⚠️  Prediction and actual label counts don't match")
        else:
            print("❌ No predictions made during test")
    
    def save_results(self):
        """Save configurable test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.selected_model
        config_suffix = f"{self.num_classes}class_{self.num_channels}ch"
        
        # Results CSV
        results_filename = f'subvocal_test_results_{config_suffix}_{model_name}_{timestamp}.csv'
        
        with open(results_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'actual_label', 'predicted_label', 'is_correct', 
                'model_path', 'test_duration', 'num_classes', 'num_channels',
                'class_names', 'features_used'
            ])
            
            for i in range(len(self.predictions)):
                is_correct = self.actual_labels[i] == self.predictions[i]
                writer.writerow([
                    self.prediction_times[i] if i < len(self.prediction_times) else i,
                    self.actual_labels[i],
                    self.predictions[i],
                    is_correct,
                    self.selected_model,
                    self.test_duration,
                    self.num_classes,
                    self.num_channels,
                    ','.join(self.class_names),
                    f'{self.total_features}_features_configurable'
                ])
        
        # Summary JSON
        summary = {
            'model_name': self.selected_model,
            'model_type': 'threshold' if self.selected_model == "Threshold_Model" else self.available_models[self.selected_model]['type'],
            'test_type': f'{self.num_classes}_class_subvocal_detection',
            'configuration': {
                'num_classes': self.num_classes,
                'num_channels': self.num_channels,
                'class_names': self.class_names,
                'total_features': self.total_features,
                'feature_names': self.feature_names
            },
            'test_duration': self.test_duration,
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'overall_accuracy': (self.correct_predictions / max(1, self.total_predictions)) * 100,
            'features_used': f'configurable_{self.total_features}_features',
            'timestamp': datetime.now().isoformat()
        }
        
        # Add threshold-specific info
        if self.selected_model == "Threshold_Model":
            summary['threshold_config'] = self.model_data['thresholds']
            summary['model_type'] = 'threshold'
        
        # Add confusion matrix and classification report if possible
        try:
            summary['confusion_matrix'] = confusion_matrix(self.actual_labels, self.predictions, 
                                                         labels=self.class_names).tolist()
            summary['classification_report'] = classification_report(self.actual_labels, self.predictions, 
                                                                   target_names=self.class_names, 
                                                                   output_dict=True, zero_division=0)
        except Exception as e:
            print(f"Could not save detailed metrics: {e}")
        
        summary_filename = f'subvocal_test_summary_{config_suffix}_{model_name}_{timestamp}.json'
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Subvocal Detection Test Results Saved:")
        print(f"   📊 Data: {results_filename}")
        print(f"   📋 Summary: {summary_filename}")


def run_configurable_model_test(models_dir="adaptive_models", serial_port='/dev/cu.usbserial-110'):
    """
    Complete configurable subvocal detection testing workflow
    
    Features:
    - User specifies number of classes and channels
    - Auto-finds compatible models
    - Adapts inference pipeline to match configuration
    - Perfect feature alignment between training and inference
    """
    print("🧪 Configurable Subvocal Detection Testing System")
    print("=" * 50)
    
    # Initialize configurable tester
    tester = ConfigurableEMGInference(models_dir=models_dir, serial_port=serial_port)
    
    # Setup user configuration
    if not tester.setup_user_configuration():
        return False
    
    # Discover compatible models
    if not tester.discover_models():
        return False
    
    # Select model (shows only compatible ones)
    if not tester.select_model_interactive():
        return False
    
    # Load model
    if not tester.load_selected_model():
        return False
    
    # Setup test parameters
    if not tester.setup_test_parameters():
        return False
    
    # Run configurable test
    success = tester.run_test()
    
    if success:
        print("✅ Subvocal detection test completed successfully")
    else:
        print("❌ Test failed")
    
    return success


def main():
    """Main function with configuration options"""
    print("🎯 Configurable Subvocal Detection System")
    print("=" * 50)
    print("Features:")
    print("  - User configurable: number of subvocal classes (1-5 + thinking)")
    print("  - User configurable: number of EMG channels (1-4)")
    print("  - Auto-finds compatible trained models")
    print("  - Perfect feature alignment: collection → training → inference")
    print("  - Adaptive GUI and evaluation metrics")
    print("  - Continuous real-time prediction")
    
    # Configuration options
    print(f"\n⚙️  Configuration Options:")
    print("-" * 30)
    
    # Models directory
    models_dir = input("Models directory (default: 'adaptive_models'): ").strip()
    if not models_dir:
        models_dir = "adaptive_models"
    
    # Serial port
    serial_port = input("Serial port (default: '/dev/cu.usbserial-110'): ").strip()
    if not serial_port:
        serial_port = '/dev/cu.usbserial-110'
    
    print(f"\n✅ Setup Configuration:")
    print(f"   Models directory: {models_dir}")
    print(f"   Serial port: {serial_port}")
    
    # Run the configurable test
    success = run_configurable_model_test(models_dir=models_dir, serial_port=serial_port)
    
    if success:
        print(f"\n🎉 Subvocal detection testing completed!")
        print(f"   The system adapted to your specified configuration")
        print(f"   Check the saved results for detailed performance metrics")
    else:
        print(f"\n❌ Testing failed or was cancelled")
    
    return success


if __name__ == "__main__":
    main()