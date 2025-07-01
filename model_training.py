"""
Complete Adaptive EMG Training System
=====================================
Automatically adapts to different configurations:
- Any number of classes (2-10)
- Any number of channels (1-4)  
- Dynamic feature extraction
- Scalable model architectures
- Configuration-aware preprocessing

This system reads CSV files and automatically detects:
- Number of classes from data
- Number of channels from column headers
- Generates appropriate models for the configuration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdaptiveEMGClassifier:
    def __init__(self, window_size=500, overlap=0.5, models_dir="adaptive_models"):
        """
        Adaptive EMG Classification System
        
        Automatically detects and adapts to:
        - Number of classes (from data labels)
        - Number of channels (from CSV columns)
        - Feature configuration (based on detected channels)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        self.models_dir = models_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Will be auto-detected from data
        self.num_classes = None
        self.num_channels = None
        self.class_names = []
        self.feature_names = []
        self.total_features = None
        
        # Configuration info
        self.config = {}
        
        # Create models directory
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
    
    def setup_user_configuration(self, csv_file):
        """Interactive setup for training configuration"""
        print("🧠 EMG Training Configuration")
        print("=" * 50)
        print("Note: 'thinking' class is automatically included as the rest state")
        
        # Get number of additional classes (thinking is always included)
        while True:
            try:
                num_additional_classes = int(input("How many additional classes in your data? (1-5, 'thinking' is always included): "))
                if 1 <= num_additional_classes <= 5:
                    break
                else:
                    print("❌ Please enter a number between 1 and 5")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Get class names (thinking is always included)
        print(f"\n'thinking' class is automatically included as the rest state.")
        print(f"Enter names for {num_additional_classes} additional classes:")
        self.class_names = []
        for i in range(num_additional_classes):
            while True:
                class_name = input(f"Additional class {i+1} name: ").strip().lower()
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
                self.num_channels = int(input(f"\nHow many EMG channels in your data? (1-4): "))
                if 1 <= self.num_channels <= 4:
                    break
                else:
                    print("❌ Please enter a number between 1 and 4")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Generate expected feature names based on user input
        self.feature_names = []
        for ch in range(1, self.num_channels + 1):
            ch_features = [
                f'ch{ch}_filtered', f'ch{ch}_envelope', 
                f'ch{ch}_filtered_rms', f'ch{ch}_filtered_movavg',
                f'ch{ch}_envelope_rms', f'ch{ch}_envelope_movavg'
            ]
            self.feature_names.extend(ch_features)
        
        self.total_features = len(self.feature_names)
        
        # Store configuration
        self.config = {
            'num_channels': self.num_channels,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'feature_names': self.feature_names,
            'total_features': self.total_features,
            'features_per_channel': 6,
            'csv_file': csv_file
        }
        
        # Display configuration summary
        self.show_configuration_summary()
        
        return True
    
    def show_configuration_summary(self):
        """Display the configuration summary"""
        print("\n" + "=" * 50)
        print("📋 TRAINING CONFIGURATION")
        print("=" * 50)
        
        active_classes = [name for name in self.class_names if name != 'thinking']
        
        print(f"🎯 Classes: {', '.join([name.upper() for name in self.class_names])}")
        print(f"   • Active classes: {', '.join([name.upper() for name in active_classes])}")
        print(f"   • Rest state: THINKING (automatic)")
        print(f"📡 Channels: {self.num_channels}")
        print(f"🔧 Features: {self.total_features} total ({self.total_features//self.num_channels} per channel)")
        print(f"📋 Expected features: {self.feature_names}")
        
        confirm = input(f"\nProceed with this configuration? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Configuration cancelled. Restarting setup...")
            return self.setup_user_configuration(self.config['csv_file'])
    
    def validate_data_against_config(self, csv_file):
        """Validate that CSV data matches user configuration"""
        print(f"📊 Validating data against configuration...")
        
        try:
            df_sample = pd.read_csv(csv_file, nrows=100)
            print(f"🔍 Available columns: {list(df_sample.columns)}")
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return False
        
        # Check if all expected features exist
        missing_features = []
        for feature in self.feature_names:
            if feature not in df_sample.columns:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing expected features: {missing_features}")
            print(f"   Expected: {self.feature_names}")
            print(f"   Available: {[col for col in df_sample.columns if col.startswith('ch')]}")
            return False
        
        # Check label column
        if 'label' not in df_sample.columns:
            print("❌ No 'label' column found in CSV file!")
            return False
        
        # Check if expected classes exist in data
        full_df = pd.read_csv(csv_file)
        data_labels = set(full_df['label'].unique())
        expected_labels = set(self.class_names)
        
        missing_classes = expected_labels - data_labels
        extra_classes = data_labels - expected_labels
        
        if missing_classes:
            print(f"❌ Missing expected classes in data: {missing_classes}")
            return False
        
        if extra_classes:
            print(f"⚠️  Extra classes found in data: {extra_classes}")
            print(f"   These will be ignored during training")
        
        print(f"✅ Data validation successful!")
        print(f"   Found all {self.num_channels} channels with {len(self.feature_names)} features")
        print(f"   Found all {self.num_classes} expected classes")
        
        return True
    
    def load_data(self, csv_file):
        """Load and validate data with detected configuration"""
        print(f"📊 Loading data with detected configuration...")
        
        try:
            self.df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(self.df)} samples")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        
        # Validate all expected columns exist
        missing_cols = [col for col in self.feature_names + ['label'] if col not in self.df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        
        # Create relative time if timestamp exists
        if 'timestamp' in self.df.columns:
            self.df['time_sec'] = self.df['timestamp'] - self.df['timestamp'].iloc[0]
        
        # Analyze class distribution
        print(f"📊 Label distribution:")
        for label, count in self.df['label'].value_counts().items():
            print(f"   • {label}: {count:,} samples ({count/len(self.df)*100:.1f}%)")
        
        # Check for class imbalance
        label_counts = self.df['label'].value_counts()
        min_count = label_counts.min()
        max_count = label_counts.max()
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 3:
            print(f"⚠️ Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
            print("   Will use balanced class weights in models")
        
        return True
    
    def create_windows(self):
        """Create sliding windows with detected features"""
        print(f"🔄 Creating sliding windows...")
        print(f"   • Window size: {self.window_size} samples (~{self.window_size/500:.1f}s)")
        print(f"   • Features: {self.total_features} ({self.num_channels} channels)")
        
        windows = []
        labels = []
        window_info = []
        
        for i in range(0, len(self.df) - self.window_size + 1, self.step_size):
            window_data = self.df.iloc[i:i + self.window_size]
            
            # Majority vote for window label
            label_counts = window_data['label'].value_counts()
            majority_label = label_counts.index[0]
            
            # Keep windows with >70% label purity
            label_purity = label_counts.iloc[0] / len(window_data)
            if label_purity > 0.7:
                # Extract features for this window
                window_features = window_data[self.feature_names].values
                windows.append(window_features)
                labels.append(majority_label)
                
                start_time = window_data['time_sec'].iloc[0] if 'time_sec' in window_data.columns else i
                window_info.append({
                    'start_idx': i,
                    'start_time': start_time,
                    'label': majority_label,
                    'purity': label_purity
                })
        
        self.X = np.array(windows)
        self.y = np.array(labels)
        self.window_info = window_info
        
        print(f"✅ Created {len(windows)} windows")
        print(f"   Shape: {self.X.shape} (windows, time_steps, features)")
        
        # Display window distribution
        unique_labels, counts = np.unique(self.y, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"   • {label}: {count:,} windows ({count/len(self.y)*100:.1f}%)")
        
        return len(windows) > 0
    
    def prepare_features(self, method='statistical'):
        """Prepare features for training with adaptive sizing"""
        print(f"🔧 Preparing {method} features for {self.num_classes}-class, {self.num_channels}-channel problem...")
        
        if method == 'statistical':
            # Extract statistical features from each channel
            features = []
            feature_names = []
            
            for i, feature_name in enumerate(self.feature_names):
                channel_data = self.X[:, :, i]  # All windows, all time points, this feature
                
                # 10 statistical features per channel feature
                means = np.mean(channel_data, axis=1)
                stds = np.std(channel_data, axis=1)
                mins = np.min(channel_data, axis=1)
                maxs = np.max(channel_data, axis=1)
                medians = np.median(channel_data, axis=1)
                p25 = np.percentile(channel_data, 25, axis=1)
                p75 = np.percentile(channel_data, 75, axis=1)
                energy = np.sum(channel_data**2, axis=1)
                power = energy / self.window_size
                
                # Zero crossings for filtered signals
                if 'filtered' in feature_name:
                    zero_crossings = np.sum(np.diff(np.sign(channel_data), axis=1) != 0, axis=1)
                else:
                    zero_crossings = np.zeros(len(channel_data))
                
                features.extend([means, stds, mins, maxs, medians, p25, p75, energy, power, zero_crossings])
                feature_names.extend([
                    f'{feature_name}_mean', f'{feature_name}_std', f'{feature_name}_min', 
                    f'{feature_name}_max', f'{feature_name}_median', f'{feature_name}_p25',
                    f'{feature_name}_p75', f'{feature_name}_energy', f'{feature_name}_power',
                    f'{feature_name}_zero_crossings'
                ])
            
            self.X_processed = np.column_stack(features)
            self.feature_names_processed = feature_names
            
            statistical_features = len(self.feature_names) * 10
            print(f"   ✅ Statistical features: {self.X_processed.shape}")
            print(f"   📊 Total: {statistical_features} features ({len(self.feature_names)} enhanced features × 10 stats each)")
            print(f"   🔧 From {self.num_channels} channel(s) with {len(self.feature_names)} total enhanced features")
            
        elif method == 'temporal':
            # Keep temporal structure for LSTM
            self.X_processed = self.X
            self.feature_names_processed = self.feature_names
            
            print(f"   ✅ Temporal features: {self.X_processed.shape}")
            print(f"   📊 Format: (windows, time_steps, channels) = {self.X_processed.shape}")
        
        return True
    
    def get_adaptive_model_configs(self):
        """Generate model configurations adapted to the number of classes and channels"""
        
        # Determine class weights strategy
        use_balanced = self.num_classes > 2  # Always balance for multi-class
        
        configs = {
            'Random_Forest': {
                'model_class': RandomForestClassifier,
                'params': {
                    'n_estimators': min(200, 50 * self.num_channels),  # More trees for more channels
                    'max_depth': max(10, 5 * self.num_classes),        # Deeper for more classes
                    'random_state': 42,
                    'class_weight': 'balanced' if use_balanced else None
                },
                'type': 'classical'
            },
            'Gradient_Boosting': {
                'model_class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': min(150, 30 * self.num_channels),
                    'learning_rate': 0.1,
                    'max_depth': max(6, 2 * self.num_classes),
                    'random_state': 42
                },
                'type': 'classical'
            },
            'SVM': {
                'model_class': SVC,
                'params': {
                    'kernel': 'rbf',
                    'random_state': 42,
                    'probability': True,
                    'class_weight': 'balanced' if use_balanced else None,
                    'C': 1.0 * self.num_channels  # Scale C with number of channels
                },
                'type': 'classical'
            },
            'Logistic_Regression': {
                'model_class': LogisticRegression,
                'params': {
                    'random_state': 42,
                    'max_iter': 1000 * self.num_classes,  # More iterations for more classes
                    'class_weight': 'balanced' if use_balanced else None,
                    'multi_class': 'ovr' if self.num_classes > 2 else 'auto'
                },
                'type': 'classical'
            },
            'LSTM': {
                'type': 'deep_learning'
            }
        }
        
        return configs
    
    def create_adaptive_lstm(self, input_shape, n_classes):
        """Create LSTM architecture adapted to the number of classes and channels"""
        
        # Scale architecture based on number of channels and classes
        base_units = 32
        lstm_units_1 = base_units * min(4, self.num_channels)  # Scale with channels
        lstm_units_2 = base_units * min(2, self.num_channels)
        dense_units = base_units * min(3, self.num_classes)    # Scale with classes
        
        # More complex architecture for more classes
        if n_classes <= 3:
            # Simple architecture for binary/ternary classification
            model = Sequential([
                LSTM(lstm_units_1, return_sequences=True, input_shape=input_shape),
                Dropout(0.3),
                LSTM(lstm_units_2, return_sequences=False),
                Dropout(0.3),
                Dense(dense_units, activation='relu'),
                Dropout(0.2),
                Dense(n_classes, activation='softmax', dtype='float32')
            ])
        else:
            # More complex architecture for multi-class problems
            model = Sequential([
                Bidirectional(LSTM(lstm_units_1, return_sequences=True), input_shape=input_shape),
                Dropout(0.3),
                Bidirectional(LSTM(lstm_units_2, return_sequences=True)),
                Dropout(0.3),
                GlobalAveragePooling1D(),  # Better for multi-class
                Dense(dense_units * 2, activation='relu'),
                Dropout(0.4),
                Dense(dense_units, activation='relu'),
                Dropout(0.3),
                Dense(n_classes, activation='softmax', dtype='float32')
            ])
        
        # Adaptive learning rate based on problem complexity
        learning_rate = 0.001 / max(1, self.num_classes // 3)
        optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"🏗️  Adaptive LSTM architecture:")
        print(f"   • LSTM units: {lstm_units_1} → {lstm_units_2}")
        print(f"   • Dense units: {dense_units}")
        print(f"   • Learning rate: {learning_rate}")
        print(f"   • Architecture type: {'Bidirectional' if n_classes > 3 else 'Standard'}")
        
        return model
    
    def train_classical_model(self, model_name, config):
        """Train classical model with adaptive configuration"""
        print(f"🤖 Training {model_name} for {self.num_classes}-class problem...")
        
        # Create model folder
        model_folder = self.create_model_folder(model_name)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(self.y)
        
        # Adaptive train/test split - more validation data for complex problems
        test_size = min(0.3, 0.15 + 0.05 * self.num_classes)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_processed, y_encoded, test_size=test_size,
            random_state=42, stratify=y_encoded
        )
        
        print(f"   📊 Train/test split: {len(X_train)}/{len(X_test)} (test: {test_size:.1%})")
        
        # Scale features for certain models
        if model_name in ['SVM', 'Logistic_Regression']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            train_data, test_data = X_train_scaled, X_test_scaled
            self.current_scaler = scaler
        else:
            train_data, test_data = X_train, X_test
            self.current_scaler = None
        
        # Create and train model
        model = config['model_class'](**config['params'])
        model.fit(train_data, y_train)
        
        # Evaluate
        y_pred = model.predict(test_data)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Per-class accuracy
        class_accuracies = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                class_accuracies[class_name] = class_acc
        
        # Save model with adaptive configuration
        model_path = os.path.join(model_folder, 'model', f'{model_name.lower()}.joblib')
        model_data = {
            'model': model,
            'scaler': self.current_scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names_processed,
            'window_size': self.window_size,
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'model_type': 'classical',
            'config': self.config,  # Store detected configuration
            'adaptive_params': config['params']
        }
        joblib.dump(model_data, model_path)
        
        # Save adaptive configuration summary
        self.save_model_artifacts(model_folder, model_name, y_test, y_pred, accuracy, class_accuracies, config['params'])
        
        print(f"   ✅ {model_name} - Overall: {accuracy:.3f}")
        for class_name, class_acc in class_accuracies.items():
            print(f"      • {class_name}: {class_acc:.3f}")
        
        return {
            'model': model,
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'model_folder': model_folder
        }
    
    def train_lstm_model(self, epochs=50, batch_size=32):
        """Train adaptive LSTM model"""
        print(f"🧠 Training adaptive LSTM for {self.num_classes}-class problem...")
        
        model_folder = self.create_model_folder('LSTM')
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(self.y)
        n_classes = len(np.unique(y_encoded))
        
        # Adaptive parameters based on problem complexity
        epochs = max(30, 10 * self.num_classes)  # More epochs for more classes
        batch_size = max(16, min(64, 32 // max(1, self.num_classes // 3)))  # Adaptive batch size
        
        print(f"   📊 Adaptive training params: epochs={epochs}, batch_size={batch_size}")
        
        # Split data
        test_size = min(0.3, 0.15 + 0.05 * self.num_classes)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_processed, y_encoded, test_size=test_size,
            random_state=42, stratify=y_encoded
        )
        
        # Convert to categorical
        y_train_cat = tf.keras.utils.to_categorical(y_train, n_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test, n_classes)
        
        # Normalize features
        X_mean = np.mean(X_train, axis=(0,1), keepdims=True)
        X_std = np.std(X_train, axis=(0,1), keepdims=True) + 1e-8
        X_train_norm = (X_train - X_mean) / X_std
        X_test_norm = (X_test - X_mean) / X_std
        
        # Create adaptive model
        input_shape = (self.window_size, self.total_features)
        model = self.create_adaptive_lstm(input_shape, n_classes)
        
        # Adaptive callbacks based on complexity
        patience = max(10, 5 * self.num_classes)
        callbacks = [
            EarlyStopping(patience=patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=patience//2, verbose=1, min_lr=1e-7)
        ]
        
        # Class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        # Train model
        history = model.fit(
            X_train_norm, y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test_norm, y_test_cat),
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Evaluate
        y_pred_prob = model.predict(X_test_norm, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Per-class accuracy
        class_accuracies = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                class_accuracies[class_name] = class_acc
        
        # Save adaptive model
        model_path = os.path.join(model_folder, 'model', 'lstm_model.h5')
        model.save(model_path)
        
        # Save preprocessing parameters
        preprocessing_params = {
            'X_mean': X_mean.tolist(),
            'X_std': X_std.tolist(),
            'label_encoder_classes': self.label_encoder.classes_.tolist(),
            'feature_names': self.feature_names,
            'window_size': self.window_size,
            'n_features': self.total_features,
            'n_classes': n_classes,
            'config': self.config,
            'adaptive_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'patience': patience,
                'class_weights': class_weight_dict
            }
        }
        
        preprocessing_path = os.path.join(model_folder, 'model', 'preprocessing_params.json')
        with open(preprocessing_path, 'w') as f:
            json.dump(preprocessing_params, f, indent=2)
        
        # Save model artifacts
        adaptive_params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'patience': patience,
            'architecture': 'Bidirectional' if n_classes > 3 else 'Standard'
        }
        self.save_model_artifacts(model_folder, 'LSTM', y_test, y_pred, accuracy, class_accuracies, adaptive_params)
        
        print(f"   ✅ LSTM - Overall: {accuracy:.3f}")
        for class_name, class_acc in class_accuracies.items():
            print(f"      • {class_name}: {class_acc:.3f}")
        
        return {
            'model': model,
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'model_folder': model_folder
        }
    
    def create_model_folder(self, model_name):
        """Create model folder with adaptive naming"""
        config_suffix = f"{self.num_classes}class_{self.num_channels}ch"
        model_folder = os.path.join(self.models_dir, f"{model_name}_{config_suffix}")
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        
        for subfolder in ['model', 'metrics', 'plots', 'results']:
            Path(os.path.join(model_folder, subfolder)).mkdir(exist_ok=True)
        
        return model_folder
    
    def save_model_artifacts(self, model_folder, model_name, y_test, y_pred, accuracy, class_accuracies, adaptive_params):
        """Save all model artifacts with adaptive configuration info"""
        
        # Confusion matrix with adaptive sizing
        plt.figure(figsize=(max(8, self.num_classes * 2), max(6, self.num_classes * 1.5)))
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[name.upper() for name in self.label_encoder.classes_],
                   yticklabels=[name.upper() for name in self.label_encoder.classes_])
        
        plt.title(f'{self.num_classes}-Class Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        cm_path = os.path.join(model_folder, 'plots', 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Classification report
        report = classification_report(y_test, y_pred, 
                                     target_names=self.label_encoder.classes_,
                                     output_dict=True, zero_division=0)
        
        report_path = os.path.join(model_folder, 'metrics', 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Adaptive model summary
        summary = {
            'model_name': model_name,
            'model_type': 'classical' if model_name != 'LSTM' else 'deep_learning',
            'adaptive_configuration': self.config,
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'adaptive_parameters': adaptive_params,
            'classes': self.label_encoder.classes_.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(model_folder, 'model_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def create_adaptive_session_summary(self, results):
        """Create comprehensive session summary"""
        print(f"\n📊 Creating adaptive session summary...")
        
        session_folder = os.path.join(self.models_dir, f"session_{self.session_name}")
        Path(session_folder).mkdir(parents=True, exist_ok=True)
        
        # Aggregate results
        summary_data = {
            'session_info': {
                'session_name': self.session_name,
                'timestamp': datetime.now().isoformat(),
                'configuration': self.config,
                'total_models_trained': len(results)
            },
            'adaptive_configuration': {
                'detected_classes': self.class_names,
                'detected_channels': self.num_channels,
                'total_features': self.total_features,
                'window_size': self.window_size,
                'overlap': self.overlap
            },
            'model_results': {}
        }
        
        # Collect all model results
        best_accuracy = 0
        best_model = None
        
        for model_name, result in results.items():
            summary_data['model_results'][model_name] = {
                'accuracy': result['accuracy'],
                'class_accuracies': result['class_accuracies'],
                'model_folder': result['model_folder']
            }
            
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = model_name
        
        summary_data['session_info']['best_model'] = best_model
        summary_data['session_info']['best_accuracy'] = best_accuracy
        
        # Save session summary
        summary_path = os.path.join(session_folder, 'session_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Create comparison visualization
        self.create_model_comparison_plot(results, session_folder)
        
        print(f"   ✅ Session summary saved: {summary_path}")
        print(f"   🏆 Best model: {best_model} ({best_accuracy:.3f} accuracy)")
        
        return summary_data
    
    def create_model_comparison_plot(self, results, session_folder):
        """Create comparison plots for all trained models"""
        
        # Overall accuracy comparison
        plt.figure(figsize=(12, 8))
        
        model_names = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in model_names]
        
        # Create color map based on performance
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
        
        bars = plt.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black')
        plt.title(f'Model Comparison - {self.num_classes} Classes, {self.num_channels} Channels', 
                 fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=14)
        plt.xlabel('Model', fontsize=14)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        
        comparison_path = os.path.join(session_folder, 'model_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Per-class accuracy heatmap
        if len(self.class_names) > 1:
            plt.figure(figsize=(max(10, len(model_names) * 2), max(6, len(self.class_names))))
            
            # Create heatmap data
            heatmap_data = []
            for class_name in self.class_names:
                class_row = []
                for model_name in model_names:
                    if class_name in results[model_name]['class_accuracies']:
                        class_row.append(results[model_name]['class_accuracies'][class_name])
                    else:
                        class_row.append(0)
                heatmap_data.append(class_row)
            
            heatmap_data = np.array(heatmap_data)
            
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                       xticklabels=model_names, yticklabels=[name.upper() for name in self.class_names],
                       cbar_kws={'label': 'Accuracy'})
            
            plt.title(f'Per-Class Accuracy Heatmap', fontsize=14, fontweight='bold')
            plt.xlabel('Models', fontsize=12)
            plt.ylabel('Classes', fontsize=12)
            plt.tight_layout()
            
            heatmap_path = os.path.join(session_folder, 'per_class_heatmap.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def train_all_adaptive_models(self, session_name=None, epochs=None):
        """Train all models with adaptive configurations"""
        if session_name:
            self.session_name = session_name
        else:
            config_suffix = f"{self.num_classes}class_{self.num_channels}ch"
            self.session_name = f"adaptive_{config_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🚀 Training Adaptive EMG Models")
        print(f"📁 Session: {self.session_name}")
        print(f"🎯 Configuration: {self.num_classes} classes, {self.num_channels} channels")
        print("=" * 70)
        
        results = {}
        model_configs = self.get_adaptive_model_configs()
        
        # Train classical models
        self.prepare_features(method='statistical')
        print(f"\n🤖 Training Classical Models...")
        print("-" * 50)
        
        for model_name, config in model_configs.items():
            if config['type'] == 'classical':
                try:
                    result = self.train_classical_model(model_name, config)
                    results[model_name] = result
                except Exception as e:
                    print(f"   ❌ {model_name} failed: {e}")
        
        # Train LSTM model
        self.prepare_features(method='temporal')
        print(f"\n🧠 Training Adaptive LSTM...")
        print("-" * 50)
        
        try:
            lstm_result = self.train_lstm_model(epochs=epochs)
            results['LSTM'] = lstm_result
        except Exception as e:
            print(f"   ❌ LSTM failed: {e}")
        
        # Create session summary
        session_summary = self.create_adaptive_session_summary(results)
        
        print(f"\n🎉 Adaptive Training Complete!")
        print(f"📁 Models saved in: {self.models_dir}")
        print(f"🏆 Best performing model: {session_summary['session_info']['best_model']}")
        print(f"📊 Best accuracy: {session_summary['session_info']['best_accuracy']:.3f}")
        
        return results, session_summary
    
    def evaluate_cross_validation(self, cv_folds=5):
        """Perform cross-validation evaluation for all models"""
        print(f"🔄 Performing {cv_folds}-fold cross-validation...")
        
        # Prepare data for CV
        self.prepare_features(method='statistical')
        y_encoded = self.label_encoder.fit_transform(self.y)
        
        model_configs = self.get_adaptive_model_configs()
        cv_results = {}
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for model_name, config in model_configs.items():
            if config['type'] == 'classical':
                print(f"   🔄 CV for {model_name}...")
                
                fold_scores = []
                for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_processed, y_encoded)):
                    X_train_fold, X_val_fold = self.X_processed[train_idx], self.X_processed[val_idx]
                    y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]
                    
                    # Scale if needed
                    if model_name in ['SVM', 'Logistic_Regression']:
                        scaler = StandardScaler()
                        X_train_fold = scaler.fit_transform(X_train_fold)
                        X_val_fold = scaler.transform(X_val_fold)
                    
                    # Train and evaluate
                    model = config['model_class'](**config['params'])
                    model.fit(X_train_fold, y_train_fold)
                    y_pred_fold = model.predict(X_val_fold)
                    fold_score = accuracy_score(y_val_fold, y_pred_fold)
                    fold_scores.append(fold_score)
                
                cv_results[model_name] = {
                    'mean_accuracy': np.mean(fold_scores),
                    'std_accuracy': np.std(fold_scores),
                    'fold_scores': fold_scores
                }
                
                print(f"      ✅ {model_name}: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")
        
        return cv_results
    
    def generate_feature_importance_analysis(self, results):
        """Generate feature importance analysis for tree-based models"""
        print(f"🔍 Analyzing feature importance...")
        
        # Prepare statistical features
        self.prepare_features(method='statistical')
        
        importance_results = {}
        
        for model_name, result in results.items():
            if model_name in ['Random_Forest', 'Gradient_Boosting']:
                model = result['model']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = self.feature_names_processed
                    
                    # Get top features
                    indices = np.argsort(importances)[::-1]
                    top_n = min(20, len(feature_names))
                    
                    importance_results[model_name] = {
                        'feature_names': [feature_names[i] for i in indices[:top_n]],
                        'importances': [importances[i] for i in indices[:top_n]]
                    }
                    
                    # Create feature importance plot
                    plt.figure(figsize=(12, 8))
                    plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
                    
                    y_pos = np.arange(top_n)
                    plt.barh(y_pos, [importances[i] for i in indices[:top_n]], alpha=0.8)
                    plt.yticks(y_pos, [feature_names[i] for i in indices[:top_n]])
                    plt.xlabel('Importance')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    
                    # Save plot
                    importance_path = os.path.join(result['model_folder'], 'plots', 'feature_importance.png')
                    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"   ✅ {model_name} feature importance saved")
        
        return importance_results


def main():
    """Main function to run adaptive EMG training"""
    
    # Example usage for different configurations
    classifier = AdaptiveEMGClassifier(
        window_size=500,  # ~1 second at 500 Hz
        overlap=0.5,      # 50% overlap
        models_dir="adaptive_emg_models"
    )
    
    # Get CSV file path from user
    csv_file = input("Enter path to EMG data CSV file: ").strip()
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    try:
        # User configuration instead of auto-detection
        print("\n🎯 STEP 1: User Configuration")
        print("=" * 50)
        classifier.setup_user_configuration(csv_file)
        
        # Validate data matches configuration
        print("\n📊 STEP 2: Data Validation")
        print("=" * 50)
        if not classifier.validate_data_against_config(csv_file):
            print("❌ Data validation failed")
            return
        
        # Load and validate data
        print("\n📊 STEP 3: Loading Data")
        print("=" * 50)
        if not classifier.load_data(csv_file):
            print("❌ Failed to load data")
            return
        
        # Create windows
        print("\n🔄 STEP 4: Creating Windows")
        print("=" * 50)
        if not classifier.create_windows():
            print("❌ Failed to create windows")
            return
        
        # Train all models
        print("\n🚀 STEP 5: Training Models")
        print("=" * 50)
        results, session_summary = classifier.train_all_adaptive_models()
        
        # Optional: Cross-validation evaluation
        print("\n🔄 STEP 6: Cross-Validation (Optional)")
        print("=" * 50)
        cv_choice = input("Perform cross-validation? (y/n): ").strip().lower()
        if cv_choice in ['y', 'yes']:
            cv_results = classifier.evaluate_cross_validation()
            print("\n📊 Cross-Validation Results:")
            for model_name, cv_result in cv_results.items():
                print(f"   • {model_name}: {cv_result['mean_accuracy']:.3f} ± {cv_result['std_accuracy']:.3f}")
        
        # Feature importance analysis
        print("\n🔍 STEP 7: Feature Importance Analysis")
        print("=" * 50)
        importance_results = classifier.generate_feature_importance_analysis(results)
        
        # Final summary
        print("\n🎉 TRAINING COMPLETE!")
        print("=" * 50)
        print(f"📁 All models saved in: {classifier.models_dir}")
        print(f"🏆 Best model: {session_summary['session_info']['best_model']}")
        print(f"📊 Best accuracy: {session_summary['session_info']['best_accuracy']:.3f}")
        print(f"🎯 Trained for: {classifier.num_classes} classes, {classifier.num_channels} channels")
        
        # Model recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if classifier.num_classes <= 3:
            print("   • For real-time use: Try Random Forest or LSTM")
            print("   • For highest accuracy: Use the best performing model above")
        else:
            print("   • For complex multi-class: LSTM recommended")
            print("   • For interpretability: Random Forest with feature importance")
        
        if classifier.num_channels >= 3:
            print("   • Multi-channel data: Consider ensemble methods")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()