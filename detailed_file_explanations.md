# Detailed Python File Explanations 📚

## 1. `data_collection.py` - Interactive EMG Data Collection System

### 🎯 Purpose
Collects labeled EMG data for training machine learning models with full user configuration control.

### 🔧 Key Features

#### **User Configuration System**
```python
# User specifies their exact setup
num_additional_classes = 3  # e.g., "hello", "yes", "no"
num_channels = 2           # Number of EMG channels
class_names = ["hello", "yes", "no", "thinking"]  # Auto-adds "thinking"
```

#### **Adaptive Feature Enhancement Pipeline**
- **Raw ESP32 Input**: `ch1_filtered, ch1_envelope, ch2_filtered, ch2_envelope`
- **Enhanced Output**: 6 features per channel
  ```python
  # For each channel:
  ch1_filtered         # Direct from ESP32
  ch1_envelope         # Direct from ESP32  
  ch1_filtered_rms     # Real-time RMS calculation
  ch1_filtered_movavg  # Moving average
  ch1_envelope_rms     # Envelope RMS
  ch1_envelope_movavg  # Envelope moving average
  ```

#### **Intelligent State Machine**
```python
# Randomized command presentation
THINKING (1-5s random) → COMMAND (1.5s fixed) → THINKING (1-5s) → ...

# Example timeline:
"thinking" → "hello" → "thinking" → "yes" → "thinking" → "no" → ...
```

#### **Real-Time Signal Quality Validation**
- **Saturation Detection**: Checks if signal is clipping (>4090 or <5)
- **Dead Signal Detection**: Ensures signal variance >5
- **Noise Detection**: Flags excessive noise (std >1500)

#### **Configurable Output Format**
```csv
timestamp,sample_num,ch1_filtered,ch1_envelope,ch1_filtered_rms,ch1_filtered_movavg,ch1_envelope_rms,ch1_envelope_movavg,ch2_filtered,ch2_envelope,ch2_filtered_rms,ch2_filtered_movavg,ch2_envelope_rms,ch2_envelope_movavg,label
```

### 🚀 Usage Flow
1. **Interactive Setup**: User specifies classes and channels
2. **ESP32 Validation**: Tests data format compatibility  
3. **Real-Time Collection**: GUI displays commands, collects enhanced data
4. **Quality Monitoring**: Continuous signal validation
5. **Labeled Export**: Saves enhanced CSV with configuration metadata

---

## 2. `visualiser.py` - Adaptive EMG Data Visualization

### 🎯 Purpose
Automatically adapts to any EMG data configuration and provides comprehensive visualization tools.

### 🔧 Key Features

#### **Auto-Detection System**
```python
# Automatically detects from CSV:
channels = [1, 2, 3, 4]  # From column names like 'ch1_filtered'
classes = ["hello", "yes", "no", "thinking"]  # From 'label' column
features = ['filtered', 'envelope', 'rms', 'movavg']  # Available features
```

#### **Multi-Scale Visualization**
1. **Single Feature Across Channels**
   ```python
   plot_single_feature('filtered')  # Shows ch1_filtered, ch2_filtered, etc.
   ```

2. **Channel Feature Comparison**
   ```python
   plot_channel_comparison(channel=1)  # All features for channel 1
   ```

3. **Complete Overview**
   ```python
   plot_all_features_overview()  # Grid view of all channels × all features
   ```

4. **Class-Specific Analysis**
   ```python
   plot_class_analysis()  # Class duration, amplitude, transitions
   ```


#### **Interactive Features**
- **Time Range Filtering**: `plot_single_feature(start_time=10, end_time=30)`
- **Command Line Interface**: `python visualiser.py data.csv --interactive`
- **Export Capabilities**: High-resolution PNG exports with metadata


---

## 3. `model_training.py` - Adaptive Multi-Model Training Pipeline

### 🎯 Purpose
Trains multiple machine learning models that automatically adapt to different class and channel configurations.

### 🔧 Key Features

#### **Configuration-Aware Architecture**
```python
# Models automatically scale based on problem complexity
def get_adaptive_model_configs():
    if num_classes <= 3:
        # Simple architecture for binary/ternary
        lstm_units = [64, 32]
        dense_units = 64
    else:
        # Complex architecture for multi-class
        lstm_units = [128, 64] 
        dense_units = 128
        # Uses Bidirectional LSTM + GlobalAveragePooling
```

#### **Feature Processing Pipeline**
1. **Statistical Features** (for classical models)
   ```python
   # 10 statistical features per enhanced feature
   for each_channel_feature:
       mean, std, min, max, median, p25, p75, energy, power, zero_crossings
   
   # Total: num_channels × 6 features × 10 stats = 60+ statistical features
   ```

2. **Temporal Features** (for LSTM)
   ```python
   # Keeps time series structure
   X_shape = (num_windows, window_size, num_enhanced_features)
   # e.g., (1000, 500, 12) for 2-channel system
   ```

#### **Adaptive Model Portfolio**
1. **Random Forest**
   ```python
   n_estimators = min(200, 50 * num_channels)  # More trees for more channels
   max_depth = max(10, 5 * num_classes)        # Deeper for more classes
   class_weight = 'balanced' if num_classes > 2 else None
   ```

2. **LSTM Architecture**
   ```python
   if num_classes <= 3:
       # Standard LSTM
       model = Sequential([
           LSTM(lstm_units_1, return_sequences=True),
           LSTM(lstm_units_2),
           Dense(dense_units, activation='relu'),
           Dense(num_classes, activation='softmax')
       ])
   else:
       # Bidirectional LSTM for complex problems
       model = Sequential([
           Bidirectional(LSTM(lstm_units_1, return_sequences=True)),
           Bidirectional(LSTM(lstm_units_2, return_sequences=True)),
           GlobalAveragePooling1D(),
           Dense(dense_units * 2, activation='relu'),
           Dense(num_classes, activation='softmax')
       ])
   ```

#### **Intelligent Model Saving**
```python
# Adaptive folder structure
model_folder = f"{model_name}_{num_classes}class_{num_channels}ch"
# e.g., "RandomForest_3class_2ch", "LSTM_4class_3ch"

# Saves complete model package:
{
    'model': trained_model,
    'scaler': feature_scaler,
    'label_encoder': class_encoder,
    'config': user_configuration,
    'window_size': 500,
    'feature_names': ['ch1_filtered', 'ch1_envelope', ...],
    'adaptive_params': architecture_settings
}
```

### 🚀 Training Process
1. **User Configuration**: Specify expected classes and channels
2. **Data Validation**: Ensures CSV matches configuration
3. **Window Creation**: Sliding windows with majority vote labeling
4. **Multi-Model Training**: Trains 4-5 models with adaptive parameters
5. **Performance Comparison**: Generates accuracy heatmaps and confusion matrices
6. **Model Export**: Saves all models with configuration metadata

---

## 4. `inference.py` - Real-Time Configurable Classification System

### 🎯 Purpose
Performs real-time EMG classification with support for both trained models and threshold-based classification.

### 🔧 Key Features

#### **Dual Classification Modes**

##### **1. Trained Model Mode**
```python
# Uses pre-trained models (Random Forest, LSTM, etc.)
compatible_models = find_models_matching(num_classes, num_channels)
selected_model = user_select_from_compatible()
```

##### **2. Threshold-Based Mode**
```python
# Simple amplitude-based classification
def setup_threshold_model():
    feature_to_threshold = "ch1_envelope"  # User selects
    
    # Binary classification example:
    if amplitude >= threshold_value:
        prediction = "hello"        # Active command
        confidence = calculate_confidence(amplitude, threshold)
    else:
        prediction = "thinking"     # Rest state
        confidence = calculate_confidence(amplitude, threshold)
```


#### **Threshold Classification Logic**

##### **Binary Classification (2 classes)**
```python
# Example: "hello" vs "thinking"
if signal_amplitude >= threshold:
    prediction = "hello"
    confidence = min(0.99, 0.5 + (amplitude - threshold) / threshold)
else:
    prediction = "thinking"  
    confidence = min(0.99, 0.5 + (threshold - amplitude) / threshold)
```

---

## 🔄 Complete System Integration

### **Data Flow Pipeline**
```
ESP32 → Raw EMG (2×channels) → Python Enhancement (6×channels features) 
→ ML Model → Class Prediction → Real-time Display → Performance Evaluation
```

### **Configuration Consistency**
```python
# All files use same configuration format
config = {
    'num_classes': 3,
    'num_channels': 2, 
    'class_names': ['hello', 'yes', 'thinking'],
    'feature_names': ['ch1_filtered', 'ch1_envelope', 'ch1_filtered_rms', ...]
}
```

### **Feature Enhancement Consistency**
```python
# Identical enhancement in collection and inference
def enhance_raw_features(raw_data):
    # Same RMS and moving average calculations
    # Same buffer management
    # Same feature ordering
    # Ensures training/inference compatibility
```

This comprehensive system provides maximum flexibility while maintaining perfect consistency between data collection, training, and real-time inference phases.