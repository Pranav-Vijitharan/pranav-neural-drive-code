# pranav-neural-drive-code

A complete end-to-end system for classifying subvocal commands using EMG signals with EXG Pills and ESP32. This project enables real-time detection of imagined or lightly subvocalized words through facial/neck muscle activity.

## 🎯 Project Overview

This system classifies mental/subvocal commands by detecting EMG signals from face and neck muscles. Users can imagine or lightly subvocalize words, and the system predicts the intended command in real-time.

### Key Features
- **Configurable Classes**: Support for 1-5 additional classes plus automatic "thinking" (rest) state
- **Multi-Channel Support**: 1-4 EMG channels with adaptive processing
- **Real-Time Inference**: Continuous prediction every 200ms
- **Complete Pipeline**: Data collection → Training → Inference
- **Multiple ML Models**: Random Forest, SVM, Gradient Boosting, LSTM
- **Adaptive Architecture**: Automatically scales to your configuration

## 🔧 Hardware Setup

### Core Components
- **2× EXG Pill** - Bioamplifier for EMG signal conditioning
- **ESP32-WROOM-32** - Microcontroller for data acquisition
- **6× EMG Electrodes** - 3 per EXG Pill (IN+, IN-, REF)
- **Jumper Wires** - For connections
- **Breadboard** (optional) - For easy prototyping

### EXG Pill Documentation
Learn more about the EXG Pill: [UpsideDown Labs Documentation](https://docs.upsidedownlabs.tech/hardware/bioamp/bioamp-exg-pill/index.html)

### Electrode Placement (2-Channel Setup)

Each EXG Pill requires 3 electrodes:
- **IN+** → Signal electrode (active muscle area)  
- **IN-** → Reference electrode (near IN+ on same muscle)
- **REF** → Shared ground reference (neutral bony area)

**Example Placement:**
- **Channel 1 (Jaw)**:
  - IN+ on jaw muscle
  - IN- near jaw muscle
- **Channel 2 (Throat)**:
  - IN+ on throat muscle  
  - IN- near throat muscle
- **REF (Shared)**: Back of ear or collarbone


## 🚀 Quick Start Guide

### 1. Hardware Setup
1. Connect EXG Pills to ESP32 as shown in wiring diagram
2. Place electrodes on face/neck muscles
3. Upload firmware code to ESP32
4. Verify serial output shows: `ch1_filtered,ch1_envelope,ch2_filtered,ch2_envelope`


### 3. Data Collection
```bash
python data_collection.py
```
**Interactive Setup:**
- Specify number of additional classes (1-5, "thinking" auto-included)
- Enter class names (e.g., "hello", "yes", "no")
- Set number of channels (1-4)
- Configure collection duration (5-30 minutes)

**Example Session:**
```
Classes: YES, THINKING
Duration: 10 minutes
Output: emg_data_3class_2ch_20241202_143022.csv
```

### 4. Data Visualization
```bash
python visualiser.py your_data.csv --interactive
```
**Features:**
- Auto-detects channels and classes from CSV
- Interactive plotting options
- Signal quality analysis
- Class distribution visualization

### 5. Model Training
```bash
python model_training.py
```
**Training Process:**
- User specifies expected configuration
- Validates data compatibility
- Trains multiple models (Random Forest, SVM, LSTM, etc.)
- Generates performance comparisons
- Saves models with adaptive naming

### 6. Real-Time Inference
```bash
python inference.py
```
**Inference Setup:**
- Specify your configuration (classes, channels)
- Select compatible trained model
- Real-time continuous prediction
- Performance evaluation and logging

## 🧪 Data Collection Protocol

### Session Design
The system uses a randomized prompt sequence:
- **Command Duration**: 1.5 seconds (display time)
- **Rest Duration**: 1-5 seconds (random)
- **Classes**: Your custom classes + "thinking" (rest state)

### Example Timeline
```
THINKING (3s) → NO (1.5s) → THINKING (2s) → YES (1.5s) → THINKING (4s) → ...
```

### Data Quality
- **Sampling Rate**: 500 Hz
- **Signal Validation**: Automatic saturation and noise detection
- **Label Purity**: >70% majority vote for sliding windows
- **Feature Enhancement**: 6 features per channel (filtered, envelope, RMS, moving average, etc.)

## 🤖 Machine Learning Pipeline

### Feature Extraction
Each channel provides 6 enhanced features:
- `ch{n}_filtered` - Bandpass filtered EMG (74.5-149.5 Hz)
- `ch{n}_envelope` - Signal envelope  
- `ch{n}_filtered_rms` - RMS of filtered signal
- `ch{n}_filtered_movavg` - Moving average of filtered signal
- `ch{n}_envelope_rms` - RMS of envelope
- `ch{n}_envelope_movavg` - Moving average of envelope

### Supported Models
1. **Random Forest** - Fast, interpretable, good baseline
2. **Gradient Boosting** - Robust ensemble method  
3. **SVM** - Support Vector Machine with RBF kernel
4. **Logistic Regression** - Linear classification
5. **LSTM** - Deep learning for temporal patterns

### Adaptive Architecture
Models automatically scale based on configuration:
- **Architecture depth** scales with number of classes
- **Feature complexity** scales with number of channels  
- **Training parameters** adapt to problem difficulty

## 📊 Performance Evaluation

### Real-Time Metrics
- **Overall Accuracy**: Correct predictions / Total predictions
- **Per-Class Accuracy**: Individual class performance
- **Confusion Matrix**: Detailed classification breakdown
- **Confidence Scores**: Model prediction confidence

### Evaluation Protocol
- **Continuous Prediction**: Every 200ms during testing
- **State-Aware Evaluation**: Accounts for command vs thinking states
- **Configurable Duration**: 30-120 second test sessions


### Factors Affecting Performance
- **Electrode Placement Quality**: Proper muscle targeting
- **Signal Strength**: Clear muscle activation patterns
- **Training Data Quality**: Consistent, clean recordings
- **User Consistency**: Repeatable subvocal patterns


## 📋 File Descriptions

### Core Scripts
- **`data_collection.py`**: Interactive EMG data collection with configurable classes and channels
- **`visualiser.py`**: Adaptive visualization tool that auto-detects data configuration  
- **`model_training.py`**: Complete training pipeline with multiple ML models
- **`inference.py`**: Real-time classification with configurable testing

### Hardware Code
- **`firmware_code.ino`**: ESP32 Arduino code for dual-channel EMG acquisition

### Output Files
- **`emg_data_*.csv`**: Raw EMG data with enhanced features
- **`adaptive_models/`**: Trained models organized by configuration  
- **`*_results.csv`**: Real-time testing performance data
- **`*_summary.json`**: Detailed evaluation metrics

## 🔬 Technical Specifications

### Signal Processing
- **Sampling Rate**: 500 Hz
- **Bandpass Filter**: 74.5-149.5 Hz (EMG frequency range)
- **Envelope Detection**: 64-sample moving average
- **Window Size**: 500 samples (1 second at 500 Hz)
- **Window Overlap**: 50%

### Real-Time Performance  
- **Prediction Latency**: <50ms
- **Update Rate**: 5 Hz (every 200ms)
- **Buffer Management**: Circular buffers for efficiency
- **Feature Enhancement**: Real-time computation


