"""
Configurable EMG Data Collection System - Updated
=================================================
Interactive setup for flexible EMG data collection with user-defined:
- Number of additional classes (1-5, 'thinking' always included)
- Class names for additional classes
- Number of channels
- Collection duration
- Command timing

Features:
- Interactive configuration
- Automatic 'thinking' class inclusion
- Signal quality validation
- Progress tracking with repetition counts
- Flexible channel support
- Enhanced timing options
"""

import serial
import time
import csv
import random
import tkinter as tk
import numpy as np
from datetime import datetime

class EMGDataCollector:
    def __init__(self):
        self.serial_port = '/dev/cu.usbserial-110'
        self.baud_rate = 115200
        
        # Configuration (will be set by user)
        self.duration = 300
        self.class_names = []  # Will include 'thinking' automatically
        self.num_channels = 2
        self.display_duration = 2.5
        self.thinking_duration_range = (3.0, 5.0)
        self.target_reps_per_class = 15
        
        # Data tracking
        self.command_counts = {}
        self.sample_count = 0
        self.ser = None
        self.gui = None
        
        # Buffers (will be created based on num_channels)
        self.channel_buffers = {}
        
    def setup_configuration(self):
        """Interactive setup for data collection parameters"""
        print("🧠 EMG Data Collection Configuration")
        print("=" * 50)
        print("Note: 'thinking' class is automatically included as the rest state")
        
        # Get number of additional classes (thinking is always included)
        while True:
            try:
                num_additional_classes = int(input("How many additional classes do you want to collect? (1-5, 'thinking' is always included): "))
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
        
        # Initialize command counts (thinking is now included)
        self.command_counts = {class_name: 0 for class_name in self.class_names}
        
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
        
        # Get collection duration
        while True:
            try:
                duration_minutes = float(input(f"\nCollection duration in minutes? (5-30): "))
                if 5 <= duration_minutes <= 30:
                    self.duration = int(duration_minutes * 60)
                    break
                else:
                    print("❌ Please enter a duration between 5 and 30 minutes")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Set fixed timing parameters
        self.display_duration = 1.5  # Fixed command display duration
        self.thinking_duration_range = (1.0, 5.0)  # Random rest time between 1-5 seconds
        
        print(f"\n⏱️  Timing Configuration (Fixed):")
        print(f"   • Command display: {self.display_duration} seconds")
        print(f"   • Rest (thinking) time: {self.thinking_duration_range[0]}-{self.thinking_duration_range[1]} seconds (random)")
        
        # Setup channel buffers based on number of channels
        self.setup_channel_buffers()
        
        # Display configuration summary
        self.show_configuration_summary()
    
    def setup_channel_buffers(self):
        """Setup buffers for each channel based on configuration"""
        self.channel_buffers = {}
        
        for ch in range(1, self.num_channels + 1):
            self.channel_buffers[f'ch{ch}_filtered'] = []
            self.channel_buffers[f'ch{ch}_envelope'] = []
    
    def show_configuration_summary(self):
        """Display the configuration summary"""
        print("\n" + "=" * 50)
        print("📋 CONFIGURATION SUMMARY")
        print("=" * 50)
        
        # Separate active classes from thinking
        active_classes = [name for name in self.class_names if name != 'thinking']
        
        print(f"🎯 Classes: {', '.join([name.upper() for name in self.class_names])}")
        print(f"   • Active classes: {', '.join([name.upper() for name in active_classes])}")
        print(f"   • Rest state: THINKING (automatic)")
        print(f"📡 Channels: {self.num_channels}")
        print(f"⏰ Duration: {self.duration//60} minutes ({self.duration} seconds)")
        print(f"🔄 Target reps per class: {self.target_reps_per_class} (for each class including thinking)")
        print(f"⏱️  Command duration: {self.display_duration} seconds")
        print(f"💭 Thinking duration: {self.thinking_duration_range[0]}-{self.thinking_duration_range[1]} seconds")
        
        total_classes = len(self.class_names)  # Now includes thinking
        print(f"📊 Expected total commands: {total_classes * self.target_reps_per_class}")
        
        expected_data_features = self.num_channels * 6  # Each channel: filtered, envelope, rms, movavg, envelope_rms, envelope_movavg
        print(f"🔧 Features per sample: {expected_data_features} ({self.num_channels} channels × 6 features)")
        print(f"📋 Total classes: {total_classes} ({len(active_classes)} active + 1 thinking)")
        
        confirm = input(f"\nProceed with this configuration? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Configuration cancelled. Restarting setup...")
            self.setup_configuration()
            return
        
        # Show ESP32 data format expectation
        expected_format = ','.join([f'ch{i+1}_filtered,ch{i+1}_envelope' for i in range(self.num_channels)])
        print(f"\n📡 Expected ESP32 format: {expected_format}")
        print("   Make sure your ESP32 code sends data in this exact order!")
    
    def moving_average(self, signal, window_size=20):
        """Compute moving average"""
        if len(signal) < window_size:
            return np.mean(signal) if len(signal) > 0 else 0
        return np.mean(signal[-window_size:])

    def compute_rms(self, signal, window_size=20):
        """Compute RMS"""
        if len(signal) < window_size:
            return np.sqrt(np.mean(np.square(signal))) if len(signal) > 0 else 0
        return np.sqrt(np.mean(np.square(signal[-window_size:])))
    
    def validate_signal_quality(self, channel_data, channel_name):
        """Validate signal quality for a channel"""
        if len(channel_data) < 50:
            return True, "Insufficient data"
        
        recent_data = channel_data[-50:]
        
        # Check for saturation
        if max(recent_data) >= 4090 or min(recent_data) <= 5:
            return False, f"{channel_name}: Signal saturated"
        
        # Check for dead signal
        if np.std(recent_data) < 5:
            return False, f"{channel_name}: Signal too flat"
        
        # Check for excessive noise
        if np.std(recent_data) > 1500:
            return False, f"{channel_name}: Signal too noisy"
        
        return True, "Good signal"
    
    def create_gui(self):
        """Create the data collection GUI"""
        class ConfigurablePromptGUI:
            def __init__(self, collector):
                self.collector = collector
                self.root = tk.Tk()
                
                total_classes = len(collector.class_names)
                active_classes = len([name for name in collector.class_names if name != 'thinking'])
                
                self.root.title(f"EMG Data Collection - {total_classes} Classes ({active_classes} active + thinking)")
                self.root.configure(bg="black")
                self.root.attributes("-fullscreen", True)
                
                # Exit button
                self.exit_btn = tk.Button(self.root, text="Exit (ESC)", command=self.root.quit, 
                                        bg="red", fg="white", font=("Arial", 12))
                self.exit_btn.pack(side="top", anchor="ne", padx=10, pady=10)
                
                # Configuration display
                active_class_names = [name for name in collector.class_names if name != 'thinking']
                config_info = (f"Classes: {', '.join([name.upper() for name in active_class_names])} + THINKING | "
                             f"Channels: {collector.num_channels} | Duration: {collector.duration//60}min")
                self.config_label = tk.Label(self.root, text=config_info, font=("Arial", 14),
                                           fg="cyan", bg="black")
                self.config_label.pack(side="top", pady=10)
                
                # Main command display
                self.label = tk.Label(self.root, text="READY", font=("Arial", 80), 
                                     fg="white", bg="black")
                self.label.pack(expand=True)
                
                # Progress display
                self.progress_label = tk.Label(self.root, text="", font=("Arial", 16),
                                             fg="yellow", bg="black")
                self.progress_label.pack(pady=10)
                
                # Status display
                self.status_label = tk.Label(self.root, text="", font=("Arial", 16), 
                                           fg="yellow", bg="black")
                self.status_label.pack(side="bottom", pady=20)
                
                self.current_label = "thinking"
                
                # Keyboard shortcuts
                self.root.bind('<Escape>', lambda e: self.root.quit())
                self.root.bind('<space>', lambda e: self.root.quit())
                self.root.focus_set()

            def update_label(self, text, color="white"):
                self.current_label = text.lower()
                display_text = text.upper() if text != "thinking" else "REST"
                self.label.config(text=display_text, fg=color)
                self.root.update()

            def update_status(self, status):
                self.status_label.config(text=status)
                self.root.update()
            
            def update_progress(self, progress_text):
                self.progress_label.config(text=progress_text)
                self.root.update()
        
        self.gui = ConfigurablePromptGUI(self)
        return self.gui
    
    def connect_serial(self):
        """Connect to the EMG device"""
        print(f"🔌 Connecting to {self.serial_port}...")
        
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            time.sleep(2)
            print("✅ Serial connection established!")
            
            # Test data format
            print("🧪 Testing data format...")
            for i in range(5):
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line and ',' in line:
                        parts = line.split(',')
                        expected_values = self.num_channels * 2  # filtered + envelope per channel
                        print(f"   Sample {i+1}: {len(parts)} values (expected {expected_values})")
                        if len(parts) == expected_values:
                            print("   ✅ Correct format")
                        else:
                            print(f"   ⚠️  Expected {expected_values} values, got {len(parts)}")
                time.sleep(0.2)
            
            return True
            
        except Exception as e:
            print(f"❌ Serial connection failed: {e}")
            return False
    
    def generate_csv_header(self):
        """Generate CSV header based on configuration"""
        header = ['timestamp', 'sample_num']
        
        # Add columns for each channel
        for ch in range(1, self.num_channels + 1):
            header.extend([
                f'ch{ch}_filtered', f'ch{ch}_envelope', 
                f'ch{ch}_filtered_rms', f'ch{ch}_filtered_movavg',
                f'ch{ch}_envelope_rms', f'ch{ch}_envelope_movavg'
            ])
        
        header.append('label')
        return header
    
    def run_collection(self):
        """Run the main data collection loop"""
        # Setup
        if not self.connect_serial():
            return False
        
        gui = self.create_gui()
        
        start_time = time.time()
        last_switch = time.time()
        current_state = "thinking"
        current_label = "thinking"
        
        # Generate filename with timestamp and configuration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_classes = len(self.class_names)
        active_classes = total_classes - 1  # Subtract thinking
        config_suffix = f"{total_classes}class_{self.num_channels}ch"
        filename = f'emg_data_{config_suffix}_{timestamp}.csv'
        
        last_gui_update = time.time()
        last_quality_check = time.time()
        
        print(f"\n🚀 Starting data collection...")
        print(f"📁 Saving to: {filename}")
        print(f"🎯 Collecting: {', '.join([name.upper() for name in self.class_names])}")
        print("🎮 Press ESC or SPACE in GUI to stop early")
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.generate_csv_header())
            
            while time.time() - start_time < self.duration:
                now = time.time()
                elapsed = now - start_time
                
                # Update GUI and handle state changes
                if now - last_gui_update > 0.5:
                    # State transitions
                    if current_state == "thinking":
                        thinking_duration = random.uniform(*self.thinking_duration_range)
                        if now - last_switch > thinking_duration:
                            # Switch to a random active command (exclude thinking)
                            active_commands = [cmd for cmd in self.class_names if cmd != 'thinking']
                            
                            if active_commands:
                                current_state = "command"
                                current_label = random.choice(active_commands)
                                gui.update_label(current_label, "lime")
                                last_switch = now
                                self.command_counts[current_label] += 1
                                print(f"💭 {current_label.upper()} (#{self.command_counts[current_label]})")
                    
                    elif current_state == "command":
                        if now - last_switch > self.display_duration:
                            current_state = "thinking"
                            current_label = "thinking"
                            self.command_counts['thinking'] += 1
                            gui.update_label("thinking", "yellow")
                            last_switch = now
                    
                    # Update progress display (show command counts and elapsed time)
                    total_commands = sum(self.command_counts[cmd] for cmd in self.class_names)
                    progress_text = f"Commands: {total_commands} | "
                    
                    class_progress = []
                    for class_name in self.class_names:
                        count = self.command_counts[class_name]
                        display_name = "REST" if class_name == "thinking" else class_name.upper()
                        class_progress.append(f"{display_name}:{count}")
                    progress_text += " | ".join(class_progress)
                    gui.update_progress(progress_text)
                    
                    # Update status display
                    remaining = self.duration - elapsed
                    status_text = f"⏰ {int(elapsed//60):02d}:{int(elapsed%60):02d} / {int(self.duration//60):02d}:{int(self.duration%60):02d} | "
                    status_text += f"Samples: {self.sample_count:,}"
                    gui.update_status(status_text)
                    
                    last_gui_update = now
                    
                    # Collection continues for the full duration (no early stopping based on targets)
                    try:
                        gui.root.update_idletasks()
                    except:
                        print("🛑 GUI closed by user")
                        break
                
                # Signal quality check (every 10 seconds)
                if now - last_quality_check > 10.0:
                    for ch_name, buffer in self.channel_buffers.items():
                        if 'filtered' in ch_name:  # Only check filtered signals
                            quality_ok, quality_msg = self.validate_signal_quality(buffer, ch_name)
                            if not quality_ok:
                                print(f"⚠️ {quality_msg}")
                    last_quality_check = now
                
                # Read and process serial data
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line and ',' in line:
                        parts = line.split(',')
                        expected_values = self.num_channels * 2
                        
                        if len(parts) == expected_values:
                            try:
                                # Parse channel data based on configuration
                                channel_data = {}
                                for ch in range(self.num_channels):
                                    filtered_val = int(parts[ch * 2].strip())
                                    envelope_val = int(parts[ch * 2 + 1].strip())
                                    
                                    ch_num = ch + 1
                                    channel_data[f'ch{ch_num}_filtered'] = filtered_val
                                    channel_data[f'ch{ch_num}_envelope'] = envelope_val
                                    
                                    # Add to buffers
                                    self.channel_buffers[f'ch{ch_num}_filtered'].append(filtered_val)
                                    self.channel_buffers[f'ch{ch_num}_envelope'].append(envelope_val)
                                    
                                    # Keep buffers manageable
                                    if len(self.channel_buffers[f'ch{ch_num}_filtered']) > 1000:
                                        self.channel_buffers[f'ch{ch_num}_filtered'] = self.channel_buffers[f'ch{ch_num}_filtered'][-500:]
                                        self.channel_buffers[f'ch{ch_num}_envelope'] = self.channel_buffers[f'ch{ch_num}_envelope'][-500:]
                                
                                # Compute enhanced features for each channel
                                enhanced_features = {}
                                for ch in range(1, self.num_channels + 1):
                                    filtered_buffer = self.channel_buffers[f'ch{ch}_filtered']
                                    envelope_buffer = self.channel_buffers[f'ch{ch}_envelope']
                                    
                                    enhanced_features[f'ch{ch}_filtered_rms'] = self.compute_rms(filtered_buffer)
                                    enhanced_features[f'ch{ch}_filtered_movavg'] = self.moving_average(filtered_buffer)
                                    enhanced_features[f'ch{ch}_envelope_rms'] = self.compute_rms(envelope_buffer)
                                    enhanced_features[f'ch{ch}_envelope_movavg'] = self.moving_average(envelope_buffer)
                                
                                # Write to CSV
                                row = [now, self.sample_count]
                                for ch in range(1, self.num_channels + 1):
                                    row.extend([
                                        channel_data[f'ch{ch}_filtered'],
                                        channel_data[f'ch{ch}_envelope'],
                                        enhanced_features[f'ch{ch}_filtered_rms'],
                                        enhanced_features[f'ch{ch}_filtered_movavg'],
                                        enhanced_features[f'ch{ch}_envelope_rms'],
                                        enhanced_features[f'ch{ch}_envelope_movavg']
                                    ])
                                row.append(current_label)
                                
                                writer.writerow(row)
                                self.sample_count += 1
                                
                            except ValueError:
                                continue  # Skip malformed data
                
                except Exception as e:
                    print(f"⚠️ Serial read error: {e}")
                    time.sleep(0.1)
        
        self.ser.close()
        self.show_final_statistics(filename, elapsed)
        return True
    
    def show_final_statistics(self, filename, elapsed):
        """Display final collection statistics"""
        print("\n" + "=" * 60)
        print("📊 COLLECTION COMPLETE!")
        print("=" * 60)
        print(f"📁 File: {filename}")
        print(f"⏰ Duration: {elapsed:.1f} seconds")
        print(f"📈 Samples collected: {self.sample_count:,}")
        print(f"📊 Sample rate: {self.sample_count/elapsed:.1f} Hz")
        
        # Calculate data distribution
        with open(filename, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            
            print(f"\n📋 Data distribution:")
            for class_name in self.class_names:
                count = sum(1 for line in lines if line.strip().endswith(f',{class_name}'))
                percentage = count / self.sample_count * 100 if self.sample_count > 0 else 0
                display_name = "THINKING (REST)" if class_name == "thinking" else class_name.upper()
                print(f"   • {display_name}: {count:,} samples ({percentage:.1f}%)")
        
        # Show class breakdown
        active_classes = [name for name in self.class_names if name != 'thinking']
        print(f"\n🎯 Class Summary:")
        print(f"   • Active classes: {len(active_classes)} ({', '.join([name.upper() for name in active_classes])})")
        print(f"   • Rest state: 1 (THINKING)")
        print(f"   • Total classes: {len(self.class_names)}")
        
        # Show command statistics
        print(f"\n📊 Command Statistics:")
        total_commands = sum(self.command_counts.values())
        for class_name in self.class_names:
            count = self.command_counts[class_name]
            percentage = count / total_commands * 100 if total_commands > 0 else 0
            display_name = "THINKING (REST)" if class_name == "thinking" else class_name.upper()
            print(f"   • {display_name}: {count} commands ({percentage:.1f}%)")
        
        print(f"\n🔧 Features: {self.num_channels} channels × 6 features = {self.num_channels * 6} total features")
        print(f"⏰ Collection ran for full duration: {elapsed:.1f} seconds")
        print("=" * 60)


def main():
    """Main function to run configurable EMG data collection"""
    print("🧠 Configurable EMG Data Collection System")
    print("=" * 60)
    print("Note: 'thinking' class is automatically included as the rest state")
    print("You can add 1-5 additional active classes")
    
    collector = EMGDataCollector()
    
    try:
        # Interactive configuration
        collector.setup_configuration()
        
        # Run collection
        success = collector.run_collection()
        
        if success:
            print("✅ Data collection completed successfully!")
        else:
            print("❌ Data collection failed")
            
    except KeyboardInterrupt:
        print("\n🛑 Collection stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()