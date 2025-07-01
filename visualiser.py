"""
Adaptive EMG Data Visualizer
============================
Automatically adapts to any configuration:
- Any number of channels (1-4)
- Any number of classes (including thinking)
- Dynamic feature plotting
- Interactive visualization options

Features:
- Auto-detects channels and classes from CSV
- Plots all available features per channel
- Color-coded labels with thinking support
- Time range filtering
- Export capabilities
- Statistical analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse
import os
import seaborn as sns

class AdaptiveEMGVisualizer:
    def __init__(self, csv_file):
        """
        Adaptive EMG Data Visualizer
        
        Automatically detects and adapts to:
        - Number of channels from CSV columns
        - Number of classes from label data
        - Available features per channel
        """
        self.csv_file = csv_file
        self.df = None
        self.num_channels = 0
        self.channels = []
        self.feature_types = ['filtered', 'envelope', 'filtered_rms', 'filtered_movavg', 'envelope_rms', 'envelope_movavg']
        self.available_features = {}
        self.class_names = []
        self.label_colors = {}
        
        # Load and analyze data
        self.load_and_analyze_data()
    
    def load_and_analyze_data(self):
        """Load CSV and automatically detect configuration"""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"📊 Loaded {len(self.df)} samples from {self.csv_file}")
            
            # Convert timestamp to relative time
            if 'timestamp' in self.df.columns:
                start_timestamp = self.df['timestamp'].iloc[0]
                self.df['time_sec'] = self.df['timestamp'] - start_timestamp
                self.duration = self.df['time_sec'].max()
                self.sample_rate = len(self.df) / self.duration
            else:
                print("⚠️ No timestamp column found, using sample indices")
                self.df['time_sec'] = np.arange(len(self.df)) * 0.002  # Assume 500Hz
                self.duration = self.df['time_sec'].max()
                self.sample_rate = 500
            
            # Auto-detect channels
            self.detect_channels()
            
            # Auto-detect classes
            self.detect_classes()
            
            # Show configuration
            self.show_configuration()
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            raise
    
    def detect_channels(self):
        """Auto-detect number of channels and available features"""
        print("🔍 Detecting channels and features...")
        
        # Find all channel columns
        channel_cols = [col for col in self.df.columns if col.startswith('ch') and '_' in col]
        
        # Extract channel numbers
        channel_numbers = set()
        for col in channel_cols:
            try:
                ch_num = int(col.split('_')[0].replace('ch', ''))
                channel_numbers.add(ch_num)
            except:
                continue
        
        self.channels = sorted(list(channel_numbers))
        self.num_channels = len(self.channels)
        
        # Detect available features for each channel
        self.available_features = {}
        for ch in self.channels:
            self.available_features[ch] = []
            for feature_type in self.feature_types:
                col_name = f'ch{ch}_{feature_type}'
                if col_name in self.df.columns:
                    self.available_features[ch].append(feature_type)
        
        print(f"✅ Detected {self.num_channels} channels: {self.channels}")
        for ch in self.channels:
            print(f"   📡 Channel {ch}: {len(self.available_features[ch])} features ({', '.join(self.available_features[ch])})")
    
    def detect_classes(self):
        """Auto-detect classes and assign colors"""
        if 'label' not in self.df.columns:
            print("⚠️ No label column found")
            self.class_names = []
            return
        
        self.class_names = sorted(self.df['label'].unique())
        print(f"✅ Detected {len(self.class_names)} classes: {', '.join(self.class_names)}")
        
        # Assign colors with special handling for thinking
        base_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
        
        self.label_colors = {}
        color_idx = 0
        
        # Assign thinking a specific color if present
        if 'thinking' in self.class_names:
            self.label_colors['thinking'] = '#95a5a6'  # Gray for thinking/rest
        
        # Assign colors to other classes
        for class_name in self.class_names:
            if class_name != 'thinking':
                self.label_colors[class_name] = base_colors[color_idx % len(base_colors)]
                color_idx += 1
        
        # Show class distribution
        print("\n📊 Class Distribution:")
        for class_name in self.class_names:
            count = sum(self.df['label'] == class_name)
            percentage = (count / len(self.df)) * 100
            print(f"   • {class_name.upper()}: {count:,} samples ({percentage:.1f}%)")
    
    def show_configuration(self):
        """Display detected configuration"""
        print(f"\n📋 DETECTED CONFIGURATION")
        print("=" * 50)
        print(f"📁 File: {os.path.basename(self.csv_file)}")
        print(f"📊 Samples: {len(self.df):,}")
        print(f"⏰ Duration: {self.duration:.1f} seconds")
        print(f"📡 Sample Rate: {self.sample_rate:.1f} Hz")
        print(f"🔧 Channels: {self.num_channels} ({', '.join([f'ch{ch}' for ch in self.channels])})")
        print(f"🎯 Classes: {len(self.class_names)} ({', '.join(self.class_names)})")
        
        total_features = sum(len(features) for features in self.available_features.values())
        print(f"📈 Total Features: {total_features}")
    
    def plot_single_feature(self, feature_type='filtered', start_time=None, end_time=None, 
                          figsize=None, save_plot=False, output_dir='plots'):
        """Plot single feature type across all channels"""
        
        # Filter time range if specified
        df_plot = self.filter_time_range(start_time, end_time)
        
        # Auto-size figure based on number of channels
        if figsize is None:
            height = max(8, 3 * self.num_channels + 2)
            figsize = (15, height)
        
        # Create subplots: one per channel + one comparison plot
        fig, axes = plt.subplots(self.num_channels + 1, 1, figsize=figsize)
        if self.num_channels == 1:
            axes = [axes[0], axes[1]]  # Ensure axes is always a list
        
        fig.suptitle(f'EMG Data - {feature_type.replace("_", " ").title()}', fontsize=16, fontweight='bold')
        
        # Plot each channel separately
        for i, ch in enumerate(self.channels):
            col_name = f'ch{ch}_{feature_type}'
            
            if col_name not in df_plot.columns:
                axes[i].text(0.5, 0.5, f'Feature {feature_type} not available for Channel {ch}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Channel {ch} - {feature_type.replace("_", " ").title()}')
                continue
            
            axes[i].set_title(f'Channel {ch} - {feature_type.replace("_", " ").title()}', fontweight='bold')
            
            # Plot with label segmentation if available
            if 'label' in df_plot.columns:
                self.plot_with_labels(axes[i], df_plot, 'time_sec', col_name)
            else:
                axes[i].plot(df_plot['time_sec'], df_plot[col_name], color='#3498db', linewidth=1)
            
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
            if i == 0 and 'label' in df_plot.columns:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Comparison plot (all channels together)
        axes[-1].set_title('All Channels Comparison', fontweight='bold')
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, ch in enumerate(self.channels):
            col_name = f'ch{ch}_{feature_type}'
            if col_name in df_plot.columns:
                color = colors[i % len(colors)]
                axes[-1].plot(df_plot['time_sec'], df_plot[col_name], 
                            label=f'Channel {ch}', color=color, alpha=0.7, linewidth=1)
        
        axes[-1].set_xlabel('Time (seconds)')
        axes[-1].set_ylabel('Amplitude')
        axes[-1].grid(True, alpha=0.3)
        axes[-1].legend()
        
        # Add statistics
        self.add_statistics_text(fig, df_plot, feature_type)
        
        plt.tight_layout()
        
        # Save if requested
        if save_plot:
            self.save_plot(fig, f'{feature_type}_channels', output_dir)
        
        plt.show()
    
    def plot_channel_comparison(self, channel, features=None, start_time=None, end_time=None, 
                              figsize=(15, 8), save_plot=False, output_dir='plots'):
        """Compare different features for a single channel"""
        
        if channel not in self.channels:
            print(f"❌ Channel {channel} not found. Available: {self.channels}")
            return
        
        if features is None:
            features = self.available_features[channel]
        
        # Filter available features
        available_features = [f for f in features if f in self.available_features[channel]]
        if not available_features:
            print(f"❌ No requested features available for channel {channel}")
            return
        
        # Filter time range
        df_plot = self.filter_time_range(start_time, end_time)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f'Channel {channel} - Feature Comparison', fontweight='bold')
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        for i, feature in enumerate(available_features):
            col_name = f'ch{channel}_{feature}'
            if col_name in df_plot.columns:
                color = colors[i % len(colors)]
                ax.plot(df_plot['time_sec'], df_plot[col_name], 
                       label=feature.replace('_', ' ').title(), 
                       color=color, alpha=0.8, linewidth=1)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_plot:
            self.save_plot(fig, f'ch{channel}_comparison', output_dir)
        
        plt.show()
    
    def plot_all_features_overview(self, start_time=None, end_time=None, save_plot=False, output_dir='plots'):
        """Create comprehensive overview of all channels and features"""
        
        df_plot = self.filter_time_range(start_time, end_time)
        
        # Calculate grid size
        max_features = max(len(features) for features in self.available_features.values())
        
        fig, axes = plt.subplots(self.num_channels, max_features, 
                               figsize=(4 * max_features, 3 * self.num_channels))
        
        if self.num_channels == 1:
            axes = axes.reshape(1, -1)
        if max_features == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Complete EMG Feature Overview', fontsize=16, fontweight='bold')
        
        for i, ch in enumerate(self.channels):
            for j, feature in enumerate(self.feature_types):
                if j >= max_features:
                    break
                    
                col_name = f'ch{ch}_{feature}'
                ax = axes[i, j]
                
                if feature in self.available_features[ch] and col_name in df_plot.columns:
                    # Plot with labels if available
                    if 'label' in df_plot.columns:
                        self.plot_with_labels(ax, df_plot, 'time_sec', col_name, show_legend=False)
                    else:
                        ax.plot(df_plot['time_sec'], df_plot[col_name], color='#3498db', linewidth=0.8)
                    
                    ax.set_title(f'Ch{ch} - {feature.replace("_", " ").title()}', fontsize=10)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12, color='gray')
                    ax.set_title(f'Ch{ch} - {feature.replace("_", " ").title()}', fontsize=10)
                
                # Remove labels for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_plot:
            self.save_plot(fig, 'complete_overview', output_dir)
        
        plt.show()
    
    def plot_class_analysis(self, feature_type='filtered', save_plot=False, output_dir='plots'):
        """Analyze and plot class-specific patterns"""
        
        if 'label' not in self.df.columns:
            print("❌ No labels available for class analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Class Analysis - {feature_type.replace("_", " ").title()}', fontsize=16, fontweight='bold')
        
        # 1. Class duration distribution
        axes[0, 0].set_title('Class Duration Distribution')
        class_durations = []
        class_labels = []
        
        # Calculate continuous segments for each class
        df_temp = self.df.copy()
        df_temp['label_change'] = df_temp['label'] != df_temp['label'].shift(1)
        df_temp['segment'] = df_temp['label_change'].cumsum()
        
        for segment_id in df_temp['segment'].unique():
            segment_data = df_temp[df_temp['segment'] == segment_id]
            if len(segment_data) > 1:
                duration = segment_data['time_sec'].iloc[-1] - segment_data['time_sec'].iloc[0]
                class_durations.append(duration)
                class_labels.append(segment_data['label'].iloc[0])
        
        # Create box plot
        class_duration_data = []
        class_names_plot = []
        for class_name in self.class_names:
            durations = [d for d, l in zip(class_durations, class_labels) if l == class_name]
            if durations:
                class_duration_data.append(durations)
                class_names_plot.append(class_name)
        
        if class_duration_data:
            axes[0, 0].boxplot(class_duration_data, labels=class_names_plot)
            axes[0, 0].set_ylabel('Duration (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Class amplitude statistics for first channel
        if self.channels:
            ch = self.channels[0]
            col_name = f'ch{ch}_{feature_type}'
            
            if col_name in self.df.columns:
                axes[0, 1].set_title(f'Channel {ch} Amplitude by Class')
                
                class_data = []
                for class_name in self.class_names:
                    class_values = self.df[self.df['label'] == class_name][col_name].values
                    if len(class_values) > 0:
                        class_data.append(class_values)
                
                if class_data:
                    axes[0, 1].boxplot(class_data, labels=self.class_names)
                    axes[0, 1].set_ylabel('Amplitude')
                    axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Class transitions
        axes[1, 0].set_title('Class Transition Timeline')
        y_pos = 0
        for class_name in self.class_names:
            class_times = self.df[self.df['label'] == class_name]['time_sec']
            if len(class_times) > 0:
                axes[1, 0].scatter(class_times, [y_pos] * len(class_times), 
                                 c=self.label_colors[class_name], label=class_name, alpha=0.6, s=1)
                y_pos += 1
        
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Class')
        axes[1, 0].set_yticks(range(len(self.class_names)))
        axes[1, 0].set_yticklabels(self.class_names)
        axes[1, 0].legend()
        
        # 4. Class statistics table
        axes[1, 1].axis('off')
        stats_data = []
        for class_name in self.class_names:
            count = sum(self.df['label'] == class_name)
            percentage = (count / len(self.df)) * 100
            stats_data.append([class_name.upper(), f'{count:,}', f'{percentage:.1f}%'])
        
        table = axes[1, 1].table(cellText=stats_data,
                               colLabels=['Class', 'Samples', 'Percentage'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('Class Statistics')
        
        plt.tight_layout()
        
        if save_plot:
            self.save_plot(fig, 'class_analysis', output_dir)
        
        plt.show()
    
    def plot_with_labels(self, ax, df, x_col, y_col, show_legend=True):
        """Plot data with label-based color coding"""
        
        if 'label' not in df.columns:
            ax.plot(df[x_col], df[y_col], color='#3498db', linewidth=1)
            return
        
        # Create segments for continuous label periods
        df_temp = df.copy()
        df_temp['label_change'] = df_temp['label'] != df_temp['label'].shift(1)
        df_temp['segment'] = df_temp['label_change'].cumsum()
        
        plotted_labels = set()
        
        for segment_id in df_temp['segment'].unique():
            segment_data = df_temp[df_temp['segment'] == segment_id]
            if len(segment_data) > 0:
                label = segment_data['label'].iloc[0]
                color = self.label_colors.get(label, '#95a5a6')
                
                # Only show label in legend once
                legend_label = label.capitalize() if label not in plotted_labels and show_legend else ""
                if label not in plotted_labels:
                    plotted_labels.add(label)
                
                ax.plot(segment_data[x_col], segment_data[y_col], 
                       color=color, label=legend_label, alpha=0.8, linewidth=1)
    
    def filter_time_range(self, start_time, end_time):
        """Filter dataframe to specified time range"""
        df_filtered = self.df.copy()
        
        if start_time is not None or end_time is not None:
            if start_time is None:
                start_time = df_filtered['time_sec'].min()
            if end_time is None:
                end_time = df_filtered['time_sec'].max()
            
            mask = (df_filtered['time_sec'] >= start_time) & (df_filtered['time_sec'] <= end_time)
            df_filtered = df_filtered[mask].copy()
            print(f"📅 Filtered to {start_time:.1f}s - {end_time:.1f}s ({len(df_filtered)} samples)")
        
        return df_filtered
    
    def add_statistics_text(self, fig, df, feature_type):
        """Add statistics text box to figure"""
        
        stats_lines = [
            f"Duration: {df['time_sec'].max():.1f}s",
            f"Samples: {len(df):,}",
            f"Sample Rate: {len(df)/df['time_sec'].max():.1f} Hz",
            ""
        ]
        
        # Add per-channel statistics
        for ch in self.channels:
            col_name = f'ch{ch}_{feature_type}'
            if col_name in df.columns:
                data = df[col_name]
                stats_lines.extend([
                    f"Channel {ch}:",
                    f"  Mean: {data.mean():.1f}",
                    f"  Std: {data.std():.1f}",
                    f"  Range: {data.min():.0f} to {data.max():.0f}",
                    ""
                ])
        
        stats_text = "\n".join(stats_lines)
        
        fig.text(0.02, 0.02, stats_text, fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def save_plot(self, fig, name, output_dir):
        """Save plot to file"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/emg_{name}_{timestamp}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to: {filename}")
    
    def interactive_menu(self):
        """Interactive command-line interface"""
        
        print(f"\n🎯 Adaptive EMG Data Visualizer")
        print("=" * 50)
        
        while True:
            print(f"\nOptions:")
            print(f"1. Plot single feature across all channels")
            print(f"2. Compare features for one channel")
            print(f"3. Complete overview (all channels & features)")
            print(f"4. Class analysis")
            print(f"5. Plot specific time range")
            print(f"6. Show data statistics")
            print(f"7. Save plots")
            print(f"8. Exit")
            
            choice = input(f"\nEnter choice (1-8): ").strip()
            
            if choice == '1':
                print(f"Available features: {', '.join(self.feature_types)}")
                feature = input("Enter feature type: ").strip()
                if feature in self.feature_types:
                    self.plot_single_feature(feature)
                else:
                    print("❌ Invalid feature type")
            
            elif choice == '2':
                print(f"Available channels: {', '.join([str(ch) for ch in self.channels])}")
                try:
                    channel = int(input("Enter channel number: "))
                    if channel in self.channels:
                        print(f"Available features for channel {channel}: {', '.join(self.available_features[channel])}")
                        features_str = input("Enter features (comma-separated, or 'all'): ").strip()
                        if features_str.lower() == 'all':
                            features = self.available_features[channel]
                        else:
                            features = [f.strip() for f in features_str.split(',')]
                        self.plot_channel_comparison(channel, features)
                    else:
                        print("❌ Invalid channel")
                except ValueError:
                    print("❌ Invalid channel number")
            
            elif choice == '3':
                self.plot_all_features_overview()
            
            elif choice == '4':
                if self.class_names:
                    print(f"Available features: {', '.join(self.feature_types)}")
                    feature = input("Enter feature type for analysis: ").strip()
                    if feature in self.feature_types:
                        self.plot_class_analysis(feature)
                    else:
                        print("❌ Invalid feature type")
                else:
                    print("❌ No class labels available")
            
            elif choice == '5':
                try:
                    start = float(input(f"Enter start time (0-{self.duration:.1f}s): "))
                    end = float(input(f"Enter end time ({start}-{self.duration:.1f}s): "))
                    feature = input("Enter feature type: ").strip()
                    if feature in self.feature_types:
                        self.plot_single_feature(feature, start_time=start, end_time=end)
                    else:
                        print("❌ Invalid feature type")
                except ValueError:
                    print("❌ Invalid time values")
            
            elif choice == '6':
                print(f"\n📊 Dataset Statistics:")
                print(f"Shape: {self.df.shape}")
                print(f"Columns: {list(self.df.columns)}")
                print(self.df.describe())
                
                if self.class_names:
                    print(f"\n📋 Class Distribution:")
                    for class_name in self.class_names:
                        count = sum(self.df['label'] == class_name)
                        percentage = (count / len(self.df)) * 100
                        print(f"   • {class_name.upper()}: {count:,} samples ({percentage:.1f}%)")
            
            elif choice == '7':
                feature = input("Enter feature type to save: ").strip()
                if feature in self.feature_types:
                    self.plot_single_feature(feature, save_plot=True)
                    if self.class_names:
                        self.plot_class_analysis(feature, save_plot=True)
                    self.plot_all_features_overview(save_plot=True)
                else:
                    print("❌ Invalid feature type")
            
            elif choice == '8':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice!")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Adaptive EMG Data Visualization Tool')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('--feature', '-f', default='filtered', 
                       help='Feature to plot (default: filtered)')
    parser.add_argument('--start', '-s', type=float, help='Start time (seconds)')
    parser.add_argument('--end', '-e', type=float, help='End time (seconds)')
    parser.add_argument('--save', action='store_true', help='Save plot to file')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Interactive mode')
    parser.add_argument('--channel', '-c', type=int, help='Specific channel for comparison')
    parser.add_argument('--overview', action='store_true', help='Show complete overview')
    parser.add_argument('--analysis', action='store_true', help='Show class analysis')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"❌ File not found: {args.csv_file}")
        return
    
    # Initialize visualizer
    visualizer = AdaptiveEMGVisualizer(args.csv_file)
    
    if args.interactive:
        visualizer.interactive_menu()
    elif args.overview:
        visualizer.plot_all_features_overview(start_time=args.start, end_time=args.end, save_plot=args.save)
    elif args.analysis:
        visualizer.plot_class_analysis(feature_type=args.feature, save_plot=args.save)
    elif args.channel:
        visualizer.plot_channel_comparison(args.channel, start_time=args.start, end_time=args.end, save_plot=args.save)
    else:
        visualizer.plot_single_feature(feature_type=args.feature, start_time=args.start, 
                                     end_time=args.end, save_plot=args.save)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("🎯 Adaptive EMG Data Visualizer")
        print("=" * 40)
        print("Usage examples:")
        print("  python visualizer.py data.csv")
        print("  python visualizer.py data.csv --feature envelope") 
        print("  python visualizer.py data.csv --start 10 --end 30")
        print("  python visualizer.py data.csv --interactive")
        print("  python visualizer.py data.csv --overview")
        print("  python visualizer.py data.csv --analysis")
        print("  python visualizer.py data.csv --channel 1")
        print("  python visualizer.py data.csv --save")
        print(f"\nFeatures: filtered, envelope, filtered_rms, filtered_movavg, envelope_rms, envelope_movavg")
        print(f"Auto-adapts to any number of channels (1-4) and classes")
    else:
        main()