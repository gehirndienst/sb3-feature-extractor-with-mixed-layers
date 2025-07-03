"""
Visualization Tools for Vibration Event Detection

This module provides comprehensive visualization capabilities for vibration
event detection analysis, including step-by-step visualization of the
detection process and interactive parameter tuning.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from typing import List, Optional, Dict, Any
import logging
from event_detector import EventDetector, DetectedEvent, EventDetectionConfig

logger = logging.getLogger(__name__)


class EventVisualization:
    """
    Comprehensive visualization tools for event detection analysis.
    
    Provides various plotting functions to understand the event detection
    process, including raw data visualization, energy profiles, threshold
    analysis, and individual event inspection.
    """
    
    def __init__(self, detector: EventDetector):
        """
        Initialize visualization with an EventDetector instance.
        
        Args:
            detector: EventDetector instance with loaded data
        """
        self.detector = detector
        self.fig = None
        self.axes = None
        
    def plot_detection_overview(self, figsize: tuple = (15, 10), save_path: Optional[str] = None) -> None:
        """
        Create comprehensive overview plot showing all detection steps.
        
        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the figure
        """
        if self.detector.data is None:
            raise ValueError("No data loaded in detector")
        
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Raw vibration data
        axes[0].plot(self.detector.time_axis, self.detector.data, 'b-', alpha=0.7, linewidth=0.5)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Raw Vibration Data')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Filtered data (if available)
        if self.detector.filtered_data is not None:
            axes[1].plot(self.detector.time_axis, self.detector.filtered_data, 'g-', alpha=0.7, linewidth=0.5)
            axes[1].set_ylabel('Filtered Amplitude')
            axes[1].set_title(f'Filtered Data ({self.detector.config.filter_lowcut}-{self.detector.config.filter_highcut} Hz)')
        else:
            axes[1].plot(self.detector.time_axis, self.detector.data, 'b-', alpha=0.7, linewidth=0.5)
            axes[1].set_ylabel('Amplitude')
            axes[1].set_title('Data (No Filtering Applied)')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Energy profile with threshold
        if self.detector.energy_profile is not None:
            axes[2].plot(self.detector.time_axis, self.detector.energy_profile, 'r-', linewidth=1)
            
            # Plot threshold line
            if self.detector.threshold is not None:
                axes[2].axhline(y=self.detector.threshold, color='k', linestyle='--', 
                               label=f'Threshold: {self.detector.threshold:.4f}')
            
            # Plot baseline
            if self.detector.baseline_mean is not None:
                axes[2].axhline(y=self.detector.baseline_mean, color='gray', linestyle=':', 
                               label=f'Baseline: {self.detector.baseline_mean:.4f}')
            
            axes[2].set_ylabel('Energy (RMS)')
            axes[2].set_title('Energy Profile with Detection Threshold')
            axes[2].legend()
        else:
            axes[2].text(0.5, 0.5, 'Energy profile not calculated', 
                        transform=axes[2].transAxes, ha='center')
            axes[2].set_title('Energy Profile (Not Calculated)')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Detected events
        axes[3].plot(self.detector.time_axis, self.detector.filtered_data or self.detector.data, 
                    'b-', alpha=0.3, linewidth=0.5, label='Data')
        
        # Highlight detected events
        if self.detector.detected_events:
            for i, event in enumerate(self.detector.detected_events):
                start_time = event.start_idx / self.detector.config.sampling_rate
                end_time = event.end_idx / self.detector.config.sampling_rate
                peak_time = event.peak_idx / self.detector.config.sampling_rate
                
                # Highlight event region
                axes[3].axvspan(start_time, end_time, alpha=0.3, color='red', 
                               label='Events' if i == 0 else "")
                
                # Mark peak
                peak_amplitude = (self.detector.filtered_data or self.detector.data)[event.peak_idx]
                axes[3].plot(peak_time, peak_amplitude, 'ro', markersize=4)
                
                # Add event number
                axes[3].text(peak_time, peak_amplitude, str(i+1), 
                           fontsize=8, ha='center', va='bottom')
        
        axes[3].set_ylabel('Amplitude')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_title(f'Detected Events (Total: {len(self.detector.detected_events)})')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved overview plot to {save_path}")
        
        plt.show()
    
    def plot_baseline_analysis(self, figsize: tuple = (12, 8), save_path: Optional[str] = None) -> None:
        """
        Plot detailed baseline and threshold analysis.
        
        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the figure
        """
        if self.detector.energy_profile is None:
            raise ValueError("Energy profile not calculated")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Energy profile histogram
        axes[0, 0].hist(self.detector.energy_profile, bins=50, alpha=0.7, color='blue', edgecolor='black')
        
        if self.detector.baseline_mean is not None:
            axes[0, 0].axvline(self.detector.baseline_mean, color='green', linestyle='-', 
                              label=f'Baseline Mean: {self.detector.baseline_mean:.4f}')
        
        if self.detector.threshold is not None:
            axes[0, 0].axvline(self.detector.threshold, color='red', linestyle='--', 
                              label=f'Threshold: {self.detector.threshold:.4f}')
        
        # Show percentile used for baseline
        percentile_value = np.percentile(self.detector.energy_profile, self.detector.config.baseline_percentile)
        axes[0, 0].axvline(percentile_value, color='orange', linestyle=':', 
                          label=f'{self.detector.config.baseline_percentile}th Percentile: {percentile_value:.4f}')
        
        axes[0, 0].set_xlabel('Energy')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Energy Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Baseline calculation explanation
        quiet_threshold = np.percentile(self.detector.energy_profile, self.detector.config.baseline_percentile)
        quiet_periods = self.detector.energy_profile <= quiet_threshold
        
        axes[0, 1].plot(self.detector.time_axis, self.detector.energy_profile, 'b-', alpha=0.7, label='Energy Profile')
        axes[0, 1].plot(self.detector.time_axis[quiet_periods], 
                       self.detector.energy_profile[quiet_periods], 'go', markersize=1, 
                       label=f'Quiet Periods ({self.detector.config.baseline_percentile}th percentile)')
        
        if self.detector.baseline_mean is not None:
            axes[0, 1].axhline(self.detector.baseline_mean, color='green', linestyle='-', 
                              label=f'Baseline Mean')
        
        if self.detector.threshold is not None:
            axes[0, 1].axhline(self.detector.threshold, color='red', linestyle='--', 
                              label=f'Detection Threshold')
        
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Energy')
        axes[0, 1].set_title('Baseline Calculation from Quiet Periods')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Threshold sensitivity analysis
        multipliers = np.linspace(1, 6, 50)
        thresholds = self.detector.baseline_mean + multipliers * self.detector.baseline_std
        event_counts = []
        
        for threshold in thresholds:
            above_threshold = self.detector.energy_profile > threshold
            # Simple event counting (number of threshold crossings / 2)
            crossings = np.sum(np.diff(above_threshold.astype(int)) == 1)
            event_counts.append(crossings)
        
        axes[1, 0].plot(multipliers, event_counts, 'b-', linewidth=2)
        axes[1, 0].axvline(self.detector.config.baseline_std_multiplier, color='red', linestyle='--', 
                          label=f'Current Multiplier: {self.detector.config.baseline_std_multiplier}')
        axes[1, 0].set_xlabel('Standard Deviation Multiplier')
        axes[1, 0].set_ylabel('Estimated Event Count')
        axes[1, 0].set_title('Threshold Sensitivity Analysis')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Statistical summary
        axes[1, 1].axis('off')
        
        # Create text summary
        stats_text = f"""
Baseline Statistics:
• Baseline Mean: {self.detector.baseline_mean:.6f}
• Baseline Std: {self.detector.baseline_std:.6f}
• Detection Threshold: {self.detector.threshold:.6f}

Configuration:
• Baseline Percentile: {self.detector.config.baseline_percentile}%
• Std Multiplier: {self.detector.config.baseline_std_multiplier}
• Window Size: {self.detector.config.window_size}
• Sampling Rate: {self.detector.config.sampling_rate} Hz

Data Information:
• Total Samples: {len(self.detector.data):,}
• Duration: {len(self.detector.data)/self.detector.config.sampling_rate:.2f} seconds
• Quiet Period Samples: {np.sum(quiet_periods):,}
• % Quiet Time: {100*np.sum(quiet_periods)/len(quiet_periods):.1f}%
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved baseline analysis plot to {save_path}")
        
        plt.show()
    
    def plot_individual_events(self, event_indices: Optional[List[int]] = None, 
                             max_events: int = 6, figsize: tuple = (15, 10),
                             save_path: Optional[str] = None) -> None:
        """
        Plot individual detected events in detail.
        
        Args:
            event_indices: Specific event indices to plot (plots all if None)
            max_events: Maximum number of events to plot
            figsize: Figure size tuple
            save_path: Optional path to save the figure
        """
        if not self.detector.detected_events:
            raise ValueError("No events detected")
        
        if event_indices is None:
            # Select events to plot (largest events first)
            events_by_energy = sorted(enumerate(self.detector.detected_events), 
                                    key=lambda x: x[1].peak_energy, reverse=True)
            event_indices = [i for i, _ in events_by_energy[:max_events]]
        
        n_events = len(event_indices)
        n_cols = min(3, n_events)
        n_rows = (n_events + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_events == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, event_idx in enumerate(event_indices):
            if i >= len(axes):
                break
                
            event = self.detector.detected_events[event_idx]
            
            # Create time axis for this event
            event_duration = event.end_idx - event.start_idx
            event_time = np.arange(event_duration) / self.detector.config.sampling_rate
            
            # Plot raw data if available
            if event.raw_data is not None:
                axes[i].plot(event_time, event.raw_data, 'b-', alpha=0.7, linewidth=1, label='Raw Data')
            
            # Plot energy profile
            if hasattr(event, 'energy_profile') and event.energy_profile is not None:
                # Interpolate energy to match raw data length
                energy_time = np.linspace(0, event_time[-1], len(event.energy_profile))
                axes[i].plot(energy_time, event.energy_profile * np.max(np.abs(event.raw_data)) / np.max(event.energy_profile), 
                           'r-', linewidth=2, label='Energy Profile (scaled)')
            
            # Mark peak
            peak_time = (event.peak_idx - event.start_idx) / self.detector.config.sampling_rate
            if event.raw_data is not None:
                peak_amplitude = event.raw_data[event.peak_idx - event.start_idx]
                axes[i].plot(peak_time, peak_amplitude, 'ro', markersize=6, label='Peak')
            
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Amplitude')
            axes[i].set_title(f'Event {event_idx + 1}\n'
                             f'Duration: {event.duration/self.detector.config.sampling_rate:.3f}s, '
                             f'Peak Energy: {event.peak_energy:.4f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(event_indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved individual events plot to {save_path}")
        
        plt.show()
    
    def plot_event_statistics(self, figsize: tuple = (12, 8), save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive event statistics and distributions.
        
        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the figure
        """
        if not self.detector.detected_events:
            raise ValueError("No events detected")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Extract event properties
        durations = [event.duration / self.detector.config.sampling_rate for event in self.detector.detected_events]
        peak_energies = [event.peak_energy for event in self.detector.detected_events]
        mean_energies = [event.mean_energy for event in self.detector.detected_events]
        start_times = [event.start_idx / self.detector.config.sampling_rate for event in self.detector.detected_events]
        
        # Plot 1: Duration distribution
        axes[0, 0].hist(durations, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Duration (s)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Event Duration Distribution\nMean: {np.mean(durations):.3f}s')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Peak energy distribution
        axes[0, 1].hist(peak_energies, bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_xlabel('Peak Energy')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Peak Energy Distribution\nMean: {np.mean(peak_energies):.4f}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Mean energy distribution
        axes[0, 2].hist(mean_energies, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 2].set_xlabel('Mean Energy')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title(f'Mean Energy Distribution\nMean: {np.mean(mean_energies):.4f}')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Event timeline
        axes[1, 0].plot(start_times, range(len(start_times)), 'bo-', markersize=4)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Event Number')
        axes[1, 0].set_title('Event Timeline')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Energy vs Duration scatter
        axes[1, 1].scatter(durations, peak_energies, alpha=0.7, c=mean_energies, cmap='viridis')
        axes[1, 1].set_xlabel('Duration (s)')
        axes[1, 1].set_ylabel('Peak Energy')
        axes[1, 1].set_title('Energy vs Duration')
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('Mean Energy')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Inter-event intervals
        if len(start_times) > 1:
            intervals = np.diff(start_times)
            axes[1, 2].hist(intervals, bins=15, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 2].set_xlabel('Inter-event Interval (s)')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title(f'Inter-event Intervals\nMean: {np.mean(intervals):.2f}s')
        else:
            axes[1, 2].text(0.5, 0.5, 'Need >1 event', transform=axes[1, 2].transAxes, ha='center')
            axes[1, 2].set_title('Inter-event Intervals')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved event statistics plot to {save_path}")
        
        plt.show()
    
    def plot_windowing_comparison(self, figsize: tuple = (12, 8), save_path: Optional[str] = None) -> None:
        """
        Compare sliding window vs adaptive windowing approaches.
        
        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the figure
        """
        if not self.detector.detected_events:
            raise ValueError("No events detected")
        
        # Create a copy of detector with adaptive windowing disabled
        config_fixed = EventDetectionConfig()
        config_fixed.__dict__.update(self.detector.config.__dict__)
        config_fixed.adaptive_window = False
        
        detector_fixed = EventDetector(config_fixed)
        detector_fixed.data = self.detector.data
        detector_fixed.filtered_data = self.detector.filtered_data
        detector_fixed.energy_profile = self.detector.energy_profile
        detector_fixed.baseline_mean = self.detector.baseline_mean
        detector_fixed.baseline_std = self.detector.baseline_std
        detector_fixed.threshold = self.detector.threshold
        detector_fixed.time_axis = self.detector.time_axis
        
        # Detect events with fixed windowing
        events_fixed = detector_fixed.detect_events()
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Adaptive windowing (current detector)
        axes[0].plot(self.detector.time_axis, self.detector.filtered_data or self.detector.data, 
                    'b-', alpha=0.3, linewidth=0.5, label='Data')
        
        for i, event in enumerate(self.detector.detected_events):
            start_time = event.start_idx / self.detector.config.sampling_rate
            end_time = event.end_idx / self.detector.config.sampling_rate
            axes[0].axvspan(start_time, end_time, alpha=0.3, color='red', 
                           label='Adaptive Events' if i == 0 else "")
        
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'Adaptive Windowing (Events: {len(self.detector.detected_events)})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Fixed windowing
        axes[1].plot(self.detector.time_axis, self.detector.filtered_data or self.detector.data, 
                    'b-', alpha=0.3, linewidth=0.5, label='Data')
        
        for i, event in enumerate(events_fixed):
            start_time = event.start_idx / self.detector.config.sampling_rate
            end_time = event.end_idx / self.detector.config.sampling_rate
            axes[1].axvspan(start_time, end_time, alpha=0.3, color='green', 
                           label='Fixed Events' if i == 0 else "")
        
        axes[1].set_ylabel('Amplitude')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title(f'Fixed Windowing (Events: {len(events_fixed)})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved windowing comparison plot to {save_path}")
        
        plt.show()


class InteractiveParameterTuning:
    """
    Interactive parameter tuning interface using matplotlib widgets.
    
    Provides real-time adjustment of detection parameters with immediate
    visual feedback on the detection results.
    """
    
    def __init__(self, detector: EventDetector):
        """
        Initialize interactive tuning interface.
        
        Args:
            detector: EventDetector instance with loaded data
        """
        self.detector = detector
        self.original_config = EventDetectionConfig()
        self.original_config.__dict__.update(detector.config.__dict__)
        
        # Store original results
        self.original_threshold = detector.threshold
        self.original_events = detector.detected_events.copy()
        
        self.fig = None
        self.ax_main = None
        self.sliders = {}
        
    def create_interactive_plot(self, figsize: tuple = (14, 10)) -> None:
        """
        Create interactive plot with parameter sliders.
        
        Args:
            figsize: Figure size tuple
        """
        if self.detector.data is None or self.detector.energy_profile is None:
            raise ValueError("Detector must have data and energy profile calculated")
        
        # Create figure and main plot
        self.fig = plt.figure(figsize=figsize)
        
        # Main plot area
        self.ax_main = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        
        # Initial plot
        self._update_plot()
        
        # Slider area
        slider_ax_height = 0.03
        slider_spacing = 0.05
        slider_left = 0.1
        slider_width = 0.8
        
        # Create sliders
        slider_configs = [
            ('baseline_percentile', 5.0, 30.0, self.detector.config.baseline_percentile, '%'),
            ('baseline_std_multiplier', 1.0, 6.0, self.detector.config.baseline_std_multiplier, ''),
            ('min_event_duration', 50, 1000, self.detector.config.min_event_duration, ' samples'),
            ('merge_threshold', 0, 500, self.detector.config.merge_threshold, ' samples'),
        ]
        
        for i, (param_name, min_val, max_val, init_val, suffix) in enumerate(slider_configs):
            ax_slider = plt.axes([slider_left, 0.15 - i * slider_spacing, slider_width, slider_ax_height])
            slider = Slider(ax_slider, f'{param_name.replace("_", " ").title()}{suffix}', 
                           min_val, max_val, valinit=init_val, valfmt='%.2f')
            slider.on_changed(lambda val, param=param_name: self._on_parameter_change(param, val))
            self.sliders[param_name] = slider
        
        # Reset button
        ax_reset = plt.axes([0.1, 0.02, 0.1, 0.04])
        button_reset = Button(ax_reset, 'Reset')
        button_reset.on_clicked(self._on_reset)
        
        # Export button
        ax_export = plt.axes([0.25, 0.02, 0.15, 0.04])
        button_export = Button(ax_export, 'Export Events')
        button_export.on_clicked(self._on_export)
        
        plt.subplots_adjust(bottom=0.2)
        plt.show()
    
    def _update_plot(self) -> None:
        """Update the main plot with current detection results"""
        self.ax_main.clear()
        
        # Plot energy profile
        self.ax_main.plot(self.detector.time_axis, self.detector.energy_profile, 
                         'b-', linewidth=1, label='Energy Profile')
        
        # Plot threshold
        if self.detector.threshold is not None:
            self.ax_main.axhline(y=self.detector.threshold, color='red', linestyle='--', 
                                linewidth=2, label=f'Threshold: {self.detector.threshold:.4f}')
        
        # Plot baseline
        if self.detector.baseline_mean is not None:
            self.ax_main.axhline(y=self.detector.baseline_mean, color='gray', linestyle=':', 
                                linewidth=2, label=f'Baseline: {self.detector.baseline_mean:.4f}')
        
        # Highlight detected events
        for i, event in enumerate(self.detector.detected_events):
            start_time = event.start_idx / self.detector.config.sampling_rate
            end_time = event.end_idx / self.detector.config.sampling_rate
            
            self.ax_main.axvspan(start_time, end_time, alpha=0.3, color='yellow', 
                               label='Events' if i == 0 else "")
        
        self.ax_main.set_xlabel('Time (s)')
        self.ax_main.set_ylabel('Energy')
        self.ax_main.set_title(f'Event Detection - {len(self.detector.detected_events)} events detected')
        self.ax_main.legend()
        self.ax_main.grid(True, alpha=0.3)
        
        # Update figure
        self.fig.canvas.draw()
    
    def _on_parameter_change(self, param_name: str, value: float) -> None:
        """Handle parameter change from slider"""
        # Update configuration
        setattr(self.detector.config, param_name, value)
        
        # Recalculate detection with new parameters
        try:
            # Recalculate baseline and threshold if baseline parameters changed
            if param_name in ['baseline_percentile', 'baseline_std_multiplier']:
                self.detector.calculate_baseline()
                self.detector.calculate_threshold()
            
            # Re-detect events
            self.detector.detect_events()
            
            # Update plot
            self._update_plot()
            
        except Exception as e:
            logger.warning(f"Error updating detection with {param_name}={value}: {str(e)}")
    
    def _on_reset(self, event) -> None:
        """Reset all parameters to original values"""
        # Restore original configuration
        self.detector.config.__dict__.update(self.original_config.__dict__)
        
        # Reset sliders
        for param_name, slider in self.sliders.items():
            slider.reset()
        
        # Restore original results
        self.detector.threshold = self.original_threshold
        self.detector.detected_events = self.original_events.copy()
        
        # Recalculate with original settings
        self.detector.calculate_baseline()
        self.detector.calculate_threshold()
        self.detector.detect_events()
        
        self._update_plot()
    
    def _on_export(self, event) -> None:
        """Export current detection results"""
        try:
            timestamp = plt.datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"detected_events_{timestamp}.csv"
            self.detector.export_events(csv_path, format='csv')
            
            logger.info(f"Exported {len(self.detector.detected_events)} events to {csv_path}")
            
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")


def create_step_by_step_visualization(detector: EventDetector, save_dir: Optional[str] = None) -> None:
    """
    Create step-by-step visualization showing the complete detection process.
    
    Args:
        detector: EventDetector instance with processed data
        save_dir: Directory to save individual step plots
    """
    if detector.data is None:
        raise ValueError("No data loaded in detector")
    
    visualizer = EventVisualization(detector)
    
    logger.info("Creating step-by-step visualization...")
    
    # Step 1: Raw data overview
    logger.info("Step 1: Raw data overview")
    visualizer.plot_detection_overview(save_path=f"{save_dir}/step1_overview.png" if save_dir else None)
    
    # Step 2: Baseline analysis
    logger.info("Step 2: Baseline and threshold analysis")
    visualizer.plot_baseline_analysis(save_path=f"{save_dir}/step2_baseline.png" if save_dir else None)
    
    # Step 3: Individual events
    if detector.detected_events:
        logger.info("Step 3: Individual event analysis")
        visualizer.plot_individual_events(save_path=f"{save_dir}/step3_events.png" if save_dir else None)
        
        # Step 4: Event statistics
        logger.info("Step 4: Event statistics")
        visualizer.plot_event_statistics(save_path=f"{save_dir}/step4_statistics.png" if save_dir else None)
        
        # Step 5: Windowing comparison
        logger.info("Step 5: Windowing comparison")
        visualizer.plot_windowing_comparison(save_path=f"{save_dir}/step5_windowing.png" if save_dir else None)
    else:
        logger.warning("No events detected - skipping individual event analysis")
    
    logger.info("Step-by-step visualization completed")