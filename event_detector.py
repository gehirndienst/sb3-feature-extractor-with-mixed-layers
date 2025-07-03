"""
Event Detection System for Vibration Data

This module provides a comprehensive event detection system for vibration data
stored in H5 format. It includes baseline noise calculation, threshold-based
event detection, and various windowing approaches.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import percentileofscore
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EventDetectionConfig:
    """Configuration parameters for event detection"""
    # Baseline calculation parameters
    baseline_percentile: float = 15.0  # Percentile for quiet periods (10-20%)
    baseline_std_multiplier: float = 3.0  # N in: threshold = baseline_mean + N * baseline_std
    
    # Window parameters
    window_size: int = 1000  # Sliding window size for energy calculation
    window_overlap: float = 0.5  # Overlap between windows (0-1)
    
    # Event detection parameters
    min_event_duration: int = 100  # Minimum samples for valid event
    max_event_duration: int = 10000  # Maximum samples for valid event
    merge_threshold: int = 200  # Merge events closer than this many samples
    
    # Adaptive windowing parameters
    adaptive_window: bool = True  # Use adaptive windowing for event capture
    energy_fallback_ratio: float = 0.3  # Fallback to % of peak energy for event end
    
    # Filtering parameters
    filter_enabled: bool = True  # Enable bandpass filtering
    filter_lowcut: float = 1.0  # Low cutoff frequency (Hz)
    filter_highcut: float = 100.0  # High cutoff frequency (Hz)
    sampling_rate: float = 1000.0  # Sampling rate (Hz)


@dataclass
class DetectedEvent:
    """Represents a detected vibration event"""
    start_idx: int
    end_idx: int
    peak_idx: int
    duration: int
    peak_energy: float
    mean_energy: float
    energy_profile: np.ndarray
    raw_data: np.ndarray


class EventDetector:
    """
    Main class for detecting events in vibration data.
    
    This class provides comprehensive event detection capabilities including:
    - Baseline noise calculation from quiet periods
    - Threshold-based event detection
    - Sliding window and adaptive windowing
    - Event statistics and analysis
    - Visualization tools
    """
    
    def __init__(self, config: Optional[EventDetectionConfig] = None):
        """
        Initialize the EventDetector.
        
        Args:
            config: Configuration parameters for event detection
        """
        self.config = config or EventDetectionConfig()
        self.data = None
        self.filtered_data = None
        self.energy_profile = None
        self.baseline_mean = None
        self.baseline_std = None
        self.threshold = None
        self.detected_events = []
        self.time_axis = None
        
        logger.info("EventDetector initialized with configuration:")
        logger.info(f"  Baseline percentile: {self.config.baseline_percentile}%")
        logger.info(f"  Threshold multiplier: {self.config.baseline_std_multiplier}")
        logger.info(f"  Window size: {self.config.window_size}")
        logger.info(f"  Adaptive windowing: {self.config.adaptive_window}")
    
    def load_h5_data(self, file_path: str, dataset_name: str = 'vibration_data') -> np.ndarray:
        """
        Load vibration data from H5 file.
        
        Args:
            file_path: Path to H5 file
            dataset_name: Name of dataset in H5 file
            
        Returns:
            Loaded vibration data array
        """
        try:
            with h5py.File(file_path, 'r') as f:
                if dataset_name in f:
                    self.data = f[dataset_name][:]
                    logger.info(f"Loaded data with shape: {self.data.shape}")
                else:
                    # Try to find any dataset if specified name doesn't exist
                    available_datasets = list(f.keys())
                    if available_datasets:
                        dataset_name = available_datasets[0]
                        self.data = f[dataset_name][:]
                        logger.warning(f"Dataset '{dataset_name}' not found. Using '{dataset_name}' instead.")
                    else:
                        raise ValueError("No datasets found in H5 file")
                        
            # Create time axis assuming sampling rate from config
            self.time_axis = np.arange(len(self.data)) / self.config.sampling_rate
            
            # Apply filtering if enabled
            if self.config.filter_enabled:
                self.filtered_data = self._apply_bandpass_filter(self.data)
                logger.info("Applied bandpass filter")
            else:
                self.filtered_data = self.data.copy()
                
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading H5 data: {str(e)}")
            raise
    
    def load_data_from_array(self, data: np.ndarray, sampling_rate: float = None) -> np.ndarray:
        """
        Load vibration data from numpy array.
        
        Args:
            data: Vibration data array
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Loaded vibration data array
        """
        self.data = data.copy()
        
        if sampling_rate is not None:
            self.config.sampling_rate = sampling_rate
            
        # Create time axis
        self.time_axis = np.arange(len(self.data)) / self.config.sampling_rate
        
        # Apply filtering if enabled
        if self.config.filter_enabled:
            self.filtered_data = self._apply_bandpass_filter(self.data)
            logger.info("Applied bandpass filter")
        else:
            self.filtered_data = self.data.copy()
            
        logger.info(f"Loaded data with shape: {self.data.shape}")
        return self.data
    
    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to the data"""
        try:
            nyquist = self.config.sampling_rate / 2
            low = self.config.filter_lowcut / nyquist
            high = self.config.filter_highcut / nyquist
            
            # Design Butterworth bandpass filter
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, data)
            
            return filtered
        except Exception as e:
            logger.warning(f"Filter application failed: {str(e)}. Using unfiltered data.")
            return data
    
    def calculate_energy_profile(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate energy profile using sliding window.
        
        Args:
            data: Input data (uses filtered_data if None)
            
        Returns:
            Energy profile array
        """
        if data is None:
            data = self.filtered_data
            
        if data is None:
            raise ValueError("No data available. Load data first.")
        
        window_size = self.config.window_size
        overlap = int(window_size * self.config.window_overlap)
        step = window_size - overlap
        
        # Calculate energy for each window
        energy_values = []
        for i in range(0, len(data) - window_size + 1, step):
            window_data = data[i:i + window_size]
            # Use RMS energy
            energy = np.sqrt(np.mean(window_data**2))
            energy_values.append(energy)
        
        # Interpolate to match original data length
        energy_indices = np.arange(0, len(data) - window_size + 1, step) + window_size // 2
        self.energy_profile = np.interp(np.arange(len(data)), energy_indices, energy_values)
        
        logger.info(f"Calculated energy profile with {len(energy_values)} windows")
        return self.energy_profile
    
    def calculate_baseline(self, energy_profile: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Calculate baseline noise statistics from quiet periods.
        
        The baseline is calculated using the bottom percentile of energy values,
        representing quiet periods in the vibration data.
        
        Args:
            energy_profile: Energy profile array (calculates if None)
            
        Returns:
            Tuple of (baseline_mean, baseline_std)
        """
        if energy_profile is None:
            energy_profile = self.energy_profile
            
        if energy_profile is None:
            energy_profile = self.calculate_energy_profile()
        
        # Find quiet periods (bottom percentile)
        threshold_value = np.percentile(energy_profile, self.config.baseline_percentile)
        quiet_periods = energy_profile[energy_profile <= threshold_value]
        
        if len(quiet_periods) == 0:
            logger.warning("No quiet periods found. Using full data for baseline.")
            quiet_periods = energy_profile
        
        self.baseline_mean = np.mean(quiet_periods)
        self.baseline_std = np.std(quiet_periods)
        
        logger.info(f"Baseline calculated from {len(quiet_periods)} quiet period samples")
        logger.info(f"  Baseline mean: {self.baseline_mean:.6f}")
        logger.info(f"  Baseline std: {self.baseline_std:.6f}")
        
        return self.baseline_mean, self.baseline_std
    
    def calculate_threshold(self, baseline_mean: Optional[float] = None, 
                          baseline_std: Optional[float] = None) -> float:
        """
        Calculate detection threshold.
        
        Threshold = baseline_mean + N * baseline_std
        where N is the standard deviation multiplier from config.
        
        Args:
            baseline_mean: Baseline mean (calculates if None)
            baseline_std: Baseline standard deviation (calculates if None)
            
        Returns:
            Detection threshold value
        """
        if baseline_mean is None or baseline_std is None:
            baseline_mean, baseline_std = self.calculate_baseline()
        
        self.threshold = baseline_mean + self.config.baseline_std_multiplier * baseline_std
        
        logger.info(f"Detection threshold: {self.threshold:.6f}")
        return self.threshold
    
    def detect_events(self, energy_profile: Optional[np.ndarray] = None,
                     threshold: Optional[float] = None) -> List[DetectedEvent]:
        """
        Detect events using threshold-based approach.
        
        Args:
            energy_profile: Energy profile array (calculates if None)
            threshold: Detection threshold (calculates if None)
            
        Returns:
            List of detected events
        """
        if energy_profile is None:
            energy_profile = self.energy_profile
            if energy_profile is None:
                energy_profile = self.calculate_energy_profile()
        
        if threshold is None:
            threshold = self.threshold
            if threshold is None:
                threshold = self.calculate_threshold()
        
        # Find points above threshold
        above_threshold = energy_profile > threshold
        
        # Find event boundaries
        event_starts = []
        event_ends = []
        
        in_event = False
        for i, is_above in enumerate(above_threshold):
            if is_above and not in_event:
                event_starts.append(i)
                in_event = True
            elif not is_above and in_event:
                event_ends.append(i)
                in_event = False
        
        # Handle case where last event doesn't end
        if in_event:
            event_ends.append(len(energy_profile) - 1)
        
        # Process detected events
        self.detected_events = []
        for start, end in zip(event_starts, event_ends):
            if self.config.adaptive_window:
                start, end = self._adaptive_event_window(energy_profile, start, end)
            
            duration = end - start
            
            # Filter events by duration
            if (self.config.min_event_duration <= duration <= self.config.max_event_duration):
                event_energy = energy_profile[start:end]
                peak_idx = start + np.argmax(event_energy)
                
                event = DetectedEvent(
                    start_idx=start,
                    end_idx=end,
                    peak_idx=peak_idx,
                    duration=duration,
                    peak_energy=energy_profile[peak_idx],
                    mean_energy=np.mean(event_energy),
                    energy_profile=event_energy,
                    raw_data=self.filtered_data[start:end] if self.filtered_data is not None else None
                )
                
                self.detected_events.append(event)
        
        # Merge close events
        if self.config.merge_threshold > 0:
            self.detected_events = self._merge_close_events(self.detected_events)
        
        logger.info(f"Detected {len(self.detected_events)} events")
        return self.detected_events
    
    def _adaptive_event_window(self, energy_profile: np.ndarray, 
                              start: int, end: int) -> Tuple[int, int]:
        """
        Adjust event window boundaries adaptively.
        
        Extends the window to capture the full event by looking for
        energy fallback points around the detected region.
        
        Args:
            energy_profile: Energy profile array
            start: Initial start index
            end: Initial end index
            
        Returns:
            Adjusted (start, end) indices
        """
        peak_idx = start + np.argmax(energy_profile[start:end])
        peak_energy = energy_profile[peak_idx]
        fallback_threshold = peak_energy * self.config.energy_fallback_ratio
        
        # Extend backward from start
        new_start = start
        for i in range(start - 1, max(0, start - self.config.max_event_duration), -1):
            if energy_profile[i] < fallback_threshold:
                new_start = i
                break
        
        # Extend forward from end
        new_end = end
        for i in range(end + 1, min(len(energy_profile), end + self.config.max_event_duration)):
            if energy_profile[i] < fallback_threshold:
                new_end = i
                break
        
        return new_start, new_end
    
    def _merge_close_events(self, events: List[DetectedEvent]) -> List[DetectedEvent]:
        """
        Merge events that are closer than the merge threshold.
        
        Args:
            events: List of detected events
            
        Returns:
            List of merged events
        """
        if len(events) <= 1:
            return events
        
        merged_events = []
        current_event = events[0]
        
        for next_event in events[1:]:
            gap = next_event.start_idx - current_event.end_idx
            
            if gap <= self.config.merge_threshold:
                # Merge events
                merged_start = current_event.start_idx
                merged_end = next_event.end_idx
                merged_duration = merged_end - merged_start
                
                # Recalculate properties for merged event
                merged_energy = self.energy_profile[merged_start:merged_end]
                merged_peak_idx = merged_start + np.argmax(merged_energy)
                
                current_event = DetectedEvent(
                    start_idx=merged_start,
                    end_idx=merged_end,
                    peak_idx=merged_peak_idx,
                    duration=merged_duration,
                    peak_energy=self.energy_profile[merged_peak_idx],
                    mean_energy=np.mean(merged_energy),
                    energy_profile=merged_energy,
                    raw_data=self.filtered_data[merged_start:merged_end] if self.filtered_data is not None else None
                )
            else:
                merged_events.append(current_event)
                current_event = next_event
        
        merged_events.append(current_event)
        
        if len(merged_events) < len(events):
            logger.info(f"Merged {len(events) - len(merged_events)} close events")
        
        return merged_events
    
    def get_event_statistics(self) -> Dict:
        """
        Calculate comprehensive statistics for detected events.
        
        Returns:
            Dictionary containing event statistics
        """
        if not self.detected_events:
            return {"error": "No events detected"}
        
        durations = [event.duration for event in self.detected_events]
        peak_energies = [event.peak_energy for event in self.detected_events]
        mean_energies = [event.mean_energy for event in self.detected_events]
        
        stats = {
            "total_events": len(self.detected_events),
            "event_rate": len(self.detected_events) / (len(self.data) / self.config.sampling_rate),  # events per second
            "duration_stats": {
                "mean": np.mean(durations),
                "std": np.std(durations),
                "min": np.min(durations),
                "max": np.max(durations),
                "median": np.median(durations)
            },
            "peak_energy_stats": {
                "mean": np.mean(peak_energies),
                "std": np.std(peak_energies),
                "min": np.min(peak_energies),
                "max": np.max(peak_energies),
                "median": np.median(peak_energies)
            },
            "mean_energy_stats": {
                "mean": np.mean(mean_energies),
                "std": np.std(mean_energies),
                "min": np.min(mean_energies),
                "max": np.max(mean_energies),
                "median": np.median(mean_energies)
            },
            "baseline_info": {
                "baseline_mean": self.baseline_mean,
                "baseline_std": self.baseline_std,
                "threshold": self.threshold
            }
        }
        
        return stats
    
    def export_events(self, file_path: str, format: str = 'csv') -> None:
        """
        Export detected events to file.
        
        Args:
            file_path: Output file path
            format: Export format ('csv' or 'h5')
        """
        if not self.detected_events:
            logger.warning("No events to export")
            return
        
        if format.lower() == 'csv':
            self._export_events_csv(file_path)
        elif format.lower() == 'h5':
            self._export_events_h5(file_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_events_csv(self, file_path: str) -> None:
        """Export events to CSV format"""
        import csv
        
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['event_id', 'start_time', 'end_time', 'duration', 
                         'peak_energy', 'mean_energy', 'start_idx', 'end_idx', 'peak_idx']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i, event in enumerate(self.detected_events):
                writer.writerow({
                    'event_id': i,
                    'start_time': event.start_idx / self.config.sampling_rate,
                    'end_time': event.end_idx / self.config.sampling_rate,
                    'duration': event.duration / self.config.sampling_rate,
                    'peak_energy': event.peak_energy,
                    'mean_energy': event.mean_energy,
                    'start_idx': event.start_idx,
                    'end_idx': event.end_idx,
                    'peak_idx': event.peak_idx
                })
        
        logger.info(f"Exported {len(self.detected_events)} events to {file_path}")
    
    def _export_events_h5(self, file_path: str) -> None:
        """Export events to H5 format"""
        with h5py.File(file_path, 'w') as f:
            # Create groups for different data types
            events_group = f.create_group('events')
            
            # Store event metadata
            n_events = len(self.detected_events)
            events_group.create_dataset('start_indices', data=[e.start_idx for e in self.detected_events])
            events_group.create_dataset('end_indices', data=[e.end_idx for e in self.detected_events])
            events_group.create_dataset('peak_indices', data=[e.peak_idx for e in self.detected_events])
            events_group.create_dataset('durations', data=[e.duration for e in self.detected_events])
            events_group.create_dataset('peak_energies', data=[e.peak_energy for e in self.detected_events])
            events_group.create_dataset('mean_energies', data=[e.mean_energy for e in self.detected_events])
            
            # Store raw data for each event
            raw_data_group = f.create_group('raw_data')
            for i, event in enumerate(self.detected_events):
                if event.raw_data is not None:
                    raw_data_group.create_dataset(f'event_{i}', data=event.raw_data)
            
            # Store configuration and statistics
            config_group = f.create_group('config')
            config_group.attrs['sampling_rate'] = self.config.sampling_rate
            config_group.attrs['baseline_percentile'] = self.config.baseline_percentile
            config_group.attrs['baseline_std_multiplier'] = self.config.baseline_std_multiplier
            config_group.attrs['threshold'] = self.threshold
            
        logger.info(f"Exported {len(self.detected_events)} events to {file_path}")