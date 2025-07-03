"""
Create synthetic vibration data for testing the event detection system.

This script generates realistic vibration data with embedded events
and saves it in H5 format for demonstration purposes.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from scipy import signal

logger = logging.getLogger(__name__)


def generate_background_noise(duration: float, sampling_rate: float = 1000.0, 
                            noise_level: float = 0.1) -> np.ndarray:
    """
    Generate background vibration noise.
    
    Args:
        duration: Duration in seconds
        sampling_rate: Sampling rate in Hz
        noise_level: Noise amplitude level
        
    Returns:
        Background noise array
    """
    n_samples = int(duration * sampling_rate)
    
    # Generate colored noise (more realistic for vibration)
    # Combine multiple frequency components
    time = np.arange(n_samples) / sampling_rate
    
    # Low frequency component (environmental vibration)
    low_freq = noise_level * 0.3 * np.sin(2 * np.pi * 2 * time + np.random.random() * 2 * np.pi)
    
    # Mid frequency component (mechanical noise)
    mid_freq = noise_level * 0.2 * np.sin(2 * np.pi * 15 * time + np.random.random() * 2 * np.pi)
    
    # High frequency white noise
    white_noise = noise_level * 0.5 * np.random.normal(0, 1, n_samples)
    
    # Apply low-pass filter to white noise
    b, a = signal.butter(4, 50, fs=sampling_rate, btype='low')
    filtered_noise = signal.filtfilt(b, a, white_noise)
    
    return low_freq + mid_freq + filtered_noise


def generate_vibration_event(duration: float, peak_amplitude: float, 
                           frequency: float = 25.0, sampling_rate: float = 1000.0,
                           event_type: str = 'impact') -> np.ndarray:
    """
    Generate a single vibration event.
    
    Args:
        duration: Event duration in seconds
        peak_amplitude: Peak amplitude of the event
        frequency: Dominant frequency of the event
        sampling_rate: Sampling rate in Hz
        event_type: Type of event ('impact', 'oscillation', 'chirp')
        
    Returns:
        Event signal array
    """
    n_samples = int(duration * sampling_rate)
    time = np.arange(n_samples) / sampling_rate
    
    if event_type == 'impact':
        # Damped oscillation (typical impact response)
        decay_rate = 5.0  # Exponential decay rate
        oscillation = np.sin(2 * np.pi * frequency * time)
        envelope = np.exp(-decay_rate * time)
        signal_data = peak_amplitude * oscillation * envelope
        
    elif event_type == 'oscillation':
        # Sustained oscillation with gradual decay
        decay_rate = 1.0
        oscillation = np.sin(2 * np.pi * frequency * time)
        envelope = np.exp(-decay_rate * time)
        signal_data = peak_amplitude * oscillation * envelope
        
    elif event_type == 'chirp':
        # Frequency sweep (chirp signal)
        f_start = frequency * 0.5
        f_end = frequency * 2.0
        chirp_signal = signal.chirp(time, f_start, duration, f_end)
        envelope = np.exp(-2.0 * time)
        signal_data = peak_amplitude * chirp_signal * envelope
        
    else:
        raise ValueError(f"Unknown event type: {event_type}")
    
    return signal_data


def create_synthetic_vibration_data(duration: float = 60.0, sampling_rate: float = 1000.0,
                                  n_events: int = 10, noise_level: float = 0.1) -> dict:
    """
    Create synthetic vibration data with embedded events.
    
    Args:
        duration: Total duration in seconds
        sampling_rate: Sampling rate in Hz
        n_events: Number of events to embed
        noise_level: Background noise level
        
    Returns:
        Dictionary containing data and metadata
    """
    n_samples = int(duration * sampling_rate)
    
    # Generate background noise
    logger.info(f"Generating {duration}s of background noise...")
    background = generate_background_noise(duration, sampling_rate, noise_level)
    
    # Initialize result data
    data = background.copy()
    time_axis = np.arange(n_samples) / sampling_rate
    
    # Generate random events
    logger.info(f"Embedding {n_events} vibration events...")
    events_metadata = []
    
    # Ensure events don't overlap by dividing time into slots
    time_slots = np.linspace(5, duration - 5, n_events + 1)  # Leave 5s at start/end
    
    for i in range(n_events):
        # Random event properties
        event_start_time = time_slots[i] + np.random.uniform(0, time_slots[1] - time_slots[0] - 2)
        event_duration = np.random.uniform(0.1, 1.5)  # 0.1 to 1.5 seconds
        peak_amplitude = np.random.uniform(0.5, 3.0)  # Relative to noise level
        frequency = np.random.uniform(10, 50)  # 10-50 Hz
        event_type = np.random.choice(['impact', 'oscillation', 'chirp'])
        
        # Generate event signal
        event_signal = generate_vibration_event(event_duration, peak_amplitude, 
                                               frequency, sampling_rate, event_type)
        
        # Embed in main signal
        start_idx = int(event_start_time * sampling_rate)
        end_idx = start_idx + len(event_signal)
        
        if end_idx <= n_samples:
            data[start_idx:end_idx] += event_signal
            
            events_metadata.append({
                'event_id': i,
                'start_time': event_start_time,
                'end_time': event_start_time + event_duration,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration': event_duration,
                'peak_amplitude': peak_amplitude,
                'frequency': frequency,
                'event_type': event_type
            })
    
    logger.info(f"Successfully embedded {len(events_metadata)} events")
    
    return {
        'vibration_data': data,
        'time_axis': time_axis,
        'sampling_rate': sampling_rate,
        'events_metadata': events_metadata,
        'background_noise_level': noise_level,
        'duration': duration
    }


def save_to_h5(data_dict: dict, filename: str) -> None:
    """
    Save synthetic vibration data to H5 file.
    
    Args:
        data_dict: Data dictionary from create_synthetic_vibration_data
        filename: Output filename
    """
    with h5py.File(filename, 'w') as f:
        # Main datasets
        f.create_dataset('vibration_data', data=data_dict['vibration_data'])
        f.create_dataset('time_axis', data=data_dict['time_axis'])
        
        # Metadata
        f.attrs['sampling_rate'] = data_dict['sampling_rate']
        f.attrs['duration'] = data_dict['duration']
        f.attrs['background_noise_level'] = data_dict['background_noise_level']
        f.attrs['n_events'] = len(data_dict['events_metadata'])
        
        # Events metadata
        if data_dict['events_metadata']:
            events_group = f.create_group('true_events')
            
            # Convert list of dicts to separate arrays
            event_ids = [e['event_id'] for e in data_dict['events_metadata']]
            start_times = [e['start_time'] for e in data_dict['events_metadata']]
            end_times = [e['end_time'] for e in data_dict['events_metadata']]
            start_indices = [e['start_idx'] for e in data_dict['events_metadata']]
            end_indices = [e['end_idx'] for e in data_dict['events_metadata']]
            durations = [e['duration'] for e in data_dict['events_metadata']]
            peak_amplitudes = [e['peak_amplitude'] for e in data_dict['events_metadata']]
            frequencies = [e['frequency'] for e in data_dict['events_metadata']]
            event_types = [e['event_type'].encode() for e in data_dict['events_metadata']]
            
            events_group.create_dataset('event_ids', data=event_ids)
            events_group.create_dataset('start_times', data=start_times)
            events_group.create_dataset('end_times', data=end_times)
            events_group.create_dataset('start_indices', data=start_indices)
            events_group.create_dataset('end_indices', data=end_indices)
            events_group.create_dataset('durations', data=durations)
            events_group.create_dataset('peak_amplitudes', data=peak_amplitudes)
            events_group.create_dataset('frequencies', data=frequencies)
            events_group.create_dataset('event_types', data=event_types)
    
    logger.info(f"Saved synthetic vibration data to {filename}")


def plot_synthetic_data(data_dict: dict, save_path: str = None) -> None:
    """
    Plot the synthetic vibration data with true events marked.
    
    Args:
        data_dict: Data dictionary from create_synthetic_vibration_data
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    time_axis = data_dict['time_axis']
    vibration_data = data_dict['vibration_data']
    
    # Plot 1: Full signal
    axes[0].plot(time_axis, vibration_data, 'b-', linewidth=0.5, alpha=0.8)
    
    # Mark true events
    for event in data_dict['events_metadata']:
        start_time = event['start_time']
        end_time = event['end_time']
        axes[0].axvspan(start_time, end_time, alpha=0.3, color='red')
        
        # Add event label
        mid_time = (start_time + end_time) / 2
        axes[0].text(mid_time, np.max(vibration_data) * 0.8, 
                    f"E{event['event_id']}", ha='center', fontsize=8)
    
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Synthetic Vibration Data (Duration: {data_dict["duration"]}s, Events: {len(data_dict["events_metadata"])})')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Zoomed view of first 10 seconds
    zoom_end = min(10, data_dict['duration'])
    zoom_mask = time_axis <= zoom_end
    
    axes[1].plot(time_axis[zoom_mask], vibration_data[zoom_mask], 'b-', linewidth=1)
    
    # Mark events in zoom view
    for event in data_dict['events_metadata']:
        if event['start_time'] <= zoom_end:
            start_time = event['start_time']
            end_time = min(event['end_time'], zoom_end)
            axes[1].axvspan(start_time, end_time, alpha=0.3, color='red')
    
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'Zoomed View (First {zoom_end}s)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved synthetic data plot to {save_path}")
    
    plt.show()


def main():
    """Generate multiple synthetic datasets for testing"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    output_dir = Path("synthetic_data")
    output_dir.mkdir(exist_ok=True)
    
    # Dataset configurations
    datasets = [
        {
            'name': 'simple_test',
            'duration': 30.0,
            'n_events': 5,
            'noise_level': 0.1,
            'description': 'Simple test case with few events'
        },
        {
            'name': 'moderate_activity',
            'duration': 60.0,
            'n_events': 12,
            'noise_level': 0.15,
            'description': 'Moderate activity with more events'
        },
        {
            'name': 'high_activity',
            'duration': 120.0,
            'n_events': 25,
            'noise_level': 0.2,
            'description': 'High activity scenario'
        },
        {
            'name': 'low_noise',
            'duration': 45.0,
            'n_events': 8,
            'noise_level': 0.05,
            'description': 'Low noise environment'
        }
    ]
    
    for dataset_config in datasets:
        logger.info(f"\nGenerating dataset: {dataset_config['name']}")
        logger.info(f"Description: {dataset_config['description']}")
        
        # Generate data
        data_dict = create_synthetic_vibration_data(
            duration=dataset_config['duration'],
            n_events=dataset_config['n_events'],
            noise_level=dataset_config['noise_level']
        )
        
        # Save to H5 file
        h5_filename = output_dir / f"{dataset_config['name']}.h5"
        save_to_h5(data_dict, str(h5_filename))
        
        # Create visualization
        plot_filename = output_dir / f"{dataset_config['name']}_preview.png"
        plot_synthetic_data(data_dict, str(plot_filename))
        
        # Print summary
        logger.info(f"Dataset summary:")
        logger.info(f"  Duration: {dataset_config['duration']}s")
        logger.info(f"  Events: {len(data_dict['events_metadata'])}")
        logger.info(f"  Noise level: {dataset_config['noise_level']}")
        logger.info(f"  Saved to: {h5_filename}")
    
    logger.info(f"\nAll datasets generated in: {output_dir}")
    logger.info("Ready for event detection testing!")


if __name__ == "__main__":
    main()