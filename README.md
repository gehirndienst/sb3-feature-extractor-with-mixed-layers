# SB3 Feature Extractor with Mixed Layers + Event Detection for Vibration Data

This repository contains two main components:

## 1. Multi-Layer Feature Extractor for Reinforcement Learning

A feature extractor that combines the subspaces of the observation space of different types and uses different layers of neural networks to extract features from each subspace. The layers include a simple flattening layer, an embedding layer, a CNN, and two different RNNs: GRU and LSTM with optional attention as in Transformer models. The last two are flexibly configurable, see the example. The features are then concatenated and passed through the main neural network defined for the deep reinforcement learning algorithm.

I also prepare a test environment to test the feature extractor with a simple out-of-the-box algorithm (PPO). The observation space of the test environment combines 4 data types: categorical, vector, image, and a variable-length sequence. The step performs a dummy update. The feature extractor is designed to handle these data types, extract features from them, and concatenate them.

## 2. Event Detection System for Vibration Data

A comprehensive event detection system for vibration data stored in H5 format. This system provides:

### Features
- **Automatic baseline noise calculation** from quiet periods in the data
- **Threshold-based event detection** with configurable parameters
- **Sliding window and adaptive windowing** approaches
- **Comprehensive visualization tools** for step-by-step analysis
- **Interactive parameter tuning** with real-time feedback
- **Event statistics and analysis** tools
- **H5 file support** for vibration data
- **Export functionality** for detected events (CSV and H5 formats)

### Event Detection Components

#### EventDetector Class
The main class for detecting events in vibration data with features including:
- Baseline noise calculation from bottom 10-20% percentile of energy values
- Threshold calculation as: `baseline_mean + N * baseline_std` (typically N=2-4)
- Both sliding window and adaptive windowing for event capture
- Event merging and filtering capabilities
- Comprehensive event statistics

#### Visualization Tools
- **Step-by-step detection visualization** showing the complete process
- **Baseline and threshold analysis** with detailed explanations
- **Individual event inspection** with energy profiles
- **Event statistics and distributions** analysis
- **Windowing approach comparisons**

#### Interactive Analysis
- **Real-time parameter tuning** with immediate visual feedback
- **Adjustable detection parameters** including baseline percentile, threshold multiplier, window sizes
- **Export capabilities** for detected events
- **Reset functionality** to restore original settings

### How Event Detection Works

1. **Baseline Calculation**: Analyze quiet periods (bottom 10-20% percentile of energy values)
2. **Threshold Setting**: Calculate threshold as `baseline_mean + N * baseline_std`
3. **Energy Profile**: Calculate RMS energy using sliding windows
4. **Event Detection**: Identify regions where energy exceeds threshold
5. **Adaptive Windowing**: Extend event boundaries to capture complete events
6. **Event Merging**: Combine closely spaced events based on merge threshold

## Installation

```bash
pip install torch stable-baselines3 gymnasium numpy h5py matplotlib scipy
```

## Usage

### Feature Extractor (Original)
```bash
python try.py
```

### Event Detection System (New)

#### Quick Start
```python
from event_detector import EventDetector, EventDetectionConfig
from visualization import EventVisualization

# Load your H5 vibration data
detector = EventDetector()
detector.load_h5_data('your_vibration_data.h5', dataset_name='vibration_data')

# Detect events
detector.calculate_energy_profile()
detector.calculate_baseline()
detector.calculate_threshold()
events = detector.detect_events()

# Visualize results
visualizer = EventVisualization(detector)
visualizer.plot_detection_overview()
```

#### Comprehensive Example
```bash
python vibration_analysis_example.py
```

This runs a complete demonstration including:
- Basic event detection with synthetic data
- Noise threshold analysis and explanation
- Windowing approaches comparison
- Interactive parameter tuning
- Comprehensive analysis with step-by-step visualization
- H5 file loading demonstration

#### Generate Test Data
```bash
python generate_synthetic_data.py
```

Creates synthetic vibration datasets with embedded events for testing.

### Event Detection Configuration

```python
from event_detector import EventDetectionConfig

config = EventDetectionConfig(
    baseline_percentile=15.0,        # Percentile for quiet periods (10-20%)
    baseline_std_multiplier=3.0,     # Threshold = baseline_mean + N * baseline_std
    window_size=1000,                # Sliding window size for energy calculation
    min_event_duration=100,          # Minimum samples for valid event
    adaptive_window=True,            # Use adaptive windowing
    filter_enabled=True,             # Enable bandpass filtering
    filter_lowcut=1.0,              # Low cutoff frequency (Hz)
    filter_highcut=100.0,           # High cutoff frequency (Hz)
    sampling_rate=1000.0            # Sampling rate (Hz)
)

detector = EventDetector(config)
```

### File Structure

```
├── event_detector.py              # Main event detection class
├── visualization.py               # Visualization tools and interactive tuning
├── generate_synthetic_data.py     # Synthetic data generation
├── vibration_analysis_example.py  # Comprehensive usage example
├── feature_extractor.py           # Original RL feature extractor
├── rnn_attention.py              # RNN attention mechanism
├── env.py                        # Test environment for RL
└── try.py                        # Original RL example
```

## Author
Nikita Smirnov.
Please contact me in case of any questions or bug reports: [mailto](mailto:detectivecolombo@gmail.com)