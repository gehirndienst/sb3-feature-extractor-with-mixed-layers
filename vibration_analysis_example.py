"""
Event Detection for Vibration Data - Example Usage

This script demonstrates the complete event detection system with
step-by-step visualization and interactive parameter tuning.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from event_detector import EventDetector, EventDetectionConfig
from visualization import EventVisualization, InteractiveParameterTuning, create_step_by_step_visualization
from generate_synthetic_data import create_synthetic_vibration_data, save_to_h5, plot_synthetic_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_basic_event_detection():
    """
    Demonstrate basic event detection functionality.
    """
    logger.info("=== BASIC EVENT DETECTION DEMONSTRATION ===")
    
    # Generate synthetic data for demonstration
    logger.info("Generating synthetic vibration data...")
    data_dict = create_synthetic_vibration_data(
        duration=30.0,  # 30 seconds
        n_events=6,     # 6 embedded events
        noise_level=0.1 # Low noise level
    )
    
    # Create and configure detector
    config = EventDetectionConfig(
        baseline_percentile=15.0,
        baseline_std_multiplier=3.0,
        window_size=500,
        min_event_duration=50,
        adaptive_window=True,
        filter_enabled=True,
        sampling_rate=1000.0
    )
    
    detector = EventDetector(config)
    
    # Load data from array
    detector.load_data_from_array(data_dict['vibration_data'], sampling_rate=1000.0)
    
    # Step 1: Calculate energy profile
    logger.info("Step 1: Calculating energy profile...")
    energy_profile = detector.calculate_energy_profile()
    logger.info(f"Energy profile calculated with {len(energy_profile)} samples")
    
    # Step 2: Calculate baseline
    logger.info("Step 2: Calculating baseline noise statistics...")
    baseline_mean, baseline_std = detector.calculate_baseline()
    logger.info(f"Baseline: mean={baseline_mean:.6f}, std={baseline_std:.6f}")
    
    # Step 3: Calculate threshold
    logger.info("Step 3: Calculating detection threshold...")
    threshold = detector.calculate_threshold()
    logger.info(f"Detection threshold: {threshold:.6f}")
    
    # Step 4: Detect events
    logger.info("Step 4: Detecting events...")
    detected_events = detector.detect_events()
    logger.info(f"Detected {len(detected_events)} events")
    
    # Step 5: Get statistics
    logger.info("Step 5: Calculating event statistics...")
    stats = detector.get_event_statistics()
    
    # Print summary
    print("\n" + "="*60)
    print("EVENT DETECTION SUMMARY")
    print("="*60)
    print(f"Data duration: {data_dict['duration']:.1f} seconds")
    print(f"Sampling rate: {detector.config.sampling_rate:.0f} Hz")
    print(f"Total samples: {len(detector.data):,}")
    print()
    print("BASELINE ANALYSIS:")
    print(f"  Baseline percentile used: {detector.config.baseline_percentile}%")
    print(f"  Baseline mean: {baseline_mean:.6f}")
    print(f"  Baseline std: {baseline_std:.6f}")
    print(f"  Detection threshold: {threshold:.6f}")
    print(f"  Std multiplier: {detector.config.baseline_std_multiplier}")
    print()
    print("EVENT DETECTION RESULTS:")
    print(f"  Events detected: {len(detected_events)}")
    print(f"  True events (embedded): {len(data_dict['events_metadata'])}")
    print(f"  Event rate: {stats['event_rate']:.2f} events/second")
    print()
    if detected_events:
        print("EVENT STATISTICS:")
        print(f"  Duration - Mean: {stats['duration_stats']['mean']:.3f}s, "
              f"Range: {stats['duration_stats']['min']:.3f}-{stats['duration_stats']['max']:.3f}s")
        print(f"  Peak Energy - Mean: {stats['peak_energy_stats']['mean']:.4f}, "
              f"Range: {stats['peak_energy_stats']['min']:.4f}-{stats['peak_energy_stats']['max']:.4f}")
    
    # Create basic visualization
    logger.info("Creating basic visualization...")
    visualizer = EventVisualization(detector)
    visualizer.plot_detection_overview()
    
    return detector, data_dict


def demonstrate_threshold_analysis():
    """
    Demonstrate noise threshold understanding and analysis.
    """
    logger.info("\n=== NOISE THRESHOLD ANALYSIS ===")
    
    # Generate data with different noise characteristics
    data_dict = create_synthetic_vibration_data(
        duration=45.0,
        n_events=8,
        noise_level=0.15
    )
    
    detector = EventDetector()
    detector.load_data_from_array(data_dict['vibration_data'])
    
    # Calculate energy and baseline
    detector.calculate_energy_profile()
    detector.calculate_baseline()
    detector.calculate_threshold()
    detector.detect_events()
    
    print("\n" + "="*60)
    print("NOISE THRESHOLD EXPLANATION")
    print("="*60)
    print("The noise threshold in vibration analysis works as follows:")
    print()
    print("1. BASELINE CALCULATION:")
    print(f"   - Analyze quiet periods (bottom {detector.config.baseline_percentile}% of energy values)")
    print(f"   - Calculate mean baseline energy: {detector.baseline_mean:.6f}")
    print(f"   - Calculate baseline standard deviation: {detector.baseline_std:.6f}")
    print()
    print("2. THRESHOLD SETTING:")
    print(f"   - Threshold = baseline_mean + N × baseline_std")
    print(f"   - N = {detector.config.baseline_std_multiplier} (std multiplier)")
    print(f"   - Threshold = {detector.baseline_mean:.6f} + {detector.config.baseline_std_multiplier} × {detector.baseline_std:.6f}")
    print(f"   - Final threshold: {detector.threshold:.6f}")
    print()
    print("3. EVENT DETECTION:")
    print(f"   - Events detected when energy > {detector.threshold:.6f}")
    print(f"   - Results: {len(detector.detected_events)} events detected")
    print()
    print("4. ADAPTIVE vs FIXED APPROACHES:")
    print("   - ADAPTIVE: Threshold calculated from actual data statistics")
    print("   - FIXED: User-defined threshold value")
    print("   - ADAPTIVE is recommended for varying noise conditions")
    
    # Create detailed baseline analysis
    logger.info("Creating threshold analysis visualization...")
    visualizer = EventVisualization(detector)
    visualizer.plot_baseline_analysis()
    
    return detector


def demonstrate_windowing_approaches():
    """
    Demonstrate different windowing approaches.
    """
    logger.info("\n=== WINDOWING APPROACHES DEMONSTRATION ===")
    
    # Generate data with varied event durations
    data_dict = create_synthetic_vibration_data(
        duration=60.0,
        n_events=10,
        noise_level=0.12
    )
    
    # Test adaptive windowing
    config_adaptive = EventDetectionConfig(
        adaptive_window=True,
        energy_fallback_ratio=0.3
    )
    
    detector_adaptive = EventDetector(config_adaptive)
    detector_adaptive.load_data_from_array(data_dict['vibration_data'])
    detector_adaptive.calculate_energy_profile()
    detector_adaptive.calculate_baseline()
    detector_adaptive.calculate_threshold()
    adaptive_events = detector_adaptive.detect_events()
    
    # Test fixed windowing  
    config_fixed = EventDetectionConfig(
        adaptive_window=False
    )
    
    detector_fixed = EventDetector(config_fixed)
    detector_fixed.load_data_from_array(data_dict['vibration_data'])
    detector_fixed.calculate_energy_profile()
    detector_fixed.calculate_baseline()
    detector_fixed.calculate_threshold()
    fixed_events = detector_fixed.detect_events()
    
    print("\n" + "="*60)
    print("WINDOWING APPROACHES COMPARISON")
    print("="*60)
    print("SLIDING WINDOW:")
    print(f"  - Fixed window size: {config_fixed.window_size} samples")
    print(f"  - Window overlap: {config_fixed.window_overlap*100:.0f}%")
    print(f"  - Events detected: {len(fixed_events)}")
    print()
    print("ADAPTIVE WINDOWING:")
    print(f"  - Initial detection with sliding window")
    print(f"  - Event boundaries extended until energy drops to {config_adaptive.energy_fallback_ratio*100:.0f}% of peak")
    print(f"  - Events detected: {len(adaptive_events)}")
    print()
    print("ADVANTAGES:")
    print("  Sliding Window:")
    print("    + Simple and fast")
    print("    + Consistent processing")
    print("    - May truncate events")
    print("  Adaptive Window:")
    print("    + Captures complete events")
    print("    + Better for varying event durations")
    print("    - More computationally intensive")
    
    # Create windowing comparison
    logger.info("Creating windowing comparison visualization...")
    visualizer = EventVisualization(detector_adaptive)
    visualizer.plot_windowing_comparison()
    
    return detector_adaptive, detector_fixed


def demonstrate_interactive_tuning():
    """
    Demonstrate interactive parameter tuning.
    """
    logger.info("\n=== INTERACTIVE PARAMETER TUNING ===")
    
    # Generate data for tuning
    data_dict = create_synthetic_vibration_data(
        duration=40.0,
        n_events=7,
        noise_level=0.13
    )
    
    detector = EventDetector()
    detector.load_data_from_array(data_dict['vibration_data'])
    detector.calculate_energy_profile()
    detector.calculate_baseline()
    detector.calculate_threshold()
    detector.detect_events()
    
    print("\n" + "="*60)
    print("INTERACTIVE PARAMETER TUNING")
    print("="*60)
    print("The interactive interface allows real-time adjustment of:")
    print("  - Baseline percentile (5-30%)")
    print("  - Standard deviation multiplier (1-6)")
    print("  - Minimum event duration (50-1000 samples)")
    print("  - Event merge threshold (0-500 samples)")
    print()
    print("Use the sliders to see immediate effects on event detection.")
    print("The 'Reset' button restores original parameters.")
    print("The 'Export Events' button saves current results to CSV.")
    print()
    
    # Create interactive tuning interface
    logger.info("Creating interactive parameter tuning interface...")
    tuner = InteractiveParameterTuning(detector)
    tuner.create_interactive_plot()
    
    return detector


def demonstrate_comprehensive_analysis():
    """
    Demonstrate comprehensive event analysis with all features.
    """
    logger.info("\n=== COMPREHENSIVE ANALYSIS DEMONSTRATION ===")
    
    # Generate a more complex dataset
    data_dict = create_synthetic_vibration_data(
        duration=90.0,
        n_events=15,
        noise_level=0.18
    )
    
    # Use optimized configuration
    config = EventDetectionConfig(
        baseline_percentile=12.0,
        baseline_std_multiplier=3.5,
        window_size=750,
        min_event_duration=75,
        max_event_duration=5000,
        merge_threshold=150,
        adaptive_window=True,
        filter_enabled=True,
        filter_lowcut=2.0,
        filter_highcut=80.0
    )
    
    detector = EventDetector(config)
    detector.load_data_from_array(data_dict['vibration_data'])
    
    # Full processing pipeline
    logger.info("Running complete analysis pipeline...")
    detector.calculate_energy_profile()
    detector.calculate_baseline()
    detector.calculate_threshold()
    detected_events = detector.detect_events()
    
    # Get comprehensive statistics
    stats = detector.get_event_statistics()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("="*60)
    print(f"Dataset: {data_dict['duration']:.0f}s duration, {len(data_dict['events_metadata'])} true events")
    print(f"Detection: {len(detected_events)} events found")
    print()
    print("CONFIGURATION USED:")
    print(f"  Baseline percentile: {config.baseline_percentile}%")
    print(f"  Std multiplier: {config.baseline_std_multiplier}")
    print(f"  Window size: {config.window_size} samples")
    print(f"  Min event duration: {config.min_event_duration} samples")
    print(f"  Adaptive windowing: {config.adaptive_window}")
    print(f"  Bandpass filter: {config.filter_lowcut}-{config.filter_highcut} Hz")
    print()
    if detected_events:
        print("DETAILED STATISTICS:")
        dur_stats = stats['duration_stats']
        energy_stats = stats['peak_energy_stats']
        print(f"  Event durations: {dur_stats['mean']:.3f}±{dur_stats['std']:.3f}s "
              f"(range: {dur_stats['min']:.3f}-{dur_stats['max']:.3f}s)")
        print(f"  Peak energies: {energy_stats['mean']:.4f}±{energy_stats['std']:.4f} "
              f"(range: {energy_stats['min']:.4f}-{energy_stats['max']:.4f})")
        print(f"  Event rate: {stats['event_rate']:.2f} events/second")
    
    # Create step-by-step visualization
    logger.info("Creating comprehensive step-by-step visualization...")
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    create_step_by_step_visualization(detector, save_dir=str(output_dir))
    
    # Export results
    detector.export_events("detected_events_comprehensive.csv", format='csv')
    detector.export_events("detected_events_comprehensive.h5", format='h5')
    
    logger.info(f"Analysis complete! Results saved to {output_dir}")
    
    return detector, stats


def demonstrate_h5_file_loading():
    """
    Demonstrate loading and analyzing H5 vibration data files.
    """
    logger.info("\n=== H5 FILE LOADING DEMONSTRATION ===")
    
    # First, create an H5 file for demonstration
    logger.info("Creating sample H5 file...")
    data_dict = create_synthetic_vibration_data(duration=25.0, n_events=5)
    h5_filename = "sample_vibration_data.h5"
    save_to_h5(data_dict, h5_filename)
    
    # Now demonstrate loading
    logger.info(f"Loading vibration data from {h5_filename}...")
    
    detector = EventDetector()
    
    # Load data from H5 file
    loaded_data = detector.load_h5_data(h5_filename, dataset_name='vibration_data')
    
    # Process the loaded data
    detector.calculate_energy_profile()
    detector.calculate_baseline()
    detector.calculate_threshold()
    events = detector.detect_events()
    
    print("\n" + "="*60)
    print("H5 FILE ANALYSIS RESULTS")
    print("="*60)
    print(f"File: {h5_filename}")
    print(f"Data shape: {loaded_data.shape}")
    print(f"Duration: {len(loaded_data)/detector.config.sampling_rate:.1f} seconds")
    print(f"Events detected: {len(events)}")
    print()
    print("H5 FILE STRUCTURE:")
    print("  /vibration_data - main signal array")
    print("  /time_axis - time values (optional)")
    print("  /true_events/ - ground truth events (if available)")
    print("  attributes: sampling_rate, duration, etc.")
    
    # Clean up
    Path(h5_filename).unlink()
    
    return detector


def main():
    """
    Main demonstration function - runs all examples.
    """
    print("="*80)
    print("EVENT DETECTION FOR VIBRATION DATA")
    print("COMPREHENSIVE DEMONSTRATION")
    print("="*80)
    
    try:
        # 1. Basic event detection
        detector1, data1 = demonstrate_basic_event_detection()
        
        # 2. Threshold analysis
        detector2 = demonstrate_threshold_analysis()
        
        # 3. Windowing approaches
        detector3, detector4 = demonstrate_windowing_approaches()
        
        # 4. H5 file loading
        detector5 = demonstrate_h5_file_loading()
        
        # 5. Interactive tuning (optional - requires user interaction)
        print("\n" + "="*60)
        print("OPTIONAL: Interactive Parameter Tuning")
        print("="*60)
        response = input("Would you like to try interactive parameter tuning? (y/n): ")
        if response.lower().startswith('y'):
            demonstrate_interactive_tuning()
        
        # 6. Comprehensive analysis
        detector6, stats = demonstrate_comprehensive_analysis()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("All event detection capabilities have been demonstrated:")
        print("✓ Basic event detection with automatic baseline calculation")
        print("✓ Noise threshold understanding and analysis")  
        print("✓ Sliding window vs adaptive windowing comparison")
        print("✓ H5 file loading and processing")
        print("✓ Interactive parameter tuning (optional)")
        print("✓ Comprehensive analysis with step-by-step visualization")
        print("✓ Event statistics and data export")
        print()
        print("Check the 'analysis_output' directory for detailed visualizations!")
        print("Event data has been exported to CSV and H5 formats.")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()