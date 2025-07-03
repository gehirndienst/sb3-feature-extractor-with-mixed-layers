#!/usr/bin/env python3
"""
Quick demonstration of the event detection system.
Run this script to see the system in action with synthetic data.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

from event_detector import EventDetector, EventDetectionConfig
from generate_synthetic_data import create_synthetic_vibration_data
import numpy as np

def main():
    print("🎯 Event Detection for Vibration Data - Quick Demo")
    print("=" * 60)
    
    # Generate synthetic vibration data
    print("📊 Generating synthetic vibration data...")
    data_dict = create_synthetic_vibration_data(
        duration=30.0,      # 30 seconds
        n_events=6,         # 6 embedded events  
        noise_level=0.15    # Medium noise level
    )
    print(f"   ✓ Generated {len(data_dict['vibration_data']):,} samples")
    print(f"   ✓ Embedded {len(data_dict['events_metadata'])} true events")
    
    # Create and configure event detector
    print("\n🔧 Configuring event detector...")
    config = EventDetectionConfig(
        baseline_percentile=15.0,       # Use bottom 15% for baseline
        baseline_std_multiplier=3.0,    # Threshold = baseline + 3×std
        window_size=800,                # 800-sample sliding windows
        adaptive_window=True,           # Use adaptive windowing
        filter_enabled=True,            # Enable bandpass filtering
        min_event_duration=50           # Minimum 50 samples for valid events
    )
    
    detector = EventDetector(config)
    print("   ✓ Detector configured")
    
    # Load data and run detection pipeline
    print("\n🔍 Running event detection pipeline...")
    
    # Step 1: Load data
    detector.load_data_from_array(data_dict['vibration_data'], sampling_rate=1000.0)
    print("   ✓ Data loaded")
    
    # Step 2: Calculate energy profile  
    energy_profile = detector.calculate_energy_profile()
    print("   ✓ Energy profile calculated")
    
    # Step 3: Calculate baseline from quiet periods
    baseline_mean, baseline_std = detector.calculate_baseline()
    print(f"   ✓ Baseline: {baseline_mean:.6f} ± {baseline_std:.6f}")
    
    # Step 4: Calculate detection threshold
    threshold = detector.calculate_threshold()
    print(f"   ✓ Threshold: {threshold:.6f}")
    
    # Step 5: Detect events
    detected_events = detector.detect_events()
    print(f"   ✓ Events detected: {len(detected_events)}")
    
    # Analyze results
    print("\n📈 Analysis Results:")
    print("-" * 40)
    
    # Get comprehensive statistics
    stats = detector.get_event_statistics()
    
    print(f"Data Duration:      {data_dict['duration']:.1f} seconds")
    print(f"Sampling Rate:      {detector.config.sampling_rate:.0f} Hz")
    print(f"Total Samples:      {len(detector.data):,}")
    print()
    print("Baseline Analysis:")
    print(f"  Quiet Percentile: {detector.config.baseline_percentile}%")
    print(f"  Baseline Mean:    {baseline_mean:.6f}")
    print(f"  Baseline Std:     {baseline_std:.6f}")
    print(f"  Threshold:        {threshold:.6f}")
    print()
    print("Event Detection:")
    print(f"  True Events:      {len(data_dict['events_metadata'])}")
    print(f"  Detected Events:  {len(detected_events)}")
    print(f"  Detection Rate:   {len(detected_events)/len(data_dict['events_metadata']):.1%}")
    print(f"  Event Rate:       {stats['event_rate']:.2f} events/second")
    
    if detected_events:
        print()
        print("Event Statistics:")
        dur_stats = stats['duration_stats']
        energy_stats = stats['peak_energy_stats']
        print(f"  Avg Duration:     {dur_stats['mean']:.3f}s (range: {dur_stats['min']:.3f}-{dur_stats['max']:.3f}s)")
        print(f"  Avg Peak Energy:  {energy_stats['mean']:.4f} (range: {energy_stats['min']:.4f}-{energy_stats['max']:.4f})")
        
        print()
        print("Individual Events:")
        for i, event in enumerate(detected_events[:5]):  # Show first 5 events
            duration_s = event.duration / detector.config.sampling_rate
            start_time = event.start_idx / detector.config.sampling_rate
            print(f"  Event {i+1}: t={start_time:.2f}s, duration={duration_s:.3f}s, peak_energy={event.peak_energy:.4f}")
        
        if len(detected_events) > 5:
            print(f"  ... and {len(detected_events) - 5} more events")
    
    # Export results
    print("\n💾 Exporting results...")
    detector.export_events('demo_events.csv', format='csv')
    print("   ✓ Events exported to demo_events.csv")
    
    # Show configuration effectiveness
    print("\n⚙️  Configuration Effectiveness:")
    detection_accuracy = min(len(detected_events) / len(data_dict['events_metadata']), 1.0)
    if detection_accuracy >= 0.8:
        print("   🎯 Excellent detection rate!")
    elif detection_accuracy >= 0.6:
        print("   👍 Good detection rate")
    else:
        print("   ⚠️  Consider adjusting threshold parameters")
    
    print(f"   Threshold effectiveness: {detection_accuracy:.1%}")
    print()
    
    # Usage suggestions
    print("💡 Next Steps:")
    print("   • Run 'python vibration_analysis_example.py' for full demo")
    print("   • Try 'python generate_synthetic_data.py' to create test datasets")
    print("   • Load your own H5 files with detector.load_h5_data()")
    print("   • Use interactive tuning for parameter optimization")
    print("   • Export results to CSV/H5 for further analysis")
    
    print(f"\n🎉 Demo completed successfully!")
    print("   Event detection system is ready for your vibration data!")

if __name__ == "__main__":
    main()