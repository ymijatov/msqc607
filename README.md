# msqc607
MS QC 607 Class Project

quantum_tda/
├── config.py              # All parameters in one place (dataclasses)
├── signal_generation.py   # Synthetic signals with parameterized drift
├── psd_analysis.py        # Welch PSD, segmentation
├── real_data.py           # Loads real GESL data
├── real_data_experiments.py # Runs full pipeline on real data
├── distance_metrics.py    # All 4 methods (Euclidean, Swap, Trace-C, Trace-Q)
├── tda_analysis.py        # H0 persistence for mode detection
├── regime_detection.py    # Transition detection from distances
├── experiments.py         # Experiment runners (single trial, full study)
├── visualization.py       # Clean plotting functions
├── run_pipeline.py        # CLI entry point
├── test_real.py           # Test real data pipeline
└── __init__.py            # Package exports
