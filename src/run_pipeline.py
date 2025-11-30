#!/usr/bin/env python3
"""
Quantum-Enhanced TDA for Power Grid Oscillation Detection

Main entry point for running experiments.

Usage:
    python run_pipeline.py                    # Run full three-mode study
    python run_pipeline.py --quick            # Quick test (5 trials)
    python run_pipeline.py --weak-only        # Only weak mode drift
    python run_pipeline.py --trials 50        # Custom trial count
"""

import argparse
import os
import matplotlib.pyplot as plt

from config import (ExperimentConfig, 
                    make_strong_mode_drift_config,
                    make_medium_mode_drift_config,
                    make_weak_mode_drift_config)
from experiments import run_experiment, run_three_mode_drift_study, summarize_study
from visualization import (plot_method_comparison, plot_three_mode_study,
                           plot_distance_correlation, plot_single_trial)
from signal_generation import generate_signal, get_mode_info


def ensure_output_dir():
    """Create figures directory if needed."""
    os.makedirs('figures', exist_ok=True)


def run_quick_test():
    """Quick smoke test with minimal trials."""
    print("\n" + "="*60)
    print("QUICK TEST (5 trials, weak mode only)")
    print("="*60)
    
    config = make_weak_mode_drift_config()
    config.n_trials = 5
    
    print(get_mode_info(config.signal))
    
    results = run_experiment(config, verbose=True)
    
    # Plot
    fig = plot_method_comparison(results)
    ensure_output_dir()
    fig.savefig('figures/quick_test_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: figures/quick_test_comparison.png")
    
    return results


def run_single_experiment(drift_type: str, n_trials: int = 20):
    """Run experiment for a single drift type."""
    configs = {
        'strong': make_strong_mode_drift_config,
        'medium': make_medium_mode_drift_config,
        'weak': make_weak_mode_drift_config,
    }
    
    if drift_type not in configs:
        raise ValueError(f"Unknown drift type: {drift_type}. Use: strong, medium, weak")
    
    config = configs[drift_type]()
    config.n_trials = n_trials
    
    print(get_mode_info(config.signal))
    
    results = run_experiment(config, verbose=True)
    
    # Plots
    ensure_output_dir()
    
    fig1 = plot_method_comparison(results)
    fig1.savefig(f'figures/{drift_type}_mode_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: figures/{drift_type}_mode_comparison.png")
    
    fig2 = plot_distance_correlation(results)
    fig2.savefig(f'figures/{drift_type}_mode_correlation.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: figures/{drift_type}_mode_correlation.png")
    
    return results


def run_full_study(n_trials: int = 20):
    """Run complete three-mode-drift study."""
    all_results = run_three_mode_drift_study(n_trials=n_trials, verbose=True)
    
    # Summary
    summarize_study(all_results)
    
    # Plots
    ensure_output_dir()
    
    fig = plot_three_mode_study(all_results)
    fig.savefig('figures/three_mode_study_summary.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: figures/three_mode_study_summary.png")
    
    # Individual comparisons
    for drift_type, results in all_results.items():
        fig = plot_method_comparison(results)
        fig.savefig(f'figures/{drift_type}_mode_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: figures/{drift_type}_mode_comparison.png")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Quantum-Enhanced TDA for Power Grid Oscillation Detection'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 5 trials')
    parser.add_argument('--weak-only', action='store_true',
                        help='Only run weak mode drift experiment')
    parser.add_argument('--medium-only', action='store_true',
                        help='Only run medium mode drift experiment')
    parser.add_argument('--strong-only', action='store_true',
                        help='Only run strong mode drift experiment')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of trials per experiment (default: 20)')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    
    args = parser.parse_args()
    
    if args.quick:
        results = run_quick_test()
    elif args.weak_only:
        results = run_single_experiment('weak', n_trials=args.trials)
    elif args.medium_only:
        results = run_single_experiment('medium', n_trials=args.trials)
    elif args.strong_only:
        results = run_single_experiment('strong', n_trials=args.trials)
    else:
        results = run_full_study(n_trials=args.trials)
    
    if args.show:
        plt.show()
    
    print("\nDone!")
    return results


if __name__ == '__main__':
    main()
