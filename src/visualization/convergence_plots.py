"""
Convergence analysis and performance benchmark plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")


def plot_convergence_analysis(
    grid_sizes: list,
    errors: dict,
    title: str = "Convergence Analysis",
    save_path: str = None
):
    """
    Plot convergence rates for different methods.

    Parameters:
    -----------
    grid_sizes : list
        List of grid sizes (N_S or N_t)
    errors : dict
        Dictionary with method names as keys and error arrays as values
        Format: {'Method Name': [error1, error2, ...]}
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Linear scale
    ax = axes[0]
    for method_name, error_values in errors.items():
        ax.plot(grid_sizes, error_values, marker='o', linewidth=2,
               markersize=8, label=method_name, alpha=0.8)

    ax.set_xlabel('Grid Size (N)', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title(f'{title}\n(Linear Scale)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Log-log scale
    ax = axes[1]
    for method_name, error_values in errors.items():
        ax.loglog(grid_sizes, error_values, marker='o', linewidth=2,
                 markersize=8, label=method_name, alpha=0.8)

    # Add reference lines for convergence orders
    if len(grid_sizes) > 1:
        # O(h) reference line
        ref_h = error_values[0] * (np.array(grid_sizes) / grid_sizes[0])**(-1)
        ax.loglog(grid_sizes, ref_h, 'k--', alpha=0.4, linewidth=1.5,
                 label='O(h) reference')

        # O(h²) reference line
        ref_h2 = error_values[0] * (np.array(grid_sizes) / grid_sizes[0])**(-2)
        ax.loglog(grid_sizes, ref_h2, 'k:', alpha=0.4, linewidth=1.5,
                 label='O(h²) reference')

    ax.set_xlabel('Grid Size (N)', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title(f'{title}\n(Log-Log Scale)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_performance_benchmark(
    methods: list,
    computation_times: list,
    errors: list,
    title: str = "Performance Benchmark",
    save_path: str = None
):
    """
    Plot accuracy vs. computation time.

    Parameters:
    -----------
    methods : list
        List of method names
    computation_times : list
        Computation times for each method (seconds)
    errors : list
        Errors for each method
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = sns.color_palette("husl", len(methods))

    # Plot 1: Computation time comparison
    ax = axes[0]
    bars = ax.barh(methods, computation_times, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for i, (bar, time) in enumerate(zip(bars, computation_times)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'  {time:.4f}s',
               ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Computation Time (seconds)', fontsize=12)
    ax.set_title('Computation Time Comparison', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 2: Accuracy vs. speed (Pareto frontier)
    ax = axes[1]
    for i, method in enumerate(methods):
        ax.scatter(computation_times[i], errors[i],
                  s=200, color=colors[i], alpha=0.7,
                  edgecolor='black', linewidth=1.5,
                  label=method, zorder=3)
        ax.annotate(method,
                   (computation_times[i], errors[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, alpha=0.8)

    ax.set_xlabel('Computation Time (seconds)', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Accuracy vs. Speed Trade-off', fontsize=13)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Add "better" region annotation
    ax.annotate('← Faster, More Accurate ↓',
               xy=(0.02, 0.98), xycoords='axes fraction',
               fontsize=10, alpha=0.6,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_ml_vs_pde_comparison(
    parameter_ranges: dict,
    ml_predictions: np.ndarray,
    pde_solutions: np.ndarray,
    ml_times: np.ndarray,
    pde_times: np.ndarray,
    save_path: str = None
):
    """
    Comprehensive comparison between ML surrogate and PDE solver.

    Parameters:
    -----------
    parameter_ranges : dict
        Dictionary with parameter names and their test values
    ml_predictions : np.ndarray
        ML model predictions
    pde_solutions : np.ndarray
        PDE solver ground truth
    ml_times : np.ndarray
        ML prediction times
    pde_times : np.ndarray
        PDE solution times
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=(16, 12))

    # 1. Prediction accuracy scatter
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(pde_solutions, ml_predictions, alpha=0.5, s=20)
    ax1.plot([pde_solutions.min(), pde_solutions.max()],
            [pde_solutions.min(), pde_solutions.max()],
            'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('PDE Solution (Ground Truth)', fontsize=11)
    ax1.set_ylabel('ML Prediction', fontsize=11)
    ax1.set_title('Prediction Accuracy', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Error distribution
    ax2 = plt.subplot(2, 3, 2)
    errors = np.abs(ml_predictions - pde_solutions)
    ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Absolute Error', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'Error Distribution\nMean: {errors.mean():.4f}, Std: {errors.std():.4f}', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. Relative error distribution
    ax3 = plt.subplot(2, 3, 3)
    rel_errors = errors / (np.abs(pde_solutions) + 1e-10) * 100
    ax3.hist(rel_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax3.set_xlabel('Relative Error (%)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title(f'Relative Error Distribution\nMedian: {np.median(rel_errors):.2f}%', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 4. Speed comparison
    ax4 = plt.subplot(2, 3, 4)
    speedup = pde_times.mean() / ml_times.mean()
    methods = ['PDE Solver', 'ML Surrogate']
    times = [pde_times.mean(), ml_times.mean()]
    colors_bar = ['#e74c3c', '#2ecc71']
    bars = ax4.bar(methods, times, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)

    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{time*1000:.2f}ms',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax4.set_ylabel('Computation Time (seconds)', fontsize=11)
    ax4.set_title(f'Speed Comparison\nSpeedup: {speedup:.1f}x', fontsize=12, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Error vs parameter (if applicable)
    ax5 = plt.subplot(2, 3, 5)
    if len(parameter_ranges) > 0:
        param_name = list(parameter_ranges.keys())[0]
        param_values = parameter_ranges[param_name]

        # Bin errors by parameter value
        bins = 20
        param_bins = np.linspace(param_values.min(), param_values.max(), bins)
        bin_indices = np.digitize(param_values, param_bins)

        bin_means = []
        bin_centers = []
        for i in range(1, bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_means.append(errors[mask].mean())
                bin_centers.append((param_bins[i-1] + param_bins[i]) / 2)

        ax5.plot(bin_centers, bin_means, 'o-', linewidth=2, markersize=6)
        ax5.set_xlabel(f'{param_name}', fontsize=11)
        ax5.set_ylabel('Mean Absolute Error', fontsize=11)
        ax5.set_title(f'Error vs {param_name}', fontsize=12)
        ax5.grid(True, alpha=0.3)

    # 6. Accuracy-Speed tradeoff with zones
    ax6 = plt.subplot(2, 3, 6)

    # Calculate metrics
    accuracy = 1 - (errors.mean() / pde_solutions.mean())
    speed_ratio = pde_times.mean() / ml_times.mean()

    # Plot zones
    ax6.axhspan(0, 0.9, alpha=0.1, color='red', label='Low Accuracy')
    ax6.axhspan(0.9, 0.95, alpha=0.1, color='yellow', label='Medium Accuracy')
    ax6.axhspan(0.95, 1.0, alpha=0.1, color='green', label='High Accuracy')

    # Plot point
    ax6.scatter(speed_ratio, accuracy, s=500, c='blue', marker='*',
               edgecolor='black', linewidth=2, zorder=10,
               label=f'ML Surrogate\n({speed_ratio:.1f}x, {accuracy*100:.2f}%)')

    # Reference line for PDE
    ax6.scatter(1, 1, s=300, c='red', marker='o',
               edgecolor='black', linewidth=2, zorder=10,
               label='PDE Solver (baseline)')

    ax6.set_xlabel('Speedup Factor', fontsize=11)
    ax6.set_ylabel('Accuracy (1 - normalized error)', fontsize=11)
    ax6.set_title('Performance Quadrant Analysis', fontsize=12, fontweight='bold')
    ax6.set_xlim(0, speed_ratio * 1.2)
    ax6.set_ylim(0.85, 1.01)
    ax6.legend(loc='lower right', fontsize=9)
    ax6.grid(True, alpha=0.3)

    plt.suptitle('ML Surrogate vs PDE Solver: Comprehensive Comparison',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig