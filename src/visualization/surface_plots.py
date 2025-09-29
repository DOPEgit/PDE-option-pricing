"""
3D Surface plots for option values and Greeks.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_option_surface(
    S_grid: np.ndarray,
    t_grid: np.ndarray,
    V: np.ndarray,
    title: str = "Option Value Surface",
    save_path: str = None,
    view_angles: tuple = (30, 45)
):
    """
    Plot 3D surface of option values.

    Parameters:
    -----------
    S_grid : np.ndarray
        Stock price grid
    t_grid : np.ndarray
        Time grid
    V : np.ndarray
        Option values
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    view_angles : tuple
        Elevation and azimuth angles for 3D view
    """
    fig = plt.figure(figsize=(14, 10))

    # Create meshgrid
    T_mesh, S_mesh = np.meshgrid(t_grid, S_grid)

    # Main 3D plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax1.plot_surface(S_mesh, T_mesh, V,
                            cmap=cm.viridis,
                            linewidth=0,
                            antialiased=True,
                            alpha=0.9)

    ax1.set_xlabel('Stock Price ($)', fontsize=11, labelpad=10)
    ax1.set_ylabel('Time (years)', fontsize=11, labelpad=10)
    ax1.set_zlabel('Option Value ($)', fontsize=11, labelpad=10)
    ax1.set_title(f'{title}\n(View 1)', fontsize=12, pad=20)
    ax1.view_init(elev=view_angles[0], azim=view_angles[1])
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # Alternative view
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(S_mesh, T_mesh, V,
                             cmap=cm.plasma,
                             linewidth=0,
                             antialiased=True,
                             alpha=0.9)

    ax2.set_xlabel('Stock Price ($)', fontsize=11, labelpad=10)
    ax2.set_ylabel('Time (years)', fontsize=11, labelpad=10)
    ax2.set_zlabel('Option Value ($)', fontsize=11, labelpad=10)
    ax2.set_title(f'{title}\n(View 2)', fontsize=12, pad=20)
    ax2.view_init(elev=20, azim=120)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    # Contour plot
    ax3 = fig.add_subplot(2, 2, 3)
    contour = ax3.contourf(T_mesh, S_mesh, V, levels=20, cmap=cm.viridis)
    ax3.set_xlabel('Time (years)', fontsize=11)
    ax3.set_ylabel('Stock Price ($)', fontsize=11)
    ax3.set_title('Contour Plot', fontsize=12)
    fig.colorbar(contour, ax=ax3)

    # Option value evolution at fixed times
    ax4 = fig.add_subplot(2, 2, 4)
    time_indices = [0, len(t_grid)//4, len(t_grid)//2, 3*len(t_grid)//4, -1]
    for idx in time_indices:
        ax4.plot(S_grid, V[:, idx], label=f't = {t_grid[idx]:.2f}', linewidth=2)

    ax4.set_xlabel('Stock Price ($)', fontsize=11)
    ax4.set_ylabel('Option Value ($)', fontsize=11)
    ax4.set_title('Value Evolution Over Time', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_greeks_surface(
    S_grid: np.ndarray,
    t_grid: np.ndarray,
    delta: np.ndarray,
    gamma: np.ndarray,
    theta: np.ndarray = None,
    title_prefix: str = "Option",
    save_path: str = None
):
    """
    Plot surfaces for Delta, Gamma, and Theta.

    Parameters:
    -----------
    S_grid : np.ndarray
        Stock price grid
    t_grid : np.ndarray
        Time grid
    delta : np.ndarray
        Delta values
    gamma : np.ndarray
        Gamma values
    theta : np.ndarray, optional
        Theta values
    title_prefix : str
        Prefix for titles
    save_path : str, optional
        Path to save figure
    """
    num_plots = 3 if theta is not None else 2
    fig = plt.figure(figsize=(16, 6))

    T_mesh, S_mesh = np.meshgrid(t_grid, S_grid)

    # Delta surface
    ax1 = fig.add_subplot(1, num_plots, 1, projection='3d')
    surf1 = ax1.plot_surface(S_mesh, T_mesh, delta,
                             cmap=cm.coolwarm,
                             linewidth=0,
                             antialiased=True,
                             alpha=0.9)
    ax1.set_xlabel('Stock Price ($)', labelpad=10)
    ax1.set_ylabel('Time (years)', labelpad=10)
    ax1.set_zlabel('Delta (∂V/∂S)', labelpad=10)
    ax1.set_title(f'{title_prefix} Delta (Δ)', fontsize=13, pad=15)
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # Gamma surface
    ax2 = fig.add_subplot(1, num_plots, 2, projection='3d')
    surf2 = ax2.plot_surface(S_mesh, T_mesh, gamma,
                             cmap=cm.RdYlGn,
                             linewidth=0,
                             antialiased=True,
                             alpha=0.9)
    ax2.set_xlabel('Stock Price ($)', labelpad=10)
    ax2.set_ylabel('Time (years)', labelpad=10)
    ax2.set_zlabel('Gamma (∂²V/∂S²)', labelpad=10)
    ax2.set_title(f'{title_prefix} Gamma (Γ)', fontsize=13, pad=15)
    ax2.view_init(elev=25, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    # Theta surface (if provided)
    if theta is not None:
        ax3 = fig.add_subplot(1, num_plots, 3, projection='3d')
        surf3 = ax3.plot_surface(S_mesh, T_mesh, theta,
                                cmap=cm.seismic,
                                linewidth=0,
                                antialiased=True,
                                alpha=0.9)
        ax3.set_xlabel('Stock Price ($)', labelpad=10)
        ax3.set_ylabel('Time (years)', labelpad=10)
        ax3.set_zlabel('Theta (∂V/∂t)', labelpad=10)
        ax3.set_title(f'{title_prefix} Theta (Θ)', fontsize=13, pad=15)
        ax3.view_init(elev=25, azim=45)
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_comparison(
    S_grid: np.ndarray,
    methods_data: dict,
    K: float,
    title: str = "Method Comparison",
    save_path: str = None
):
    """
    Compare option values from different methods.

    Parameters:
    -----------
    S_grid : np.ndarray
        Stock price grid
    methods_data : dict
        Dictionary with keys as method names and values as option values
        Format: {'Method Name': option_values_array}
    K : float
        Strike price
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: All methods at t=0
    ax = axes[0, 0]
    for method_name, V in methods_data.items():
        ax.plot(S_grid, V[:, 0], label=method_name, linewidth=2, alpha=0.8)

    ax.axvline(x=K, color='red', linestyle='--', alpha=0.5, label=f'Strike K={K}')
    ax.set_xlabel('Stock Price ($)', fontsize=11)
    ax.set_ylabel('Option Value ($)', fontsize=11)
    ax.set_title('Option Value at t=0', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Absolute differences from first method
    ax = axes[0, 1]
    reference_method = list(methods_data.keys())[0]
    reference_V = methods_data[reference_method]

    for method_name, V in methods_data.items():
        if method_name != reference_method:
            diff = np.abs(V[:, 0] - reference_V[:, 0])
            ax.plot(S_grid, diff, label=f'{method_name} vs {reference_method}',
                   linewidth=2, alpha=0.8)

    ax.set_xlabel('Stock Price ($)', fontsize=11)
    ax.set_ylabel('Absolute Difference ($)', fontsize=11)
    ax.set_title('Absolute Differences', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: Relative errors
    ax = axes[1, 0]
    for method_name, V in methods_data.items():
        if method_name != reference_method:
            rel_error = np.abs((V[:, 0] - reference_V[:, 0]) / (reference_V[:, 0] + 1e-10)) * 100
            ax.plot(S_grid, rel_error, label=f'{method_name}', linewidth=2, alpha=0.8)

    ax.set_xlabel('Stock Price ($)', fontsize=11)
    ax.set_ylabel('Relative Error (%)', fontsize=11)
    ax.set_title(f'Relative Error vs {reference_method}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Near-the-money zoom
    ax = axes[1, 1]
    K_idx = np.argmin(np.abs(S_grid - K))
    zoom_range = 20  # Points around strike

    for method_name, V in methods_data.items():
        ax.plot(S_grid[K_idx-zoom_range:K_idx+zoom_range],
               V[K_idx-zoom_range:K_idx+zoom_range, 0],
               label=method_name, linewidth=2, alpha=0.8, marker='o', markersize=3)

    ax.axvline(x=K, color='red', linestyle='--', alpha=0.5, label=f'Strike K={K}')
    ax.set_xlabel('Stock Price ($)', fontsize=11)
    ax.set_ylabel('Option Value ($)', fontsize=11)
    ax.set_title('Near-the-Money Detail', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig