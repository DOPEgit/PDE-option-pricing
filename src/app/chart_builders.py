"""
Chart building functions for the Option Pricing Dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


def create_price_sensitivity_chart(S_range: np.ndarray, call_prices: np.ndarray,
                                  put_prices: np.ndarray, K: float, S0: float) -> go.Figure:
    """
    Create a price sensitivity chart showing option values vs stock price.

    Parameters:
    -----------
    S_range : np.ndarray
        Range of stock prices
    call_prices : np.ndarray
        Call option prices
    put_prices : np.ndarray
        Put option prices
    K : float
        Strike price
    S0 : float
        Current spot price

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()

    # Add call prices
    fig.add_trace(go.Scatter(
        x=S_range,
        y=call_prices,
        mode='lines',
        name='Call',
        line=dict(color='green', width=2),
        hovertemplate='Stock: $%{x:.2f}<br>Call Value: $%{y:.2f}<extra></extra>'
    ))

    # Add put prices
    fig.add_trace(go.Scatter(
        x=S_range,
        y=put_prices,
        mode='lines',
        name='Put',
        line=dict(color='red', width=2),
        hovertemplate='Stock: $%{x:.2f}<br>Put Value: $%{y:.2f}<extra></extra>'
    ))

    # Add strike line
    fig.add_vline(
        x=K,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Strike: ${K:.0f}"
    )

    # Add current spot line
    fig.add_vline(
        x=S0,
        line_dash="dot",
        line_color="blue",
        annotation_text=f"Current: ${S0:.0f}"
    )

    fig.update_layout(
        title="Option Value vs Stock Price",
        xaxis_title="Stock Price ($)",
        yaxis_title="Option Value ($)",
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def create_greeks_surface(S_mesh: np.ndarray, T_mesh: np.ndarray,
                         Z: np.ndarray, greek_name: str) -> go.Figure:
    """
    Create a 3D surface plot for Greeks.

    Parameters:
    -----------
    S_mesh : np.ndarray
        Stock price mesh
    T_mesh : np.ndarray
        Time to maturity mesh
    Z : np.ndarray
        Greek values
    greek_name : str
        Name of the Greek

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Choose colorscale based on Greek
    colorscales = {
        'Delta': 'Viridis',
        'Gamma': 'Plasma',
        'Theta': 'RdBu_r',  # Reversed for negative values
        'Vega': 'Cividis',
        'Price': 'Viridis'
    }

    colorscale = colorscales.get(greek_name, 'Viridis')

    fig = go.Figure(data=[go.Surface(
        x=S_mesh,
        y=T_mesh,
        z=Z,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title=greek_name,
            thickness=20,
            len=0.7
        ),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project=dict(z=True)
            )
        )
    )])

    fig.update_layout(
        title=f"{greek_name} Surface",
        scene=dict(
            xaxis_title="Stock Price ($)",
            yaxis_title="Time to Maturity (years)",
            zaxis_title=greek_name,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray'),
            zaxis=dict(gridcolor='lightgray')
        ),
        autosize=True,
        height=600
    )

    return fig


def create_greeks_heatmap(S_range: np.ndarray, T_range: np.ndarray,
                         Z: np.ndarray, greek_name: str) -> go.Figure:
    """
    Create a heatmap for Greeks visualization.

    Parameters:
    -----------
    S_range : np.ndarray
        Range of stock prices
    T_range : np.ndarray
        Range of times to maturity
    Z : np.ndarray
        Greek values
    greek_name : str
        Name of the Greek

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Choose colorscale
    if greek_name == 'Theta':
        colorscale = 'RdBu_r'
    else:
        colorscale = 'Viridis'

    fig = px.imshow(
        Z,
        labels=dict(x="Stock Price ($)", y="Time to Maturity (years)", color=greek_name),
        x=S_range,
        y=T_range,
        aspect="auto",
        color_continuous_scale=colorscale
    )

    fig.update_layout(
        title=f"{greek_name} Heatmap",
        height=400,
        template='plotly_white'
    )

    return fig


def create_pnl_chart(spot_changes: np.ndarray, pnl_scenarios: List[float]) -> go.Figure:
    """
    Create a P&L chart for portfolio analysis.

    Parameters:
    -----------
    spot_changes : np.ndarray
        Array of spot price changes (in %)
    pnl_scenarios : List[float]
        P&L values for each scenario

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Determine colors for positive/negative regions
    colors = ['green' if pnl > 0 else 'red' for pnl in pnl_scenarios]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=spot_changes,
        y=pnl_scenarios,
        mode='lines',
        name='P&L',
        line=dict(width=3, color='blue'),
        fill='tozeroy',
        fillcolor='rgba(0, 100, 255, 0.2)',
        hovertemplate='Price Change: %{x:.1f}%<br>P&L: $%{y:,.2f}<extra></extra>'
    ))

    # Add breakeven line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Breakeven"
    )

    # Mark max profit and max loss
    max_profit_idx = np.argmax(pnl_scenarios)
    max_loss_idx = np.argmin(pnl_scenarios)

    fig.add_trace(go.Scatter(
        x=[spot_changes[max_profit_idx]],
        y=[pnl_scenarios[max_profit_idx]],
        mode='markers',
        marker=dict(size=10, color='green', symbol='diamond'),
        name='Max Profit',
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=[spot_changes[max_loss_idx]],
        y=[pnl_scenarios[max_loss_idx]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Max Loss',
        showlegend=True
    ))

    fig.update_layout(
        title="Portfolio P&L vs Stock Price Change",
        xaxis_title="Stock Price Change (%)",
        yaxis_title="P&L ($)",
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig


def create_volatility_smile(strikes: np.ndarray, ivs: np.ndarray, S0: float) -> go.Figure:
    """
    Create a volatility smile chart.

    Parameters:
    -----------
    strikes : np.ndarray
        Strike prices
    ivs : np.ndarray
        Implied volatilities
    S0 : float
        Current spot price

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=strikes,
        y=ivs * 100,  # Convert to percentage
        mode='lines+markers',
        name='IV Smile',
        line=dict(width=3, color='purple'),
        marker=dict(size=8),
        hovertemplate='Strike: $%{x:.0f}<br>IV: %{y:.1f}%<extra></extra>'
    ))

    # Add ATM marker
    fig.add_vline(
        x=S0,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"ATM: ${S0:.0f}"
    )

    fig.update_layout(
        title="Implied Volatility Smile",
        xaxis_title="Strike Price ($)",
        yaxis_title="Implied Volatility (%)",
        template='plotly_white',
        hovermode='x'
    )

    return fig


def create_performance_comparison(df_results: pd.DataFrame) -> go.Figure:
    """
    Create a scatter plot comparing ML vs PDE predictions.

    Parameters:
    -----------
    df_results : pd.DataFrame
        DataFrame with PDE and ML results

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=df_results['PDE_price'],
        y=df_results['ML_price'],
        mode='markers',
        marker=dict(
            color=df_results['error_pct'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Error %", thickness=15),
            size=8,
            opacity=0.7
        ),
        text=[f"Error: {e:.2f}%" for e in df_results['error_pct']],
        hovertemplate="PDE: $%{x:.2f}<br>ML: $%{y:.2f}<br>%{text}<extra></extra>",
        name='Predictions'
    ))

    # Add perfect prediction line
    min_price = min(df_results['PDE_price'].min(), df_results['ML_price'].min())
    max_price = max(df_results['PDE_price'].max(), df_results['ML_price'].max())

    fig.add_trace(go.Scatter(
        x=[min_price, max_price],
        y=[min_price, max_price],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Prediction',
        showlegend=True,
        hoverinfo='skip'
    ))

    # Add R² annotation
    from sklearn.metrics import r2_score
    r2 = r2_score(df_results['PDE_price'], df_results['ML_price'])

    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"R² = {r2:.4f}",
        showarrow=False,
        font=dict(size=14),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    fig.update_layout(
        title="ML vs PDE Predictions",
        xaxis_title="PDE Price ($)",
        yaxis_title="ML Price ($)",
        template='plotly_white',
        hovermode='closest',
        showlegend=True
    )

    return fig


def create_time_comparison_box(df_results: pd.DataFrame) -> go.Figure:
    """
    Create box plot comparing computation times.

    Parameters:
    -----------
    df_results : pd.DataFrame
        DataFrame with timing results

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Box(
        y=df_results['PDE_time'] * 1000,  # Convert to ms
        name='PDE Solver',
        marker_color='blue',
        boxmean='sd'  # Show mean and standard deviation
    ))

    fig.add_trace(go.Box(
        y=df_results['ML_time'] * 1000,  # Convert to ms
        name='ML Model',
        marker_color='green',
        boxmean='sd'
    ))

    fig.update_layout(
        title="Computation Time Comparison",
        yaxis_title="Time (ms)",
        template='plotly_white',
        yaxis_type="log",
        showlegend=False
    )

    # Add speedup annotation
    avg_speedup = df_results['PDE_time'].mean() / df_results['ML_time'].mean()

    fig.add_annotation(
        x=0.5,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"Average Speedup: {avg_speedup:.0f}x",
        showarrow=False,
        font=dict(size=14, color="green"),
        bgcolor="white",
        bordercolor="green",
        borderwidth=1
    )

    return fig


def create_error_distribution(df_results: pd.DataFrame) -> go.Figure:
    """
    Create histogram of prediction errors.

    Parameters:
    -----------
    df_results : pd.DataFrame
        DataFrame with error results

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df_results['error_pct'],
        nbinsx=30,
        name='Error Distribution',
        marker=dict(
            color='purple',
            line=dict(color='black', width=1)
        ),
        opacity=0.7,
        hovertemplate='Error Range: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ))

    # Add mean and median lines
    mean_error = df_results['error_pct'].mean()
    median_error = df_results['error_pct'].median()

    fig.add_vline(
        x=mean_error,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_error:.2f}%"
    )

    fig.add_vline(
        x=median_error,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Median: {median_error:.2f}%"
    )

    fig.update_layout(
        title="Prediction Error Distribution",
        xaxis_title="Error (%)",
        yaxis_title="Count",
        template='plotly_white',
        showlegend=False
    )

    return fig