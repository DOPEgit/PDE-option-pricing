"""
Interactive Option Pricing Dashboard
====================================
A Streamlit app for real-time option pricing using PDE solvers and ML surrogates.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.pde_solvers.black_scholes import BlackScholesPDE
from src.numerical_methods.crank_nicolson import CrankNicolson
from src.numerical_methods.implicit_fd import ImplicitFD
from src.numerical_methods.explicit_fd import ExplicitFD
from src.ml_models.surrogate_models import OptionPricingSurrogate, MultiOutputSurrogate
from src.market_data.yahoo_fetcher import YahooDataFetcher
from src.market_data.volatility_calc import VolatilityCalculator
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Option Pricing Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None
if 'yahoo_fetcher' not in st.session_state:
    st.session_state.yahoo_fetcher = None

@st.cache_resource
def load_ml_models():
    """Load pre-trained ML models if available."""
    models = {}
    model_paths = {
        'xgboost': 'data/models/xgboost/price_model.joblib',
        'random_forest': 'data/models/random_forest/price_model.joblib'
    }

    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                # Load the saved model data (which is a dict)
                model_data = joblib.load(path)
                # Extract the actual model and metadata
                models[name] = {
                    'model': model_data['model'],
                    'scaler': model_data['scaler'],
                    'feature_names': model_data.get('feature_names', None)
                }
                st.success(f"✓ Loaded {name} model")
            except Exception as e:
                st.warning(f"Could not load {name} model: {e}")

    return models

@st.cache_resource
def init_data_fetcher():
    """Initialize Yahoo Finance data fetcher."""
    try:
        fetcher = YahooDataFetcher()
        return fetcher
    except Exception as e:
        st.warning(f"Could not initialize data fetcher: {e}")
        return None

def calculate_option_pde(S0, K, T, r, sigma, option_type='call', method='crank_nicolson'):
    """Calculate option price using PDE solver."""
    # Create PDE instance
    pde = BlackScholesPDE(
        S_max=3*K,
        T=T,
        r=r,
        sigma=sigma,
        N_S=100,
        N_t=1000
    )

    # Select solver
    if method == 'crank_nicolson':
        solver = CrankNicolson(pde)
    elif method == 'implicit':
        solver = ImplicitFD(pde)
    else:
        solver = ExplicitFD(pde)

    # Set payoff and boundary conditions
    if option_type == 'call':
        payoff = pde.european_call_payoff(K)
        boundary_func = lambda t_idx: pde.apply_boundary_conditions_call(K, t_idx)
    else:
        payoff = pde.european_put_payoff(K)
        boundary_func = lambda t_idx: pde.apply_boundary_conditions_put(K, t_idx)

    # Solve
    start_time = time.time()
    solver.solve(payoff, boundary_func)
    solve_time = time.time() - start_time

    # Get option value at S0
    S_idx = np.abs(pde.S_grid - S0).argmin()
    price = pde.V[S_idx, -1]

    # Calculate Greeks
    # Delta and Gamma take time index, return array for all stock prices
    delta = pde.calculate_delta(t_idx=-1)[S_idx]
    gamma = pde.calculate_gamma(t_idx=-1)[S_idx]
    # Theta takes stock price index, returns array for all times
    theta = pde.calculate_theta(S_idx=S_idx)[-1]

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'time': solve_time
    }

def calculate_option_ml(S0, K, T, r, sigma, model_data):
    """Calculate option price using ML surrogate model."""
    if model_data is None:
        return None

    # Prepare features
    features = pd.DataFrame({
        'S0': [S0],
        'K': [K],
        'T': [T],
        'r': [r],
        'sigma': [sigma],
        'moneyness': [S0/K],
        'log_moneyness': [np.log(S0/K)],
        'sqrt_T': [np.sqrt(T)],
        'vol_sqrt_T': [sigma * np.sqrt(T)]
    })

    # Apply scaler if available
    if model_data.get('scaler') is not None:
        features_scaled = model_data['scaler'].transform(features)
    else:
        features_scaled = features

    start_time = time.time()
    # Use the actual model from the model_data dict
    price = model_data['model'].predict(features_scaled)[0]
    predict_time = time.time() - start_time

    return {
        'price': price,
        'time': predict_time
    }

def create_greeks_surface(S_range, T_range, K, r, sigma):
    """Create 3D surface plots for Greeks."""
    S_mesh, T_mesh = np.meshgrid(S_range, T_range)

    # Initialize arrays for Greeks
    delta_surface = np.zeros_like(S_mesh)
    gamma_surface = np.zeros_like(S_mesh)
    theta_surface = np.zeros_like(S_mesh)
    price_surface = np.zeros_like(S_mesh)

    # Calculate Greeks for each point
    for i, T in enumerate(T_range):
        for j, S in enumerate(S_range):
            result = calculate_option_pde(S, K, T, r, sigma, 'call', 'crank_nicolson')
            price_surface[i, j] = result['price']
            delta_surface[i, j] = result['delta']
            gamma_surface[i, j] = result['gamma']
            theta_surface[i, j] = result['theta']

    return S_mesh, T_mesh, price_surface, delta_surface, gamma_surface, theta_surface

# Main app
def main():
    st.title("📈 Interactive Option Pricing Dashboard")
    st.markdown("Real-time option pricing using PDE solvers and ML surrogates")

    # Sidebar for parameters
    with st.sidebar:
        st.header("⚙️ Parameters")

        st.subheader("Market Data")
        ticker = st.text_input("Ticker Symbol", value="AAPL")
        use_live_data = st.checkbox("Use Live Data", value=False)

        if use_live_data and st.session_state.yahoo_fetcher is None:
            st.session_state.yahoo_fetcher = init_data_fetcher()

        st.subheader("Option Parameters")
        S0 = st.number_input("Spot Price (S₀)", value=100.0, min_value=1.0, step=1.0)
        K = st.number_input("Strike Price (K)", value=100.0, min_value=1.0, step=1.0)
        T = st.slider("Time to Maturity (years)", 0.1, 2.0, 1.0, 0.1)
        r = st.slider("Risk-free Rate", 0.01, 0.10, 0.05, 0.01)
        sigma = st.slider("Volatility", 0.10, 0.50, 0.20, 0.01)

        st.subheader("Solver Settings")
        pde_method = st.selectbox(
            "PDE Method",
            ["crank_nicolson", "implicit", "explicit"]
        )

        st.subheader("ML Model")
        available_models = load_ml_models()
        if available_models:
            model_choice = st.selectbox(
                "ML Model",
                list(available_models.keys())
            )
            st.session_state.ml_model = available_models.get(model_choice)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Real-Time Pricing",
        "📈 Interactive Greeks",
        "📉 Historical Analysis",
        "💼 Portfolio Risk",
        "🤖 Model Performance"
    ])

    with tab1:
        st.header("Real-Time Option Pricing")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("European Call Option")

            # Calculate using PDE
            call_pde = calculate_option_pde(S0, K, T, r, sigma, 'call', pde_method)

            # Calculate using ML if available
            call_ml = None
            if st.session_state.ml_model:
                call_ml = calculate_option_ml(S0, K, T, r, sigma, st.session_state.ml_model)

            # Display results
            metrics_col1, metrics_col2 = st.columns(2)

            with metrics_col1:
                st.metric("PDE Price", f"${call_pde['price']:.4f}")
                st.metric("Delta", f"{call_pde['delta']:.4f}")
                st.metric("Computation Time", f"{call_pde['time']*1000:.2f} ms")

            with metrics_col2:
                if call_ml:
                    st.metric("ML Price", f"${call_ml['price']:.4f}")
                    error = abs(call_ml['price'] - call_pde['price']) / call_pde['price'] * 100
                    st.metric("Error", f"{error:.2f}%")
                    st.metric("Speedup", f"{call_pde['time']/call_ml['time']:.0f}x")

        with col2:
            st.subheader("European Put Option")

            # Calculate using PDE
            put_pde = calculate_option_pde(S0, K, T, r, sigma, 'put', pde_method)

            # Display results
            st.metric("PDE Price", f"${put_pde['price']:.4f}")
            st.metric("Delta", f"{put_pde['delta']:.4f}")
            st.metric("Gamma", f"{put_pde['gamma']:.4f}")
            st.metric("Theta", f"{put_pde['theta']:.4f}")

        # Price sensitivity chart
        st.subheader("Price Sensitivity")

        S_range = np.linspace(0.5*K, 1.5*K, 50)
        call_prices = []
        put_prices = []

        for S in S_range:
            call_result = calculate_option_pde(S, K, T, r, sigma, 'call', 'crank_nicolson')
            put_result = calculate_option_pde(S, K, T, r, sigma, 'put', 'crank_nicolson')
            call_prices.append(call_result['price'])
            put_prices.append(put_result['price'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=S_range,
            y=call_prices,
            mode='lines',
            name='Call',
            line=dict(color='green', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=S_range,
            y=put_prices,
            mode='lines',
            name='Put',
            line=dict(color='red', width=2)
        ))
        fig.add_vline(x=K, line_dash="dash", line_color="gray", annotation_text="Strike")
        fig.add_vline(x=S0, line_dash="dot", line_color="blue", annotation_text="Current")

        fig.update_layout(
            title="Option Value vs Stock Price",
            xaxis_title="Stock Price",
            yaxis_title="Option Value",
            hovermode='x unified',
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Interactive Greeks Visualization")

        # Controls for surface plot
        col1, col2 = st.columns([1, 3])

        with col1:
            greek_choice = st.radio(
                "Select Greek",
                ["Price", "Delta", "Gamma", "Theta"]
            )

            st.write("Surface Parameters:")
            S_min = st.number_input("S min", value=50.0, step=10.0)
            S_max = st.number_input("S max", value=150.0, step=10.0)
            T_min = st.number_input("T min", value=0.1, step=0.1)
            T_max = st.number_input("T max", value=2.0, step=0.1)

            calculate_surface = st.button("Generate Surface")

        with col2:
            if calculate_surface:
                with st.spinner("Calculating surface..."):
                    S_range = np.linspace(S_min, S_max, 20)
                    T_range = np.linspace(T_min, T_max, 20)

                    S_mesh, T_mesh, price_surf, delta_surf, gamma_surf, theta_surf = create_greeks_surface(
                        S_range, T_range, K, r, sigma
                    )

                    # Select surface to display
                    surfaces = {
                        "Price": price_surf,
                        "Delta": delta_surf,
                        "Gamma": gamma_surf,
                        "Theta": theta_surf
                    }

                    Z = surfaces[greek_choice]

                    # Create 3D surface plot
                    fig = go.Figure(data=[go.Surface(
                        x=S_mesh,
                        y=T_mesh,
                        z=Z,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=greek_choice)
                    )])

                    fig.update_layout(
                        title=f"{greek_choice} Surface",
                        scene=dict(
                            xaxis_title="Stock Price",
                            yaxis_title="Time to Maturity",
                            zaxis_title=greek_choice,
                            camera=dict(
                                eye=dict(x=1.5, y=1.5, z=1.5)
                            )
                        ),
                        autosize=True,
                        height=600
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Heatmap view
                st.subheader("Heatmap View")

                fig_heat = px.imshow(
                    Z,
                    labels=dict(x="Stock Price", y="Time to Maturity", color=greek_choice),
                    x=S_range,
                    y=T_range,
                    aspect="auto",
                    color_continuous_scale="RdBu_r" if greek_choice == "Theta" else "Viridis"
                )
                fig_heat.update_layout(
                    title=f"{greek_choice} Heatmap",
                    height=400
                )

                st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.header("Historical Analysis")

        if st.session_state.yahoo_fetcher and use_live_data:
            try:
                # Fetch historical data
                hist_data = st.session_state.yahoo_fetcher.get_historical_prices(
                    ticker, period='6mo'
                )

                if hist_data is not None and not hist_data.empty:
                    # Calculate historical volatility
                    vol_calc = VolatilityCalculator()
                    hist_vol = vol_calc.historical_volatility(hist_data['Close'])

                    st.subheader(f"{ticker} Price History")

                    # Price chart
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue', width=2)
                    ))

                    fig_price.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode='x unified',
                        template='plotly_white'
                    )

                    st.plotly_chart(fig_price, use_container_width=True)

                    # Volatility analysis
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Historical Volatility (30d)", f"{hist_vol:.2%}")

                        # Rolling volatility
                        returns = hist_data['Close'].pct_change()
                        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)

                        fig_vol = go.Figure()
                        fig_vol.add_trace(go.Scatter(
                            x=rolling_vol.index,
                            y=rolling_vol,
                            mode='lines',
                            name='30d Rolling Vol',
                            line=dict(color='red', width=2)
                        ))

                        fig_vol.update_layout(
                            title="Rolling Volatility",
                            xaxis_title="Date",
                            yaxis_title="Volatility",
                            template='plotly_white',
                            height=300
                        )

                        st.plotly_chart(fig_vol, use_container_width=True)

                    with col2:
                        st.metric("Current Price", f"${hist_data['Close'].iloc[-1]:.2f}")

                        # Returns distribution
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Histogram(
                            x=returns.dropna(),
                            nbinsx=50,
                            name='Daily Returns',
                            histnorm='probability'
                        ))

                        fig_dist.update_layout(
                            title="Returns Distribution",
                            xaxis_title="Daily Return",
                            yaxis_title="Probability",
                            template='plotly_white',
                            height=300
                        )

                        st.plotly_chart(fig_dist, use_container_width=True)
                else:
                    st.warning("No historical data available for this ticker")
            except Exception as e:
                st.error(f"Error fetching historical data: {e}")
        else:
            st.info("Enable 'Use Live Data' in the sidebar to fetch historical data")

    with tab4:
        st.header("Portfolio Risk Analysis")

        st.subheader("Build Your Portfolio")

        # Portfolio builder
        num_positions = st.number_input("Number of Positions", 1, 10, 1)

        positions = []
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_value = 0

        cols = st.columns(num_positions if num_positions <= 3 else 3)

        for i in range(num_positions):
            with cols[i % 3]:
                st.write(f"**Position {i+1}**")

                pos_type = st.selectbox(f"Type", ["Call", "Put"], key=f"type_{i}")
                pos_quantity = st.number_input(f"Quantity", 1, 100, 1, key=f"qty_{i}")
                pos_strike = st.number_input(f"Strike", 50.0, 200.0, K, key=f"strike_{i}")

                # Calculate position Greeks
                result = calculate_option_pde(
                    S0, pos_strike, T, r, sigma,
                    pos_type.lower(), 'crank_nicolson'
                )

                position = {
                    'type': pos_type,
                    'quantity': pos_quantity,
                    'strike': pos_strike,
                    'price': result['price'],
                    'delta': result['delta'],
                    'gamma': result['gamma'],
                    'theta': result['theta']
                }
                positions.append(position)

                # Aggregate
                sign = 1 if pos_type == "Call" else -1
                total_value += pos_quantity * result['price']
                total_delta += pos_quantity * result['delta'] * sign
                total_gamma += pos_quantity * result['gamma']
                total_theta += pos_quantity * result['theta']

        # Display portfolio metrics
        st.subheader("Portfolio Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Value", f"${total_value:.2f}")
        with col2:
            st.metric("Portfolio Delta", f"{total_delta:.2f}")
        with col3:
            st.metric("Portfolio Gamma", f"{total_gamma:.4f}")
        with col4:
            st.metric("Portfolio Theta", f"{total_theta:.2f}")

        # P&L scenarios
        st.subheader("P&L Scenarios")

        spot_changes = np.linspace(-20, 20, 41)
        pnl_scenarios = []

        for change in spot_changes:
            new_S = S0 * (1 + change/100)
            pnl = 0

            for pos in positions:
                new_result = calculate_option_pde(
                    new_S, pos['strike'], T, r, sigma,
                    pos['type'].lower(), 'crank_nicolson'
                )
                pnl += pos['quantity'] * (new_result['price'] - pos['price'])

            pnl_scenarios.append(pnl)

        # P&L chart
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=spot_changes,
            y=pnl_scenarios,
            mode='lines',
            name='P&L',
            line=dict(width=3),
            fill='tozeroy'
        ))

        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")

        fig_pnl.update_layout(
            title="Portfolio P&L vs Stock Price Change",
            xaxis_title="Stock Price Change (%)",
            yaxis_title="P&L ($)",
            template='plotly_white',
            hovermode='x unified'
        )

        st.plotly_chart(fig_pnl, use_container_width=True)

    with tab5:
        st.header("Model Performance Analysis")

        if st.session_state.ml_model:
            st.subheader("PDE vs ML Model Comparison")

            # Generate test cases
            n_tests = st.slider("Number of Test Cases", 10, 100, 50, 10)

            with st.spinner(f"Running {n_tests} test cases..."):
                test_results = []

                np.random.seed(42)
                for _ in range(n_tests):
                    test_S = np.random.uniform(60, 140)
                    test_K = np.random.uniform(70, 130)
                    test_T = np.random.uniform(0.1, 2.0)
                    test_r = np.random.uniform(0.01, 0.10)
                    test_sigma = np.random.uniform(0.10, 0.50)

                    # Calculate with PDE
                    pde_result = calculate_option_pde(
                        test_S, test_K, test_T, test_r, test_sigma,
                        'call', 'crank_nicolson'
                    )

                    # Calculate with ML
                    ml_result = calculate_option_ml(
                        test_S, test_K, test_T, test_r, test_sigma,
                        st.session_state.ml_model
                    )

                    test_results.append({
                        'S': test_S,
                        'K': test_K,
                        'T': test_T,
                        'r': test_r,
                        'sigma': test_sigma,
                        'PDE_price': pde_result['price'],
                        'ML_price': ml_result['price'],
                        'PDE_time': pde_result['time'],
                        'ML_time': ml_result['time'],
                        'error': abs(ml_result['price'] - pde_result['price']),
                        'error_pct': abs(ml_result['price'] - pde_result['price']) / pde_result['price'] * 100
                    })

                df_results = pd.DataFrame(test_results)

            # Performance metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Mean Absolute Error",
                    f"${df_results['error'].mean():.4f}"
                )
                st.metric(
                    "Max Error",
                    f"${df_results['error'].max():.4f}"
                )

            with col2:
                st.metric(
                    "Mean Error %",
                    f"{df_results['error_pct'].mean():.2f}%"
                )
                st.metric(
                    "Max Error %",
                    f"{df_results['error_pct'].max():.2f}%"
                )

            with col3:
                avg_speedup = df_results['PDE_time'].mean() / df_results['ML_time'].mean()
                st.metric(
                    "Average Speedup",
                    f"{avg_speedup:.0f}x"
                )
                st.metric(
                    "ML Avg Time",
                    f"{df_results['ML_time'].mean()*1000:.3f} ms"
                )

            # Scatter plot
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=df_results['PDE_price'],
                y=df_results['ML_price'],
                mode='markers',
                marker=dict(
                    color=df_results['error_pct'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Error %"),
                    size=8
                ),
                text=[f"Error: {e:.2f}%" for e in df_results['error_pct']],
                hovertemplate="PDE: $%{x:.2f}<br>ML: $%{y:.2f}<br>%{text}<extra></extra>"
            ))

            # Add perfect prediction line
            min_price = min(df_results['PDE_price'].min(), df_results['ML_price'].min())
            max_price = max(df_results['PDE_price'].max(), df_results['ML_price'].max())
            fig_scatter.add_trace(go.Scatter(
                x=[min_price, max_price],
                y=[min_price, max_price],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction',
                showlegend=True
            ))

            fig_scatter.update_layout(
                title="ML vs PDE Predictions",
                xaxis_title="PDE Price ($)",
                yaxis_title="ML Price ($)",
                template='plotly_white',
                hovermode='closest'
            )

            st.plotly_chart(fig_scatter, use_container_width=True)

            # Error distribution
            col1, col2 = st.columns(2)

            with col1:
                fig_error_hist = go.Figure()
                fig_error_hist.add_trace(go.Histogram(
                    x=df_results['error_pct'],
                    nbinsx=30,
                    name='Error Distribution'
                ))

                fig_error_hist.update_layout(
                    title="Error Distribution",
                    xaxis_title="Error (%)",
                    yaxis_title="Count",
                    template='plotly_white'
                )

                st.plotly_chart(fig_error_hist, use_container_width=True)

            with col2:
                # Time comparison
                fig_time = go.Figure()
                fig_time.add_trace(go.Box(
                    y=df_results['PDE_time'] * 1000,
                    name='PDE',
                    marker_color='blue'
                ))
                fig_time.add_trace(go.Box(
                    y=df_results['ML_time'] * 1000,
                    name='ML',
                    marker_color='green'
                ))

                fig_time.update_layout(
                    title="Computation Time Comparison",
                    yaxis_title="Time (ms)",
                    template='plotly_white',
                    yaxis_type="log"
                )

                st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.warning("No ML model loaded. Please train models first by running main_demo.py")

if __name__ == "__main__":
    main()