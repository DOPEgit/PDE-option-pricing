# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative finance project implementing PDE solvers for option pricing with ML surrogate models. The system achieves 100-1000x speedup for real-time risk management by training ML models on PDE solver outputs.

## Development Environment

### Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install as editable package
pip install -e .
```

### Common Commands

#### Running Tests
```bash
# Run all tests with coverage
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_solvers.py -v

# Run specific test class
pytest tests/test_numerical_methods.py::TestCrankNicolson -v

# Run single test
pytest tests/test_greeks.py::TestDelta::test_delta_bounds -v
```

#### Running Simulations
```bash
# Main demo (trains models and generates all plots)
python main_demo.py

# Quick system test (30 seconds)
python quick_test.py

# Individual simulations
python simulations/european_call/sim_crank_nicolson.py
python simulations/american_options/sim_american_put.py
python simulations/convergence_analysis/stability_analysis.py
```

#### Generating Training Data
```python
# Generate ML training data from PDE solver
from src.ml_models.data_generator import OptionDataGenerator
generator = OptionDataGenerator(method='crank_nicolson')
X, y = generator.generate_dataset(n_samples=5000, option_type='call')
```

## Architecture & Key Components

### Core PDE Solver Architecture

The system uses a **solver hierarchy** with abstract base class:

1. **`PDESolverBase`** (src/numerical_methods/solver_base.py)
   - Abstract interface: `solve()`, `check_stability()`
   - All solvers inherit from this

2. **Concrete Solvers** implement different finite difference schemes:
   - `ExplicitFD`: Fast but conditionally stable (α ≤ 0.5)
   - `ImplicitFD`: Unconditionally stable, requires matrix inversion
   - `CrankNicolson`: θ-method with θ=0.5, best accuracy-stability tradeoff

3. **`BlackScholesPDE`** (src/pde_solvers/black_scholes.py)
   - Core PDE implementation
   - Grid initialization: `S_grid`, `t_grid`, `V` matrix
   - Boundary conditions: `apply_boundary_conditions_call/put()`
   - Greeks calculation: `calculate_delta()`, `calculate_gamma()`, `calculate_theta()`

### ML Surrogate Model Pipeline

```
PDE Solver → Data Generator → Feature Engineering → ML Training → Fast Inference
```

1. **Data Generation** (src/ml_models/data_generator.py)
   - Uses PDE solver to generate ground truth
   - Parameter ranges: S₀∈[60,140], K∈[70,130], T∈[0.1,2.0], r∈[0.01,0.10], σ∈[0.10,0.50]
   - Feature engineering: adds `moneyness`, `log_moneyness`, `sqrt_T`, `vol_sqrt_T`

2. **Surrogate Models** (src/ml_models/surrogate_models.py)
   - `OptionPricingSurrogate`: Single output (price)
   - `MultiOutputSurrogate`: Multiple outputs (price + all Greeks)
   - Supports: Random Forest, XGBoost, Gradient Boosting

### Critical Implementation Details

#### Theta Calculation Bug (Fixed)
The theta calculation in `calculate_theta()` previously had an incorrect negative sign. The correct implementation:
```python
theta[1:] = (self.V[S_idx, 1:] - self.V[S_idx, :-1]) / self.dt
```
Theta is naturally negative for long options (time decay).

#### Stability Requirements
- **Explicit FD**: Requires `N_t ≥ 10 * N_S` for typical parameters
- **Implicit/CN**: Works with any `N_t` (unconditionally stable)
- Check stability: `solver.check_stability()` returns `(is_stable, alpha_max)`

#### Grid Initialization Pattern
All PDE instances follow:
```python
pde = BlackScholesPDE(S_max=300, T=1.0, r=0.05, sigma=0.2, N_S=100, N_t=1000)
payoff = pde.european_call_payoff(K=100)
boundary_func = lambda t_idx: pde.apply_boundary_conditions_call(K, t_idx)
solver = CrankNicolson(pde)
solver.solve(payoff, boundary_func, use_sparse=True)
```

## Model Training & Deployment

### Training ML Surrogates
```bash
# Full pipeline in main_demo.py
# Generates 2000 samples, trains Random Forest and XGBoost
# Models saved to data/models/{model_type}/
```

### Model Files
- `data/models/xgboost/`: price_model.joblib, delta_model.joblib, gamma_model.joblib, theta_model.joblib
- `data/models/random_forest/`: Same structure

### Performance Benchmarks
- PDE Solvers: 10-100ms per option
- ML Surrogates: 0.01ms per option (1000-1500x speedup)
- Accuracy: <1% error for standard parameter ranges

## Visualization Outputs

All plots generated in:
- `plots/`: Main comparison plots
- `simulations/*/plots/`: Simulation-specific plots

Key plots:
- `option_surfaces/`: 3D value surfaces
- `greeks/`: Delta, Gamma, Theta visualizations
- `ml_vs_pde/`: Performance comparisons
- `convergence/`: Numerical convergence analysis

## Testing Strategy

The test suite validates:
1. **Solver correctness**: Against analytical Black-Scholes formula
2. **Stability conditions**: Von Neumann stability for explicit method
3. **Greeks calculation**: Put-call parity, boundary behaviors
4. **Convergence rates**: O(h²) for Crank-Nicolson
5. **ML model accuracy**: <1% error on test set

## Common Issues & Solutions

1. **Import errors**: Ensure virtual environment is activated
2. **Theta test failures**: Check that theta calculation doesn't have negative sign
3. **Memory issues with large grids**: Use `use_sparse=True` for implicit methods
4. **Stability failures**: Increase N_t or switch to implicit/CN method
- when commiting and pushing always make a series of maximum commits. First think it throught. If you cannot amke maximimum commits then dont if it exceeds more commits feel free. First gather the commits that you need to do and then commit all the commits seperately in one go if you are able to do so