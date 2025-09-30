# API Reference

Complete documentation of all classes and functions.

---

## Core Modules

### `src.pde_solvers.black_scholes`

#### `BlackScholesPDE`

Main PDE solver class for Black-Scholes equation.

```python
class BlackScholesPDE:
    def __init__(self, S_max=300.0, T=1.0, r=0.05, sigma=0.2, N_S=100, N_t=1000)
```

**Parameters:**
- `S_max` (float): Maximum stock price in grid
- `T` (float): Time to maturity (years)
- `r` (float): Risk-free interest rate
- `sigma` (float): Volatility
- `N_S` (int): Number of spatial grid points
- `N_t` (int): Number of time steps

**Attributes:**
- `S_grid` (ndarray): Stock price grid points
- `t_grid` (ndarray): Time grid points
- `V` (ndarray): Solution grid (N_S+1 × N_t+1)
- `dS`, `dt` (float): Grid spacings

**Methods:**

##### `european_call_payoff(K)`
```python
def european_call_payoff(self, K: float) -> np.ndarray
```
Returns payoff array for European call option at maturity.

**Returns:** Array of shape (N_S+1,) with max(S-K, 0)

##### `european_put_payoff(K)`
```python
def european_put_payoff(self, K: float) -> np.ndarray
```
Returns payoff for European put option.

**Returns:** Array with max(K-S, 0)

##### `barrier_call_payoff(K, H, barrier_type)`
```python
def barrier_call_payoff(self, K: float, H: float,
                       barrier_type: str = "up_and_out") -> np.ndarray
```

**Parameters:**
- `barrier_type`: "up_and_out", "down_and_out", "up_and_in", "down_and_in"

##### `apply_boundary_conditions_call(K, t_idx)`
```python
def apply_boundary_conditions_call(self, K: float, t_idx: int) -> None
```
Apply Dirichlet boundary conditions for call option at time index.

##### `get_analytical_bs_call(S0, K, t)`
```python
def get_analytical_bs_call(self, S0: float, K: float, t: float) -> float
```
Calculate analytical Black-Scholes call price.

**Returns:** Call option value

##### `calculate_delta(t_idx)`
```python
def calculate_delta(self, t_idx: int) -> np.ndarray
```
Calculate Delta (∂V/∂S) using central differences.

**Returns:** Delta values at each grid point

##### `calculate_gamma(t_idx)`
```python
def calculate_gamma(self, t_idx: int) -> np.ndarray
```
Calculate Gamma (∂²V/∂S²).

##### `calculate_theta(S_idx)`
```python
def calculate_theta(self, S_idx: int) -> np.ndarray
```
Calculate Theta (∂V/∂t) at fixed stock price index.

---

### `src.numerical_methods`

#### `PDESolverBase`

Abstract base class for all numerical solvers.

```python
class PDESolverBase(ABC):
    def __init__(self, pde_instance)

    @abstractmethod
    def solve(self, payoff, boundary_func, **kwargs)

    def check_stability(self) -> Tuple[bool, float]
```

#### `ExplicitFD`

Explicit finite difference solver (FTCS).

```python
class ExplicitFD(PDESolverBase):
    def solve(self, payoff, boundary_func, barrier_func=None, **kwargs) -> np.ndarray

    def solve_vectorized(self, payoff, boundary_func, **kwargs) -> np.ndarray

    def check_stability(self) -> Tuple[bool, float]
```

**Stability:** Returns `(is_stable, alpha_max)` where `alpha_max ≤ 0.5` required.

**Example:**
```python
from src.pde_solvers.black_scholes import BlackScholesPDE
from src.numerical_methods.explicit_fd import ExplicitFD

pde = BlackScholesPDE(N_S=100, N_t=3000)
solver = ExplicitFD(pde)
is_stable, alpha = solver.check_stability()
if is_stable:
    solver.solve_vectorized(payoff, boundary_func)
```

#### `ImplicitFD`

Implicit finite difference solver (BTCS).

```python
class ImplicitFD(PDESolverBase):
    def solve(self, payoff, boundary_func, use_sparse=True, **kwargs) -> np.ndarray

    def build_coefficient_matrix(self) -> np.ndarray

    def build_sparse_matrix(self)  # Returns scipy.sparse matrix

    def thomas_algorithm(self, a, b, c, d) -> np.ndarray
```

**Features:**
- Unconditionally stable
- Solves tridiagonal system at each step
- `use_sparse=True` for large grids

**Example:**
```python
solver = ImplicitFD(pde)
solver.solve(payoff, boundary_func, use_sparse=True)
```

#### `CrankNicolson`

Crank-Nicolson solver (θ-method with θ=0.5).

```python
class CrankNicolson(PDESolverBase):
    def __init__(self, pde_instance, theta=0.5)

    def solve(self, payoff, boundary_func, use_sparse=True, **kwargs) -> np.ndarray

    def build_matrices(self) -> Tuple[np.ndarray, np.ndarray]

    def get_convergence_order(self) -> Tuple[int, int]
```

**Returns:** `(time_order, space_order)` = (2, 2) for θ=0.5

**Example:**
```python
solver = CrankNicolson(pde, theta=0.5)
solver.solve(payoff, boundary_func, use_sparse=True)
```

---

### `src.ml_models`

#### `OptionDataGenerator`

Generate training data from PDE solver.

```python
class OptionDataGenerator:
    def __init__(self, method='crank_nicolson')

    def generate_dataset(self, n_samples=10000, option_type='call',
                        parameter_ranges=None, grid_size=(100, 1000),
                        random_seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]

    def save_dataset(self, X, y, filepath_X, filepath_y)

    def load_dataset(self, filepath_X, filepath_y) -> Tuple[pd.DataFrame, pd.DataFrame]
```

**Example:**
```python
from src.ml_models.data_generator import OptionDataGenerator

generator = OptionDataGenerator()
X_train, y_train = generator.generate_dataset(
    n_samples=5000,
    option_type='call',
    parameter_ranges={
        'S0': (60, 140),
        'K': (70, 130),
        'T': (0.1, 2.0),
        'r': (0.01, 0.10),
        'sigma': (0.10, 0.50)
    }
)

generator.save_dataset(X_train, y_train, 'X.csv', 'y.csv')
```

**Generated Features:**
- Base: `S0`, `K`, `T`, `r`, `sigma`
- Derived: `moneyness`, `log_moneyness`, `sqrt_T`, `vol_sqrt_T`

#### `OptionPricingSurrogate`

ML surrogate model for fast option pricing.

```python
class OptionPricingSurrogate:
    def __init__(self, model_type='random_forest')

    def build_model(self, **kwargs)

    def train(self, X, y, target_col='price', test_size=0.2) -> Dict

    def predict(self, X, scale_features=True) -> np.ndarray

    def predict_with_timing(self, X) -> Tuple[np.ndarray, float]

    def get_feature_importance(self, top_n=10) -> pd.DataFrame

    def save_model(self, filepath)

    def load_model(self, filepath)
```

**Supported Models:**
- `'random_forest'`: RandomForestRegressor
- `'gradient_boosting'`: GradientBoostingRegressor
- `'xgboost'`: XGBRegressor

**Example:**
```python
from src.ml_models.surrogate_models import OptionPricingSurrogate

surrogate = OptionPricingSurrogate(model_type='xgboost')
surrogate.build_model(n_estimators=200, max_depth=8)
metrics = surrogate.train(X_train, y_train, target_col='price')

# Fast prediction
predictions = surrogate.predict(X_test)
```

#### `MultiOutputSurrogate`

Predict price and all Greeks simultaneously.

```python
class MultiOutputSurrogate:
    def __init__(self, model_type='random_forest')

    def train_all(self, X, y, test_size=0.2) -> Dict

    def predict_all(self, X) -> Dict[str, np.ndarray]

    def save_all(self, directory)

    def load_all(self, directory)
```

**Example:**
```python
from src.ml_models.surrogate_models import MultiOutputSurrogate

surrogate = MultiOutputSurrogate('xgboost')
all_metrics = surrogate.train_all(X_train, y_train)

# Predict all
predictions = surrogate.predict_all(X_test)
# Returns: {'price': [...], 'delta': [...], 'gamma': [...], 'theta': [...]}
```

---

### `src.visualization`

#### `plot_option_surface(S_grid, t_grid, V, title, save_path, view_angles)`

Create 3D surface plot of option values.

```python
def plot_option_surface(
    S_grid: np.ndarray,
    t_grid: np.ndarray,
    V: np.ndarray,
    title: str = "Option Value Surface",
    save_path: str = None,
    view_angles: tuple = (30, 45)
) -> Figure
```

**Returns:** matplotlib Figure object

#### `plot_greeks_surface(S_grid, t_grid, delta, gamma, theta, title_prefix, save_path)`

Plot Delta, Gamma, Theta surfaces.

```python
def plot_greeks_surface(
    S_grid: np.ndarray,
    t_grid: np.ndarray,
    delta: np.ndarray,
    gamma: np.ndarray,
    theta: np.ndarray = None,
    title_prefix: str = "Option",
    save_path: str = None
) -> Figure
```

#### `plot_comparison(S_grid, methods_data, K, title, save_path)`

Compare multiple methods side-by-side.

```python
def plot_comparison(
    S_grid: np.ndarray,
    methods_data: dict,
    K: float,
    title: str = "Method Comparison",
    save_path: str = None
) -> Figure
```

**Example:**
```python
from src.visualization.surface_plots import plot_comparison

methods_data = {
    'Crank-Nicolson': V_cn,
    'Implicit FD': V_imp,
    'Explicit FD': V_exp
}

plot_comparison(S_grid, methods_data, K=100,
               save_path='comparison.png')
```

#### `plot_convergence_analysis(grid_sizes, errors, title, save_path)`

Generate convergence rate plots.

```python
def plot_convergence_analysis(
    grid_sizes: list,
    errors: dict,
    title: str = "Convergence Analysis",
    save_path: str = None
) -> Figure
```

#### `plot_ml_vs_pde_comparison(parameter_ranges, ml_predictions, pde_solutions, ml_times, pde_times, save_path)`

Comprehensive ML vs PDE comparison dashboard.

```python
def plot_ml_vs_pde_comparison(
    parameter_ranges: dict,
    ml_predictions: np.ndarray,
    pde_solutions: np.ndarray,
    ml_times: np.ndarray,
    pde_times: np.ndarray,
    save_path: str = None
) -> Figure
```

---

## Complete Workflow Examples

### Example 1: Price European Call with All Methods

```python
from src.pde_solvers.black_scholes import BlackScholesPDE
from src.numerical_methods import ExplicitFD, ImplicitFD, CrankNicolson

# Setup
pde = BlackScholesPDE(S_max=300, T=1.0, r=0.05, sigma=0.2, N_S=100, N_t=1000)
K = 100
payoff = pde.european_call_payoff(K)
boundary = lambda t: pde.apply_boundary_conditions_call(K, t)

# Solve with all methods
solvers = {
    'Explicit': ExplicitFD(pde),
    'Implicit': ImplicitFD(pde),
    'Crank-Nicolson': CrankNicolson(pde)
}

results = {}
for name, solver in solvers.items():
    solver.solve(payoff.copy(), boundary, use_sparse=True)
    S_idx = np.argmin(np.abs(pde.S_grid - 100))
    results[name] = pde.V[S_idx, 0]

print(results)
# {'Explicit': 10.44, 'Implicit': 10.45, 'Crank-Nicolson': 10.45}
```

### Example 2: Train and Deploy ML Surrogate

```python
from src.ml_models.data_generator import OptionDataGenerator
from src.ml_models.surrogate_models import MultiOutputSurrogate

# Generate training data
generator = OptionDataGenerator()
X, y = generator.generate_dataset(n_samples=5000)

# Train model
surrogate = MultiOutputSurrogate('xgboost')
metrics = surrogate.train_all(X, y)

# Save for production
surrogate.save_all('models/production/')

# Deploy: predict new options
X_new = pd.DataFrame({
    'S0': [95, 100, 105],
    'K': [100, 100, 100],
    'T': [1.0, 1.0, 1.0],
    'r': [0.05, 0.05, 0.05],
    'sigma': [0.2, 0.2, 0.2],
    # Add derived features
    'moneyness': [0.95, 1.0, 1.05],
    'log_moneyness': np.log([0.95, 1.0, 1.05]),
    'sqrt_T': [1.0, 1.0, 1.0],
    'vol_sqrt_T': [0.2, 0.2, 0.2]
})

predictions = surrogate.predict_all(X_new)
print(f"Prices: {predictions['price']}")
print(f"Deltas: {predictions['delta']}")
```

### Example 3: Run Full Convergence Study

```python
from src.pde_solvers.black_scholes import BlackScholesPDE
from src.numerical_methods.crank_nicolson import CrankNicolson
from src.visualization.convergence_plots import plot_convergence_analysis

grid_sizes = [25, 50, 75, 100, 150]
errors = {'Crank-Nicolson': []}

for N in grid_sizes:
    pde = BlackScholesPDE(N_S=N, N_t=N*10)
    solver = CrankNicolson(pde)
    solver.solve(payoff, boundary)

    numerical = pde.V[N//2, 0]
    analytical = pde.get_analytical_bs_call(100, 100, 0)
    errors['Crank-Nicolson'].append(abs(numerical - analytical))

plot_convergence_analysis(grid_sizes, errors, save_path='convergence.png')
```

---

## Testing

All classes include comprehensive unit tests in `tests/` directory.

**Run tests:**
```bash
pytest tests/                    # All tests
pytest tests/test_solvers.py     # Solver tests only
pytest tests/ -v --cov=src       # With coverage
```

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .  # Install as editable package
```

---

**Last Updated:** 2025-09-29
**Author:** Sakeeb Rahman