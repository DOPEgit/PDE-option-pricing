# Mathematical Methodology

## Overview

This document provides the mathematical foundation for the PDE-based option pricing methods implemented in this project. We cover the Black-Scholes PDE derivation, finite difference discretizations, and convergence theory.

---

## 1. The Black-Scholes Partial Differential Equation

### 1.1 Derivation

Consider a European option with value `V(S,t)` where:
- `S` = underlying stock price
- `t` = time
- `T` = maturity time

Under the Black-Scholes assumptions (no dividends, constant volatility σ, constant risk-free rate r), the option value satisfies:

```
∂V/∂t + rS(∂V/∂S) + (σ²/2)S²(∂²V/∂S²) - rV = 0
```

with terminal condition:
```
V(S,T) = Payoff(S)
```

**Key Assumptions:**
1. No arbitrage opportunities
2. Frictionless markets (no transaction costs)
3. Continuous trading possible
4. Stock follows geometric Brownian motion: `dS = μS dt + σS dW`
5. Constant risk-free rate `r` and volatility `σ`

### 1.2 Boundary Conditions

#### European Call Option (Strike K):
- **S → 0**: `V(0,t) = 0` (worthless)
- **S → ∞**: `V(S,t) ~ S - Ke^(-r(T-t))` (intrinsic value)
- **t = T**: `V(S,T) = max(S - K, 0)`

#### European Put Option (Strike K):
- **S → 0**: `V(0,t) = Ke^(-r(T-t))` (worth present value of strike)
- **S → ∞**: `V(S,t) → 0` (worthless)
- **t = T**: `V(S,T) = max(K - S, 0)`

---

## 2. Finite Difference Methods

We discretize the PDE on a grid:
- **Space**: `S_i = i·ΔS`, `i = 0,1,...,N_S`
- **Time**: `t_n = n·Δt`, `n = 0,1,...,N_t`
- **Grid values**: `V_i^n ≈ V(S_i, t_n)`

### 2.1 Explicit Finite Difference (FTCS)

**Forward Time, Centered Space** discretization:

```
(V_i^{n+1} - V_i^n)/Δt + rS_i(V_{i+1}^n - V_{i-1}^n)/(2ΔS)
    + (σ²/2)S_i²(V_{i+1}^n - 2V_i^n + V_{i-1}^n)/ΔS² - rV_i^n = 0
```

**Rearranging:**
```
V_i^{n+1} = V_i^n + α(V_{i+1}^n - 2V_i^n + V_{i-1}^n) + β(V_{i+1}^n - V_{i-1}^n)
```

where:
```
α = (σ²S_i²Δt)/(2ΔS²)
β = (rS_iΔt)/(2ΔS)
```

**Stability Condition** (von Neumann analysis):
```
α ≤ 0.5  ⟹  Δt ≤ ΔS²/(σ²S_max²)
```

**Properties:**
- ✅ Simple to implement
- ✅ Fast computation (explicit update)
- ❌ **Conditionally stable** - requires small Δt
- ❌ First-order accurate in time: O(Δt)
- ✅ Second-order accurate in space: O(ΔS²)

### 2.2 Implicit Finite Difference (BTCS)

**Backward Time, Centered Space** discretization:

```
(V_i^{n+1} - V_i^n)/Δt + rS_i(V_{i+1}^{n+1} - V_{i-1}^{n+1})/(2ΔS)
    + (σ²/2)S_i²(V_{i+1}^{n+1} - 2V_i^{n+1} + V_{i-1}^{n+1})/ΔS² - rV_i^{n+1} = 0
```

**Matrix form:** `AV^{n+1} = V^n`

where A is a tridiagonal matrix:
```
A_i,i-1 = -α_i + β_i
A_i,i   = 1 + 2α_i + rΔt
A_i,i+1 = -α_i - β_i
```

**Solution:** Use Thomas algorithm (tridiagonal solver) or sparse LU decomposition.

**Properties:**
- ✅ **Unconditionally stable** - any Δt works
- ✅ More accurate for large Δt
- ❌ Requires solving linear system (higher cost per step)
- ❌ First-order accurate in time: O(Δt)
- ✅ Second-order accurate in space: O(ΔS²)

### 2.3 Crank-Nicolson Method

**Average of explicit and implicit** (θ-method with θ=0.5):

```
(V_i^{n+1} - V_i^n)/Δt = (1/2)[L(V^n) + L(V^{n+1})]
```

where L is the spatial discretization operator.

**Matrix form:** `(I - θΔtL)V^{n+1} = (I + (1-θ)ΔtL)V^n`

With θ=0.5:
```
(I - 0.5ΔtL)V^{n+1} = (I + 0.5ΔtL)V^n
```

**Properties:**
- ✅ **Unconditionally stable** for θ ≥ 0.5
- ✅ **Second-order accurate in time**: O(Δt²) when θ=0.5
- ✅ **Second-order accurate in space**: O(ΔS²)
- ✅ **Best accuracy-stability tradeoff**
- ❌ Requires solving linear system

---

## 3. Convergence Theory

### 3.1 Consistency

A finite difference scheme is **consistent** if the local truncation error tends to zero as Δt, ΔS → 0.

**Explicit FD:** O(Δt) + O(ΔS²)
**Implicit FD:** O(Δt) + O(ΔS²)
**Crank-Nicolson:** O(Δt²) + O(ΔS²)

### 3.2 Stability

A scheme is **stable** if errors do not grow unboundedly as we step forward in time.

**Von Neumann Stability Analysis:**
Substitute V_i^n = e^(ikΔS)λ^n and analyze growth factor λ.

- **Explicit:** Stable if `|λ| ≤ 1` ⟹ `α ≤ 0.5`
- **Implicit:** Always `|λ| ≤ 1` (unconditionally stable)
- **Crank-Nicolson:** Always `|λ| ≤ 1` for θ ≥ 0.5

### 3.3 Convergence

**Lax Equivalence Theorem:** For a consistent finite difference scheme applied to a well-posed linear PDE:

```
Stability + Consistency ⟹ Convergence
```

**Convergence Rates:**
- Explicit FD: O(Δt + ΔS²)
- Implicit FD: O(Δt + ΔS²)
- Crank-Nicolson: O(Δt² + ΔS²)

---

## 4. Option Greeks Calculation

Greeks measure sensitivity of option value to various parameters.

### 4.1 Delta (Δ)

**Definition:** `Δ = ∂V/∂S` (sensitivity to stock price)

**Finite Difference Approximation:**
```
Δ_i ≈ (V_{i+1} - V_{i-1})/(2ΔS)  (centered)
```

**Properties:**
- Call: 0 ≤ Δ ≤ 1
- Put: -1 ≤ Δ ≤ 0

### 4.2 Gamma (Γ)

**Definition:** `Γ = ∂²V/∂S²` (curvature, sensitivity of Delta)

**Finite Difference Approximation:**
```
Γ_i ≈ (V_{i+1} - 2V_i + V_{i-1})/ΔS²
```

**Properties:**
- Always non-negative for long positions
- Peaks at-the-money
- Approaches 0 deep ITM/OTM

### 4.3 Theta (Θ)

**Definition:** `Θ = ∂V/∂t` (time decay)

**Finite Difference Approximation:**
```
Θ ≈ -(V^{n+1} - V^n)/Δt  (backward)
```

**Properties:**
- Generally negative for long options (time decay)
- Increases (more negative) near maturity

---

## 5. Analytical Black-Scholes Formula

For validation, we use the closed-form Black-Scholes formulas:

### European Call:
```
C(S,t) = S·N(d₁) - K·e^(-r(T-t))·N(d₂)
```

### European Put:
```
P(S,t) = K·e^(-r(T-t))·N(-d₂) - S·N(-d₁)
```

where:
```
d₁ = [ln(S/K) + (r + σ²/2)(T-t)] / (σ√(T-t))
d₂ = d₁ - σ√(T-t)
N(x) = cumulative standard normal distribution
```

### Put-Call Parity:
```
C - P = S - K·e^(-r(T-t))
```

---

## 6. Numerical Stability Considerations

### 6.1 Grid Size Selection

**For Explicit Method:**
```
Δt ≤ ΔS²/(σ²S_max²)
```

**Practical Choice:**
```
N_t ≥ 10·N_S  (for explicit stability)
N_t = N_S  (sufficient for implicit/CN)
```

### 6.2 Boundary Placement

**S_max Selection:**
```
S_max ≥ 3·max(S₀, K)
```

Ensures boundary effects don't contaminate solution in region of interest.

### 6.3 Numerical Artifacts

**Oscillations:**
- Can occur in explicit method near stability limit
- Solution: Use Crank-Nicolson with θ=0.5

**Boundary Effects:**
- Incorrect boundaries can propagate into interior
- Solution: Place boundaries sufficiently far

---

## 7. American Options

For American options (early exercise allowed), modify the PDE constraint:

```
V(S,t) ≥ Payoff(S)  (for all t)
```

**Implementation:**
At each time step, apply:
```
V_i^n = max(V_i^n, Payoff(S_i))
```

This creates a **free boundary problem** where the optimal exercise boundary must be determined.

---

## 8. References

1. **Black, F., & Scholes, M. (1973).** "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637-654.

2. **Wilmott, P., Howison, S., & Dewynne, J. (1995).** *The Mathematics of Financial Derivatives.* Cambridge University Press.

3. **Duffy, D. J. (2006).** *Finite Difference Methods in Financial Engineering: A Partial Differential Equation Approach.* John Wiley & Sons.

4. **Tavella, D., & Randall, C. (2000).** *Pricing Financial Instruments: The Finite Difference Method.* John Wiley & Sons.

5. **Strikwerda, J. C. (2004).** *Finite Difference Schemes and Partial Differential Equations* (2nd ed.). SIAM.

6. **Hull, J. C. (2017).** *Options, Futures, and Other Derivatives* (10th ed.). Pearson.

---

## Appendix A: Stability Proof (von Neumann Analysis)

For the explicit scheme, substitute:
```
V_i^n = e^(ikΔS)λ^n
```

The amplification factor λ satisfies:
```
λ = 1 - 2α(1-cos(kΔS)) - 2iβ·sin(kΔS) - rΔt
```

For stability, require |λ| ≤ 1 for all k, which gives:
```
α ≤ 1/2
```

---

## Appendix B: Convergence Rate Verification

To verify O(h^p) convergence:

1. Run solver with grid sizes N₁, N₂, N₃ (typically doubling)
2. Calculate errors E₁, E₂, E₃ vs analytical solution
3. Plot log(E) vs log(N) - slope should be -p
4. Calculate empirical rate: p ≈ log(E₁/E₂)/log(N₂/N₁)

**Expected Rates:**
- Explicit/Implicit: p ≈ 1 (first-order in time)
- Crank-Nicolson: p ≈ 2 (second-order)

---

**Last Updated:** 2025-09-29
**Author:** Sakeeb Rahman