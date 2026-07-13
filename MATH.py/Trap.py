import numpy as np
from scipy.integrate import quad
from scipy.special import roots_jacobi

# ---------------- Integrands (Fully Fixed) ----------------
def f1(x): return x**2
def f2(x): return x**(3/2)
def f3(x): return np.sqrt(x) * np.cos(x)
def f4(x):
    return np.where(x < 1e-10, 0.0, np.sqrt(x) * np.log(x))

# ----------------- Exact Values ----------------
exact = {
    'x^2': 1/3,
    'x^(3/2)': 2/5,
    'sqrt(x)cos(x)': quad(lambda x: np.sqrt(x)*np.cos(x), 0, 1)[0],
    'sqrt(x)ln(x)': -4/9
}
# ---------------- Quadrature Rules (Strictly Theoretically Correct) ----------------
# 1. Trapezoidal rule I1
def trapezoidal(f):
    return 0.5 * (f(0) + f(1))

# 2. I2 from part (a): 0.25*f(0) + 0.75*f(2/3)
def I2(f):
    return 0.25 * f(0) + 0.75 * f(2/3)

# 3. 2-point Gauss-Legendre I3 from part (b)
def I3(f):
    x0 = (1 - 1/np.sqrt(3)) / 2
    x1 = (1 + 1/np.sqrt(3)) / 2
    return 0.5 * f(x0) + 0.5 * f(x1)

# 4. Weighted Gauss I4 with weight sqrt(x) (part d)
def I4(f):
    t, w = roots_jacobi(2, 0, 0.5)
    x = (t + 1) / 2
    A = w * (2 ** (-1.5))
    return np.sum(A * f(x))

# ---------------- Output Tables ----------------
functions = [f1, f2, f3, f4]
names = [
    r'$\int_0^1 x^2 dx$',
    r'$\int_0^1 x^{3/2} dx$',
    r'$\int_0^1 \sqrt{x} \cos x dx$',
    r'$\int_0^1 \sqrt{x} \ln x dx$'
]
exact_vals = [exact['x^2'], exact['x^(3/2)'], exact['sqrt(x)cos(x)'], exact['sqrt(x)ln(x)']]

# (c) Comparison Table
print("="*80)
print("(c) Comparison of Three Quadrature Formulas (Exact vs Approximation)")
print("-"*80)
print(f"{'Integral':<25} {'Exact':<12} {'I1(Trapz)':<12} {'I2':<12} {'I3(Gauss)':<12}")
print("-"*80)
for name, f, exact in zip(names, functions, exact_vals):
    i1 = trapezoidal(f)
    i2 = I2(f)
    i3 = I3(f)
    print(f"{name:<25} {exact:<12.6f} {i1:<12.6f} {i2:<12.6f} {i3:<12.6f}")

# (c) Absolute Error Table
print("\n(c) Absolute Errors of Three Formulas:")
print("-"*80)
print(f"{'Integral':<25} {'I1 Error':<12} {'I2 Error':<12} {'I3 Error':<12}")
print("-"*80)
for name, f, exact in zip(names, functions, exact_vals):
    err1 = abs(trapezoidal(f) - exact)
    err2 = abs(I2(f) - exact)
    err3 = abs(I3(f) - exact)
    print(f"{name:<25} {err1:<12.2e} {err2:<12.2e} {err3:<12.2e}")

# (d)(i) I4 Parameters
print("\n" + "="*60)
print("(d)(i) Parameters of I4(f) (2-point Gauss with weight sqrt(x)):")
t4, w4 = roots_jacobi(2, 0, 0.5)
x0_I4 = (t4[0] + 1)/2
x1_I4 = (t4[1] + 1)/2
A0_I4 = w4[0] * (2 ** (-1.5))
A1_I4 = w4[1] * (2 ** (-1.5))
print(f"A0 = {A0_I4:.10f}, A1 = {A1_I4:.10f}")
print(f"x0 = {x0_I4:.10f}, x1 = {x1_I4:.10f} (x0 ≤ x1)")
print("Degree of precision: 3 (exact for all polynomials of degree ≤3)")

# ---------------- For I4 (deduct sqrt(x) weight) ----------------
def g1(x): return x**(1.5)  #  x^1.5 * sqrt(x) = x^2
def g2(x): return x         #  x * sqrt(x) = x^(3/2)
def g3(x): return np.cos(x) #  cos(x) * sqrt(x) = sqrt(x)cos(x)
def g4(x): return np.log(x) #  ln(x) * sqrt(x) = sqrt(x)ln(x)

functions_I4 = [g1, g2, g3, g4]

# (d)(ii) I4 Comparison
print("\n" + "="*80)
print("(d)(ii) I4(f) Results (Exact vs Approximation)")
print("-"*80)
print(f"{'Integral':<25} {'Exact':<12} {'I4(Weighted Gauss)':<18} {'I4 Abs Error':<12}")
print("-"*80)
# We use functions_I4 ( g1, g2, g3, g4)，but still compare with Exact values
for name, g, exact in zip(names, functions_I4, exact_vals):
    i4 = I4(g)  
    err4 = abs(i4 - exact)
    print(f"{name:<25} {exact:<12.6f} {i4:<18.6f} {err4:<12.2e}")