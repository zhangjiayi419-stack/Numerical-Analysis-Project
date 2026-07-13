import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline

# Data points
x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
y = np.array([1/26, 1/17, 1/10, 1/5, 1/2, 1, 1/2, 1/5, 1/10, 1/17, 1/26])

# the point for drawing the interpolation curve
x_dense = np.linspace(-5, 5, 500)

# 1) Lagrange interpolation
poly_lag = lagrange(x, y)   
y_lag = poly_lag(x_dense)

# 2) Newton interpolation (implementing divided difference table)
def newton_interp(x, y, x_eval):
    n = len(x)
    # divided difference table
    coef = y.copy()
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x[i] - x[i-j])
    # Evaluate
    y_eval = np.zeros_like(x_eval)
    for i, xx in enumerate(x_eval):
        temp = coef[0]
        prod = 1.0
        for k in range(1, n):
            prod *= (xx - x[k-1])
            temp += coef[k] * prod
        y_eval[i] = temp
    return y_eval

y_newton = newton_interp(x, y, x_dense)

# 3) Piecewise linear interpolation
y_linear = np.interp(x_dense, x, y)

# 4) Cubic spline interpolation
d_left = 10/676
d_right = -10/676
cs = CubicSpline(x, y, bc_type=((1, d_left), (1, d_right)))
y_spline = cs(x_dense)

# Draw the graph
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'ko', markersize=8, label='Data points')
plt.plot(x_dense, y_lag, 'b-', linewidth=1.5, label='Lagrange')
plt.plot(x_dense, y_newton, 'g--', linewidth=1.5, label='Newton')
plt.plot(x_dense, y_linear, 'r:', linewidth=2, label='Piecewise linear')
plt.plot(x_dense, y_spline, 'm-', linewidth=1.5, label='Cubic spline')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Interpolation Methods')
plt.legend()
plt.grid(True)
plt.xlim(-5.2, 5.2)
plt.ylim(-0.2, 1.2)
plt.show()

# Evaluate
# Define the points to evaluate
x1, x2 = 1.2, 3.6

# Calculate the values of each method at these points
y1_lag = poly_lag(x1)          # Lagrange
y2_lag = poly_lag(x2)
y1_newton = newton_interp(x, y, np.array([x1]))[0]   # Newton
y2_newton = newton_interp(x, y, np.array([x2]))[0]
y1_linear = np.interp(x1, x, y)                     # 分段线性
y2_linear = np.interp(x2, x, y)
y1_spline = cs(x1)                                  # 三次样条
y2_spline = cs(x2)

# Exact value
exact1 = 25/61
exact2 = 25/349

# Print results
print(f"x = {x1}:")
print(f"  Exact value = {exact1:.10f}")
print(f"  Lagrange = {y1_lag:.10f}, Error = {abs(y1_lag-exact1):.2e}")
print(f"  Newton = {y1_newton:.10f}, Error = {abs(y1_newton-exact1):.2e}")
print(f"  Piecewise linear = {y1_linear:.10f}, Error = {abs(y1_linear-exact1):.2e}")
print(f"  Cubic spline = {y1_spline:.10f}, Error = {abs(y1_spline-exact1):.2e}")

print(f"\nx = {x2}:")
print(f"  Exact value = {exact2:.10f}")
print(f"  Lagrange = {y2_lag:.10f}, Error = {abs(y2_lag-exact2):.2e}")
print(f"  Newton = {y2_newton:.10f}, Error = {abs(y2_newton-exact2):.2e}")
print(f"  Piecewise linear = {y2_linear:.10f}, Error = {abs(y2_linear-exact2):.2e}")
print(f"  Cubic spline = {y2_spline:.10f}, Error = {abs(y2_spline-exact2):.2e}")