import numpy as np

def f(x, y):
    return -20 * y + 20 * np.sin(x) + np.cos(x)

def df_dx(x, y):
    # used in (e) question, i.e., y''
    y_prime = f(x, y)
    return -20 * y_prime + 20 * np.cos(x) - np.sin(x)

def exact_sol(x):
    return np.sin(x) + np.exp(-20 * x)

# Parameter settings
h = 0.1
x_end = 1.0
steps = int(x_end / h)
x_vals = np.linspace(0, x_end, steps + 1)

# (a) Modified Euler
y_me = np.zeros(steps + 1)
y_me[0] = 1.0
for n in range(steps):
    k1 = f(x_vals[n], y_me[n])
    k2 = f(x_vals[n+1], y_me[n] + h * k1)
    y_me[n+1] = y_me[n] + (h/2) * (k1 + k2)

# (b) Trapezoidal method
y_trap = np.zeros(steps + 1)
y_trap[0] = 1.0

for n in range(steps):
    # define G(x) = 20*sin(x) + cos(x)
    def G(x):
        return 20 * np.sin(x) + np.cos(x)
    
    # y_{n+1} = (y_n * (1 - 10h) + 0.5 * h * (G(x_n) + G(x_{n+1}))) / (1 + 10h)
    numerator = y_trap[n] * (1 - 10*h) + 0.5 * h * (G(x_vals[n]) + G(x_vals[n+1]))
    denominator = 1 + 10*h
    y_trap[n+1] = numerator / denominator

# (d) RK4
y_rk4 = np.zeros(steps + 1)
y_rk4[0] = 1.0
for n in range(steps):
    k1 = f(x_vals[n], y_rk4[n])
    k2 = f(x_vals[n] + h/2, y_rk4[n] + h/2 * k1)
    k3 = f(x_vals[n] + h/2, y_rk4[n] + h/2 * k2)
    k4 = f(x_vals[n+1], y_rk4[n] + h * k3)
    y_rk4[n+1] = y_rk4[n] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# (e) Special implicit method
y_special = np.zeros(steps + 1)
y_special[0] = 1.0
for n in range(steps):
    y_guess = y_special[n]
    f_n = f(x_vals[n], y_special[n])
    df_n = df_dx(x_vals[n], y_special[n])
    for _ in range(10):
        y_guess = y_special[n] + (h/6) * (4*f_n + 2*f(x_vals[n+1], y_guess) + h*df_n)
    y_special[n+1] = y_guess

# print results
print(f"{'x':<5} {'Exact':<10} {'Mod-Euler':<10} {'Trapezoidal':<10} {'RK4':<10} {'Special-Imp':<10}")
for i in range(len(x_vals)):
    print(f"{x_vals[i]:.1f}  {exact_sol(x_vals[i]):.6f}  {y_me[i]:.6f}   {y_trap[i]:.6f}    {y_rk4[i]:.6f}  {y_special[i]:.6f}")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Draw the exact solution
x_fine = np.linspace(0, x_end, 200)
plt.plot(x_fine, exact_sol(x_fine), 'k-', label='Exact Solution', linewidth=2, zorder=1)

# Draw the numerical solutions
plt.plot(x_vals, y_me, 'r--o', label='Modified Euler (Unstable)', alpha=0.7)
plt.plot(x_vals, y_trap, 'g-s', label='Trapezoidal', alpha=0.8)
plt.plot(x_vals, y_rk4, 'b-d', label='RK4', alpha=0.8)
plt.plot(x_vals, y_special, 'm-^', label='Special Implicit (e)', linewidth=2)

# Plot settings
plt.title('Numerical Solution Comparison (h=0.1)', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)

# Configure the x and y limits
plt.ylim(-0.2, 1.2) 

plt.tight_layout()
plt.show()