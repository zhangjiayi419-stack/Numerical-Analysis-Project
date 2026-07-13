import numpy as np

def f(x):
    return np.exp(-x) - np.sin(x)

def df(x):
    return -np.exp(-x) - np.cos(x)

# (i) Bisection
def bisection(a, b, tol):
    iters = 0
    while (b - a) / 2 > tol:
        iters += 1
        mid = (a + b) / 2
        if f(mid) == 0: return mid, iters
        elif f(a) * f(mid) < 0: b = mid
        else: a = mid
    return (a + b) / 2, iters

# (ii) Newton's Method
def newton(x0, tol):
    iters = 0
    x = x0
    while True:
        iters += 1
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < tol:
            return x_new, iters
        x = x_new

# (iii) Secant Method
def secant(x0, x1, tol):
    iters = 0
    while True:
        iters += 1
        f_x0, f_x1 = f(x0), f(x1)
        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        if abs(x_new - x1) < tol:
            return x_new, iters
        x0, x1 = x1, x_new

# Caculation
tol = 1e-6
print(f"(a)(i)   Bisection: root={bisection(0, 1, tol)[0]:.6f}, iters={bisection(0, 1, tol)[1]}")
print(f"(a)(ii)  Newton:    root={newton(0.5, tol)[0]:.6f}, iters={newton(0.5, tol)[1]}")
print(f"(a)(iii) Secant:    root={secant(0, 1, tol)[0]:.6f}, iters={secant(0, 1, tol)[1]}")

# (c) Testing different starting points
for start in [0, 0.2, 1]:
    r, it = newton(start, tol)
    print(f"(c) Newton start {start}: root={r:.6f}, iters={it}")

# print the results
if __name__ == "__main__":
    tol = 1e-6
    
    print("--- (a) Results ---")
    root_bi, it_bi = bisection(0, 1, tol)
    print(f"Bisection: Root = {root_bi:.6f}, Iters = {it_bi}")
    
    root_new, it_new = newton(0.5, tol)
    print(f"Newton (x0=0.5): Root = {root_new:.6f}, Iters = {it_new}")
    
    root_sec, it_sec = secant(0, 1, tol)
    print(f"Secant (0, 1): Root = {root_sec:.6f}, Iters = {it_sec}")
    
    print("\n--- (c) Newton Convergence Test ---")
    for start in [0, 0.2, 1]:
        r, it = newton(start, tol)
        print(f"Start x0 = {start}: Root = {r:.6f}, Iters = {it}")