import numpy as np
import matplotlib.pyplot as plt


# Пример функции, её первая и вторая производные
def f(x):
    return (x**2)/2 - np.cos(x)  # Квадратичная функция

def df(x):
    return x + np.sin(x) # Первая производная

def ddf(x):
    return 1 + np.cos(x)  # Вторая производная

# Метод Ньютона с выводом количества итераций
def newton_method_with_iterations(x0, tol=1e-5, max_iter=400):
    x = x0
    iterations = 0
    for _ in range(max_iter):
        x_new = x - df(x) / ddf(x)
        iterations += 1
        if abs(x_new - x) < tol:
            break
        x = x_new
    return x, iterations

# Начальная точка
x0 = -1
minimum, num_iterations = newton_method_with_iterations(x0)

print(f"Найденный минимум функции: x = {minimum}, f(x) = {f(minimum)} Количество итераций = {num_iterations}")
print(f" Количество итераций = {num_iterations}" )

# Построение графика
x_values = np.linspace(-3, 1, 400)
y_values = f(x_values)

plt.plot(x_values, y_values, label='f(x) = (x^2)/2 - cos(x)')
plt.scatter(minimum, f(minimum), color='red', label='Минимум', zorder=5)
plt.title('График функции и её минимум')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid()
plt.show()

