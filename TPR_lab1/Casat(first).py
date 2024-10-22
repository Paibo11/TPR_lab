import numpy as np
import matplotlib.pyplot as plt


# Функция и её производная
def f(x):
    return (x**2)/2 - np.cos(x)  # Пример: квадратичная функция


def df(x):
    return x + np.sin(x)  # Производная функции


# Построение касательной в точке x_0
def tangent_line(f, df, x0):
    return lambda x: f(x0) + df(x0) * (x - x0)


# Метод касательных
def casat(f, df, a, b, tol=1e-5, max_iter=100):
    x0 = np.random.uniform(a, b)  # Случайный выбор начальной точки
    history = [x0]

    for i in range(max_iter):
        # Построение касательной
        p0 = tangent_line(f, df, x0)

        # Нахождение минимума касательной на интервале [a, b]
        x1 = np.argmin([p0(x) for x in np.linspace(a, b, 1000)]) * (b - a) / 1000 + a

        history.append(x1)

        if abs(x1 - x0) < tol:  # Проверка на сходимость
            print(f"Сошелся за {i + 1} итераций.")
            break

        # Построение следующей касательной
        p1 = lambda x: max(f(x), p0(x))
        x0 = x1

    return x1, history


# Интервал поиска минимума
a, b = -2, 3

# Поиск минимума методом касательных
minimum, history = casat(f, df, a, b)

# График функции и касательных
x_vals = np.linspace(a, b, 400)
y_vals = f(x_vals)

plt.plot(x_vals, y_vals, label='f(x)')
plt.scatter(history, [f(x) for x in history], color='red', s=80, zorder=5, label="Итерации")
plt.scatter(minimum, f(minimum), color='black', s=100, zorder=10, label="Минимум")
plt.title(f'Метод касательных. Минимум при x = {minimum:.5f}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
