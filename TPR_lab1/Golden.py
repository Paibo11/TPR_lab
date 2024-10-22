import matplotlib.pyplot as plt
import numpy as np


def golden_section_search(f, a, b, tol=1e-5):
    teta = (1 + np.sqrt(5)) / 2  # Золотое сечение
    resphi = 2 - teta
    iterations = []

    c = b - resphi * (b - a)
    d = a + resphi * (b - a)

    while abs(b - a) > tol:
        iterations.append((a, b))  # Для визуализации на графике
        if f(c) < f(d):
            b = d
        else:
            a = c

        c = b - resphi * (b - a)
        d = a + resphi * (b - a)

    minimum = (b + a) / 2
    return minimum, iterations


# Пример функции
def f(x):
    return (x**2)/2 - np.cos(x)


# Поиск минимума
a, b = 0, 3
minimum, iterations = golden_section_search(f, a, b)

# График функции и процесса поиска минимума
x_vals = np.linspace(a, b, 400)
y_vals = f(x_vals)

plt.plot(x_vals, y_vals, label='f(x)')
for i, (a, b) in enumerate(iterations):
    plt.scatter([a, b], [f(a), f(b)], s=50, color='red', zorder=5)  # Точки итераций

plt.scatter(minimum, f(minimum), color='black', s=100, zorder=10, label="Минимум")
plt.title(f'Метод золотого сечения. Минимум при x = {minimum:.5f}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
