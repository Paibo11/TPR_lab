import numpy as np
import matplotlib.pyplot as plt

# Новая функция для минимизации
def f(x):
    x1, x2 = x[0], x[1]
    return x1**2 + x1*x2 + x2**2

# Функция штрафа
def penalty(x):
    g = x[0] + x[1] - 2  # Уравнение плоскости
    return g**2

# Новая целевая функция с штрафом
def penalized_function(x, penalty_weight):
    return f(x) + penalty_weight * penalty(x)

# Метод Нелдера-Мида (Метод деформируемого многогранника)
def nelder_mead(f, x0, tol=1e-6, max_iter=1000):
    a = 0.9  # коэффициент отражения
    y = 2    # коэффициент растяжения
    b = 0.4  # коэффициент сжатия
    sigma = 0.5  # коэффициент уменьшения

    n = len(x0)
    simplex = [x0]
    shift = 0.05

    for i in range(n):
        x = np.copy(x0)
        x[i] = x[i] + shift
        simplex.append(x)

    f_values = [f(x) for x in simplex]
    iter_count = 0
    simplex_history = [np.array(simplex)]

    while iter_count < max_iter:
        indices = np.argsort(f_values)
        simplex = [simplex[i] for i in indices]
        f_values = [f_values[i] for i in indices]
        centroid = np.mean(simplex[:-1], axis=0)

        xr = centroid + a * (centroid - simplex[-1])
        fr = f(xr)

        if f_values[0] <= fr < f_values[-2]:
            simplex[-1] = xr
            f_values[-1] = fr
        elif fr < f_values[0]:
            xe = centroid + y * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                f_values[-1] = fe
            else:
                simplex[-1] = xr
                f_values[-1] = fr
        else:
            xc = centroid + b * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < f_values[-1]:
                simplex[-1] = xc
                f_values[-1] = fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                f_values = [f(x) for x in simplex]

        simplex_history.append(np.array(simplex))

        if np.max(np.abs(np.array(f_values) - f_values[0])) < tol:
            break

        iter_count += 1

    return simplex[0], f_values[0], iter_count, simplex_history


# Начальная точка
x0 = np.array([1.0, 1.0])

# Запуск метода Нелдера-Мида с учетом штрафа
penalty_weight = 0.1
minimum_penalty, f_min_penalty, iterations_penalty, simplex_history_penalty = nelder_mead(
    lambda x: penalized_function(x, penalty_weight), x0
)

print("Минимум функции с учетом штрафа:", minimum_penalty)
print("Значение функции в минимуме (с учетом штрафа):", f_min_penalty)
print("Количество итераций:", iterations_penalty)

# Построение графика
x_range = np.linspace(-1, 2, 200)
y_range = np.linspace(-1, 2, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = X**2 + X*Y + Y**2

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=50, cmap="viridis")
plt.colorbar(label="f(x)")

# Отображение симплексов
for simplex in simplex_history_penalty:
    plt.plot(simplex[:, 0], simplex[:, 1], 'k-', alpha=0.4)
    plt.plot(simplex[:, 0], simplex[:, 1], 'ko', markersize=3)

# Отображение ограничения
x_line = np.linspace(-1, 1.5, 200)
y_line = -(x_line + 5) / 8
plt.plot(x_line, y_line, 'r--', label='Ограничение: $x_1 + 8x_2 + 5 = 0$')

# Точки минимума
plt.plot(x0[0], x0[1], 'bo', label='Начальная точка (x0)')
plt.plot(minimum_penalty[0], minimum_penalty[1], 'ro', label='Минимум функции с учетом штрафа')

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Минимизация функции методом Нелдера-Мида с учетом штрафа")
plt.legend()

plt.show()
