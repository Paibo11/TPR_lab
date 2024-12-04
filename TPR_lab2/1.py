import numpy as np
import matplotlib.pyplot as plt

# Метод Нелдера-Мида (Метод деформируемого многогранника)
def nelder_mead(f, x0, tol=1e-6, max_iter=1000):
    # Коэффициенты для алгоритма
    a = 0.9  # коэффициент отражения
    y = 2  # коэффициент растяжения
    b = 0.4  # коэффициент сжатия
    sigma = 0.5  # коэффициент уменьшения

    # Создание начального симплекса вокруг точки x0
    n = len(x0)
    simplex = [x0]

    # Смещение для создания симплекса (меньшее значение для функции с экспонентой)
    shift = 0.05  # Выбираем смещение на 0.05 для начала

    # Смещение для создания соседних точек
    for i in range(n):
        x = np.copy(x0)
        x[i] = x[i] + shift  # Смещаем по оси i
        simplex.append(x)

    # Вычисляем значения функции в вершинах симплекса
    f_values = [f(x) for x in simplex]

    iter_count = 0  # Счетчик итераций
    simplex_history = [np.array(simplex)]  # История симплексов для графика

    # Основной цикл оптимизации
    while iter_count < max_iter:
        # Сортировка вершин симплекса по значению функции
        indices = np.argsort(f_values)
        simplex = [simplex[i] for i in indices]
        f_values = [f_values[i] for i in indices]

        # Вычисление центра тяжести (исключая наихудшую точку)
        centroid = np.mean(simplex[:-1], axis=0)

        # Отражение
        xr = centroid + a * (centroid - simplex[-1])
        fr = f(xr)

        # Проверка условий для отражения, растяжения и сжатия
        if f_values[0] <= fr < f_values[-2]:
            simplex[-1] = xr
            f_values[-1] = fr
        elif fr < f_values[0]:
            # Растяжение, если отраженная точка лучше
            xe = centroid + y * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                f_values[-1] = fe
            else:
                simplex[-1] = xr
                f_values[-1] = fr
        else:
            # Сжатие
            xc = centroid + b * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < f_values[-1]:
                simplex[-1] = xc
                f_values[-1] = fc
            else:
                # Уменьшение симплекса
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                f_values = [f(x) for x in simplex]

        # Сохранение текущего симплекса для визуализации
        simplex_history.append(np.array(simplex))

        # Условие остановки
        if np.max(np.abs(np.array(f_values) - f_values[0])) < tol:
            break

        iter_count += 1

    return simplex[0], f_values[0], iter_count, simplex_history

# Функция для минимизации
def f(x):
    x1, x2 = x[0], x[1]
    return np.sqrt(1 + 2*x1**2 + x2**2) + np.exp(x1**2 + 2*x2**2) - x1 - x2

# Начальная точка
x0 = np.array([1.0, 1.0])

# Запуск метода Нелдера-Мида
minimum, f_min, iterations, simplex_history = nelder_mead(f, x0)

print("Минимум функции:", minimum)
print("Значение функции в минимуме:", f_min)
print("Количество итераций:", iterations)

# Построение графика
x_range = np.linspace(-1, 1.5, 200)
y_range = np.linspace(-1, 1.5, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = np.sqrt(1 + 2*X**2 + Y**2) + np.exp(X**2 + 2*Y**2) - X - Y

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=50, cmap="viridis")
plt.colorbar(label="f(x)")

for simplex in simplex_history:
    plt.plot(simplex[:, 0], simplex[:, 1], 'k-', alpha=0.4)
    plt.plot(simplex[:, 0], simplex[:, 1], 'ko', markersize=3)

plt.plot(x0[0], x0[1], 'bo', label='Начальная точка (x0)')
plt.plot(minimum[0], minimum[1], 'ro', label='Минимум функции')

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Минимизация функции методом Нелдера-Мида")
plt.legend()

plt.show()
