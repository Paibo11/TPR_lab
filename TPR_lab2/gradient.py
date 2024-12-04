import numpy as np
import matplotlib.pyplot as plt

# Определяем функцию и её градиент
def f(x1, x2):
    return x1**2 + x2**2 + np.exp(x1 + x2)

def grad_f(x1, x2):
    df_dx1 = 2*x1 + np.exp(x1 + x2)
    df_dx2 = 2*x2 + np.exp(x1 + x2)
    return np.array([df_dx1, df_dx2])

# Градиентный метод с дроблением шага
def gradient_descent_with_line_search(f, grad_f, x0, alpha=0.1, tol=1e-6, max_iter=1000):
    x = np.array(x0)
    iter_count = 0
    trajectory = [x.copy()]  # Список для хранения траектории
    while iter_count < max_iter:
        grad = grad_f(x[0], x[1])
        # Снижение шага, если нужно
        step_size = alpha
        new_x = x - step_size * grad
        # Проверяем, если шаг улучшает функцию
        if f(new_x[0], new_x[1]) < f(x[0], x[1]):
            x = new_x
        else:
            # Если не улучшает, уменьшаем шаг
            alpha *= 0.5
        iter_count += 1
        # Добавляем точку в траекторию
        trajectory.append(x.copy())
        # Проверяем условие сходимости
        if np.linalg.norm(grad) < tol:
            break
    return x, f(x[0], x[1]), iter_count, trajectory

# Начальная точка
x0 = (1, 1)

# Запуск градиентного спуска
minimum, min_value, iterations, trajectory = gradient_descent_with_line_search(f, grad_f, x0)

print(f"Минимум найден в точке: {minimum}")
print(f"Значение функции в этой точке: {min_value}")
print(f"Количество итераций: {iterations}")

# Построение графика
# Создание сетки для отображения уровней функции
x_range = np.linspace(-1, 1.5, 200)
y_range = np.linspace(-1, 1.5, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = X ** 2 + Y ** 2 + np.exp(X + Y)

# Настройка графика
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=50, cmap="viridis")  # Контурный график функции
plt.colorbar(label="f(x)")  # Цветовая шкала значений функции

# Отображение траектории градиентного спуска
trajectory = np.array(trajectory)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'k-', alpha=0.6)  # Линия, соединяющая точки
plt.plot(trajectory[:, 0], trajectory[:, 1], 'ko', markersize=4)  # Вершины траектории

# Обозначение начальной и конечной точек
plt.plot(x0[0], x0[1], 'bo', label='Начальная точка (x0)')  # Начальная точка
plt.plot(minimum[0], minimum[1], 'ro', label='Минимум функции')  # Найденный минимум

# Подписи осей и легенда
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Минимизация функции методом градиентного спуска с дроблением шага")
plt.legend()

# Показ графика
plt.show()
