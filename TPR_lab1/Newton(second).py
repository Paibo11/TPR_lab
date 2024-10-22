import numpy as np
import matplotlib.pyplot as plt


# Пример функции, её первая и вторая производные
def f(x):
    return (x**2)/2 - np.cos(x)  # Квадратичная функция


def df(x):
    return x + np.sin(x) # Первая производная


def ddf(x):
    return 1 + np.cos(x)  # Вторая производная


# Метод Ньютона для нахождения минимума
def newton_method(f, df, ddf, x0, tol=1e-2, max_iter=100):
    x = x0
    history = [x]  # Сохраняем историю точек для визуализации

    for i in range(max_iter):
        f_prime = df(x)
        f_double_prime = ddf(x)

        if abs(f_prime) < tol:  # Если градиент мал, останавливаемся
            print(f"Минимум найден за {i + 1} итераций.")
            break

        # Обновляем x по методу Ньютона
        x_next = x - f_prime / f_double_prime
        history.append(x_next)

        if abs(x_next - x) < tol:  # Проверка на сходимость
            break

        x = x_next

    return x, history


# Начальная точка
x0 = 2

# Поиск минимума методом Ньютона
minimum, history = newton_method(f, df, ddf, x0)

# График функции и процесса нахождения минимума
x_vals = np.linspace(-3, 3, 400)
y_vals = f(x_vals)

plt.plot(x_vals, y_vals, label='f(x)')
plt.scatter(history, [f(x) for x in history], color='red', s=80, zorder=5, label="Итерации")
plt.scatter(minimum, f(minimum), color='black', s=100, zorder=10, label="Минимум")
plt.title(f'Метод Ньютона. Минимум при x = {minimum:.5f}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
