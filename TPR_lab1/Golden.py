import matplotlib.pyplot as plt
import numpy as np

#Функция
def f(x):
    return (x**2)/2 - np.cos(x)


# Метод золотого сечения
def golden_section_search(f, a, b, tol=1e-5):  # tol - погрешность
    phi = (1 + np.sqrt(5)) / 2  # phi — это золотое сечение, а resphi — его дополнение, используемое для вычислений
    resphi = 2 - phi

    # Начальные точки
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    # Список для хранения итерационных точек
    points = [(x1, f1), (x2, f2)]

    # Счетчик итераций
    iterations = 0

    while abs(b - a) > tol:
        iterations += 1  # Увеличиваем счетчик итераций
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = f(x2)

        # Сохраняем текущие точки
        points.append((x1, f1))
        points.append((x2, f2))

    return (a + b) / 2, points, iterations  # Возвращаем также количество итераций

# Границы поиска
a = 0
b = 3

# Находим минимум
minimum, points, iterations = golden_section_search(f, a, b)
print(f"Минимум функции находится в точке x = {minimum}, f(x) = {f(minimum)}")
print(f"Количество итераций для нахождения минимума: {iterations}")

# Построение графика функции
x = np.linspace(a, b, 30)
y = f(x)

plt.plot(x, y, label='f(x) = (x^2)/2 - cos(x)')

# Отображение итерационных точек
for point in points:
    plt.scatter(point[0], point[1], color='blue', alpha=0.5)

# Отображение точки минимума
plt.scatter(minimum, f(minimum), color='orange', s=100, label='Минимум', edgecolor='black')

plt.title('График функции и её минимум')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()
