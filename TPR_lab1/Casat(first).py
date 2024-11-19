import numpy as np
import matplotlib.pyplot as plt


# Функция и её производная
def f(x):
    return (x**2)/2 - np.cos(x)  # Функция


def df(x):
    return x + np.sin(x)  # Производная функции

def TangentMethod(a, b, eps):
    # Xmin используется для хранения текущего приближения к точке минимума
    # C определяет следующую точку приближения
    Xming, c = 0, (f(b) - f(a) + df(a) * a - df(b) * b) / (df(a) - df(b))

    # Счетчик итераций
    iterations = 0

    #  пока абсолютное значение производной в точке c больше eps и длина отрезка [a, b] больше eps
    while (b - a) > eps: #abs(df(c)) > eps and
        iterations += 1  # Увеличиваем счетчик итераций

        # Если производная в точке c равна нулю, то c является точкой минимума, и цикл прерывается.
        if df(c) == 0:
            Xming = c
            break

        # Если производная в точке c положительна, то b обновляется значением c, иначе a обновляется значением c.
        elif df(c) > 0:
            b = c
        else:
            a = c

        # После каждой итерации c пересчитывается по формуле.
        c = (f(b) - f(a) + df(a) * a - df(b) * b) / (df(a) - df(b))

        # В конце цикла Xming присваивается значение c, которое является приближением к точке минимума.
        Xming = c
    return Xming, iterations  # Возвращаем Xming и количество итераций


# График
x = np.linspace(-3, 1, 100)
y = f(x)
y_derivative = df(x)  # Вычисляем значения производной

plt.plot(x, y, label="f(x)")
plt.plot(x, y_derivative, label="df(x)")  # Рисуем график производной
plt.xlabel("x")
plt.ylabel("y")
plt.title("Графики функции f(x) и ее производной df(x)")

# Вывод точки минимума
x_min, iterations = TangentMethod(-3, 1, 0.000001)
y_min = f(x_min)
plt.scatter(x_min, y_min, color='red', label="Точка минимума")

plt.legend()
plt.grid(True)
plt.show()

# Вывод результата в заданном формате
print(f"x = {x_min:.6f}; f(x) = {y_min:.6f}; Итераций: {iterations}")