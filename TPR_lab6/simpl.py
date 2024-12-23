import numpy as np
def simplex_method(c, A, b, num_artificial):

    # Симплекс-метод с учетом искусственных переменных.

    num_constraints, num_variables = A.shape
    tableau = np.zeros((num_constraints + 1, num_variables + 1))
    tableau[:num_constraints, :num_variables] = A
    tableau[:num_constraints, -1] = b

    # Добавляем искусственные переменные
    artificial_costs = np.zeros(num_variables)
    artificial_costs[-num_artificial:] = 1
    tableau[-1, :num_variables] = artificial_costs
    tableau[-1, -1] = sum(b)

    while True:
        pivot_col = np.argmin(tableau[-1, :-1])
        if tableau[-1, pivot_col] >= 0:
            break
        ratios = [tableau[i, -1] / tableau[i, pivot_col] if tableau[i, pivot_col] > 0 else np.inf
                  for i in range(num_constraints)]
        pivot_row = np.argmin(ratios)
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        for i in range(num_constraints + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    # Удаление искусственных переменных
    tableau = np.delete(tableau, np.s_[-num_artificial-1:-1], axis=1)
    c = np.hstack([c, np.zeros(tableau.shape[1] - len(c) - 1)])
    tableau[-1, :] = 0
    tableau[-1, :len(c)] = -c

    while True:
        pivot_col = np.argmin(tableau[-1, :-1])
        if tableau[-1, pivot_col] >= 0:
            break
        ratios = [tableau[i, -1] / tableau[i, pivot_col] if tableau[i, pivot_col] > 0 else np.inf
                  for i in range(num_constraints)]
        pivot_row = np.argmin(ratios)
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        for i in range(num_constraints + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    solution = np.zeros(len(c))
    for j in range(len(c)):
        col = tableau[:, j]
        if sum(col[:-1] == 1) == 1 and sum(col[:-1] == 0) == num_constraints - 1:
            solution[j] = tableau[np.argmax(col[:-1]), -1]

    return solution, tableau[-1, -1]

def preprocess_constraints(A, b, signs):

    # Приведение ограничений к стандартному виду.

    num_constraints, num_variables = A.shape
    slack_vars = []
    artificial_vars = []
    new_A = []
    new_b = []

    for i in range(num_constraints):
        if signs[i] == '<=':
            slack = np.zeros(num_constraints)
            slack[i] = 1
            slack_vars.append(slack)
            artificial_vars.append(np.zeros(num_constraints))
            new_A.append(A[i])
            new_b.append(b[i])
        elif signs[i] == '>=':
            slack = np.zeros(num_constraints)
            slack[i] = -1
            artificial = np.zeros(num_constraints)
            artificial[i] = 1
            slack_vars.append(slack)
            artificial_vars.append(artificial)
            new_A.append(A[i])
            new_b.append(b[i])
        elif signs[i] == '=':
            slack_vars.append(np.zeros(num_constraints))
            artificial = np.zeros(num_constraints)
            artificial[i] = 1
            artificial_vars.append(artificial)
            new_A.append(A[i])
            new_b.append(b[i])

    new_A = np.hstack([np.array(new_A),
                       np.array(slack_vars).T,
                       np.array(artificial_vars).T])
    return np.array(new_A), np.array(new_b), len(slack_vars[0]), len(artificial_vars[0])


# Пример задачи
c = np.array([7, 6])
A = np.array([
    [2, 5],
    [5, 2],
    [1, 0],
    [0, 1]
])
b = np.array([10, 10, 6, 5])
signs = ['>=', '>=', '<=', '<=']

# Приведение ограничений
A, b, num_slack, num_artificial = preprocess_constraints(A, b, signs)

# Решение симплекс-методом
solution, optimal_value = simplex_method(c, A, b, num_artificial)

print("Решение:", solution)
print("Оптимальное значение целевой функции:", optimal_value)
