# Представляет ячейку таблицы с координатами строки (row) и столбца (col)
class Cell:
    def __init__(self, row, col):
        self.row = row
        self.col = col


# Для построения циклов перераспределения
class State:
    def __init__(self, cell, prev_dir, next_cells):
        self.cell = cell
        self.prev_dir = prev_dir
        self.next_cells = next_cells

# Начальное распределение методом северо-западного угла
def solve(a, b, costs):
    assert len(costs) == len(a)
    assert len(costs[0]) == len(b)
    a_sum = sum(a)
    b_sum = sum(b)

    if a_sum > b_sum:
        b.append(a_sum - b_sum)
        for row in costs:
            row.append(0)
    elif a_sum < b_sum:
        a.append(b_sum - a_sum)
        costs.append([0] * len(b))

    x = [[0] * len(b) for _ in range(len(a))]
    a_copy = a[:]
    b_copy = b[:]

    indexes_for_baza = []
    i, j = 0, 0
    while True:
        if a_copy[i] < b_copy[j]:
            x[i][j] = a_copy[i]
            indexes_for_baza.append(Cell(i, j))
            b_copy[j] -= a_copy[i]
            a_copy[i] = 0
            i += 1
        else:
            x[i][j] = b_copy[j]
            indexes_for_baza.append(Cell(i, j))
            a_copy[i] -= b_copy[j]
            b_copy[j] = 0
            j += 1

        if sum(a_copy) == 0 and sum(b_copy) == 0:
            break

    result = sum(x[cell.row][cell.col] * costs[cell.row][cell.col] for cell in indexes_for_baza)
    print(f"Z = {result} (метод северо-западного угла)")

    potential_method(a, b, x, costs, indexes_for_baza)

# Улучшение базисного плана методом потенциалов
def potential_method(a, b, x, costs, indexes_for_baza):
    m, n = len(a), len(b)

    while True:
        indexes_for_baza.sort(key=lambda cell: (cell.row, cell.col))

        u = [0] * m
        v = [0] * n
        fill_u = [False] * m
        fill_v = [False] * n
        fill_u[0] = True

        while not all(fill_u) or not all(fill_v):
            for cell in indexes_for_baza:
                i, j = cell.row, cell.col
                if fill_u[i]:
                    v[j] = costs[i][j] - u[i]
                    fill_v[j] = True
                elif fill_v[j]:
                    u[i] = costs[i][j] - v[j]
                    fill_u[i] = True

        not_optimal_cells = []
        economies = []
        for i in range(m):
            for j in range(n):
                if all(cell.row != i or cell.col != j for cell in indexes_for_baza):
                    diff = u[i] + v[j] - costs[i][j]
                    if diff > 0:
                        not_optimal_cells.append(Cell(i, j))
                        economies.append(diff)

        if not not_optimal_cells:
            break

        max_economy = max(economies)
        cells_with_max_economy = [cell for cell, economy in zip(not_optimal_cells, economies) if economy == max_economy]

        min_cost_cell = min(cells_with_max_economy, key=lambda cell: costs[cell.row][cell.col])
        indexes_for_baza.append(min_cost_cell)

        path = build_path(min_cost_cell, indexes_for_baza)

        minus_cells = path[1::2]
        min_x_value = min(x[cell.row][cell.col] for cell in minus_cells)

        for idx, cell in enumerate(path):
            if idx % 2 == 0:
                x[cell.row][cell.col] += min_x_value
            else:
                x[cell.row][cell.col] -= min_x_value

        for cell in minus_cells:
            if x[cell.row][cell.col] == 0:
                indexes_for_baza.remove(cell)

    result = sum(x[cell.row][cell.col] * costs[cell.row][cell.col] for cell in indexes_for_baza)
    print(f"Z = {result} (метод потенциалов)")

# Замкнутый цикл для перераспределения значений
def build_path(start_cell, baza_cells):
    stack = [State(start_cell, 'v',
                   [cell for cell in baza_cells if cell.row == start_cell.row and cell.col != start_cell.col])]

    while stack:
        head = stack[-1]

        if len(stack) >= 4 and ((head.cell.row == start_cell.row) or (head.cell.col == start_cell.col)):
            break

        if not head.next_cells:
            stack.pop()
            continue

        next_cell = head.next_cells.pop()
        next_dir = 'h' if head.prev_dir == 'v' else 'v'
        next_cells = [
            cell for cell in baza_cells
            if (cell.col == next_cell.col if next_dir == 'h' else cell.row == next_cell.row)
               and (cell.row != next_cell.row if next_dir == 'h' else cell.col != next_cell.col)
        ]

        stack.append(State(next_cell, next_dir, next_cells))

    return [state.cell for state in stack]


if __name__ == "__main__":
    a = [30, 50, 20]
    b = [15, 15, 40, 30]
    costs = [
        [1, 8, 2, 3],
        [4, 7, 5, 1],
        [5, 3, 4, 4]
    ]

    solve(a, b, costs)

    # a = [30, 20, 40, 50]
    # b = [35, 20, 55, 30]
    # costs = [
    #     [2, 4, 1, 3],
    #     [5, 6, 5, 4],
    #     [3, 7, 9, 5],
    #     [1, 2, 2, 7]
    # ]
