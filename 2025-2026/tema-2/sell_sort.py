from dataclasses import dataclass


@dataclass
class Metrics:
    comps: int = 0
    moves: int = 0


def shell_sort(a):
    a = list(a)
    n = len(a)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            key = a[i]
            j = i
            while j >= gap and a[j - gap] > key:
                a[j] = a[j - gap]
                j -= gap
            a[j] = key
        gap //= 2
    return a


def shell_sort_instrumented(a, gaps=None):
    a = list(a)
    n = len(a)
    if gaps is None:
        # Halving clásico
        gaps = []
        g = n // 2
        while g > 0:
            gaps.append(g)
            g //= 2
    m = Metrics()
    for gap in gaps:
        for i in range(gap, n):
            key = a[i]
            j = i
            while j >= gap:
                m.comps += 1
                if a[j - gap] > key:
                    a[j] = a[j - gap]
                    m.moves += 1
                    j -= gap
                else:
                    break
            a[j] = key
            m.moves += 1
    return a, m


# Pruebas simples
if __name__ == "__main__":

    v = [5, 2, 4, 6, 1, 3, 4]
    sorted_arr, metrics = shell_sort_instrumented(v)
    print("Array original:", v)
    print("Array ordenado:", sorted_arr)
    print("Comparaciones:", metrics.comps)
    print("Movimientos:", metrics.moves)
