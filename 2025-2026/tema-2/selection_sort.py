from dataclasses import dataclass


@dataclass
class Metrics:
    comps: int = 0
    swaps: int = 0


def selection_sort(a):
    n = len(a)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        if min_idx != i:
            a[i], a[min_idx] = a[min_idx], a[i]
    return a


def selection_sort_instrumented(a):
    a = list(a)
    n = len(a)
    m = Metrics()
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            m.comps += 1
            if a[j] < a[min_idx]:
                min_idx = j
        if min_idx != i:
            a[i], a[min_idx] = a[min_idx], a[i]
            m.swaps += 1
    return a, m


# Pruebas simples
if __name__ == "__main__":

    v = [5, 2, 4, 6, 1, 3, 4]
    sorted_arr, metrics = selection_sort_instrumented(v)
    print("Array original:", v)
    print("Array ordenado:", sorted_arr)
    print("Comparaciones:", metrics.comps)
    print("Swaps:", metrics.swaps)
