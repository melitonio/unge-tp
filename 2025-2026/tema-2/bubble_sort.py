from dataclasses import dataclass


@dataclass
class Metrics:
    comps: int = 0
    swaps: int = 0


def bubble_sort(a):
    a = list(a)
    n = len(a)
    while True:
        swapped = False
        for j in range(0, n - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        n -= 1  # el último ya quedó en su sitio
        if not swapped or n <= 1:
            break
    return a


def bubble_sort_instrumented(a):
    a = list(a)
    n = len(a)
    m = Metrics()
    while True:
        swapped = False
        for j in range(0, n - 1):
            m.comps += 1
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                m.swaps += 1
                swapped = True
        n -= 1
        if not swapped or n <= 1:
            break
    return a, m


# Pruebas simples
if __name__ == "__main__":

    v = [5, 2, 4, 6, 1, 3, 4]
    sorted_arr, metrics = bubble_sort_instrumented(v)
    print("Array original:", v)
    print("Array ordenado:", sorted_arr)
    print("Comparaciones:", metrics.comps)
    print("Swaps:", metrics.swaps)
