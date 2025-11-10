from dataclasses import dataclass
import random


@dataclass
class QSMetrics:
    comps: int = 0
    swaps: int = 0


# Partición de Hoare
def _partition_hoare(a, lo, hi):
    pivot = a[lo]
    i, j = lo - 1, hi + 1
    while True:
        i += 1
        while a[i] < pivot:
            i += 1
        j -= 1
        while a[j] > pivot:
            j -= 1
        if i >= j:
            return j
        a[i], a[j] = a[j], a[i]


# Quicksort con partición de Hoare y optimización de cola
def quicksort_hoare(a):
    a = list(a)

    def _qs(lo, hi):
        while lo < hi:
            p = _partition_hoare(a, lo, hi)
            # Tail-recursion elimination: entrar por la rama más pequeña
            if p - lo < hi - (p + 1):
                _qs(lo, p)
                lo = p + 1
            else:
                _qs(p + 1, hi)
                hi = p
    _qs(0, len(a) - 1)
    return a


# Partición de Lomuto
def _partition_lomuto(a, lo, hi):
    pivot = a[hi]
    i = lo
    for j in range(lo, hi):
        if a[j] < pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[hi] = a[hi], a[i]
    return i


# Quicksort con partición de Lomuto y optimización de cola
def quicksort_lomuto(a):
    a = list(a)

    def _qs(lo, hi):
        while lo < hi:
            p = _partition_lomuto(a, lo, hi)
            # procesar subarray más pequeño primero
            if p - lo < hi - p:
                _qs(lo, p - 1)
                lo = p + 1
            else:
                _qs(p + 1, hi)
                hi = p - 1
    _qs(0, len(a) - 1)
    return a


# Quicksort con partición 3-way y optimización de cola
def quicksort_3way(a):
    a = list(a)

    def _qs(lo, hi):
        while lo < hi:
            # pivote aleatorio para robustez
            p = random.randint(lo, hi)
            a[lo], a[p] = a[p], a[lo]
            pivot = a[lo]
            lt, i, gt = lo, lo + 1, hi
            # partición 3-way
            while i <= gt:
                if a[i] < pivot:
                    a[lt], a[i] = a[i], a[lt]
                    lt += 1
                    i += 1
                elif a[i] > pivot:
                    a[i], a[gt] = a[gt], a[i]
                    gt -= 1
                else:  # ==
                    i += 1
            # ahora [lo..lt-1] <, [lt..gt] ==, [gt+1..hi] >
            # ordenar la parte menor primero para limitar profundidad
            if (lt - lo) < (hi - gt):
                _qs(lo, lt - 1)
                lo = gt + 1
            else:
                _qs(gt + 1, hi)
                hi = lt - 1
    _qs(0, len(a) - 1)
    return a


# Quicksort con partición de Hoare instrumentado
def quicksort_hoare_instrumented(a):
    a = list(a)
    m = QSMetrics()

    def part(lo, hi):
        pivot = a[lo]
        i, j = lo - 1, hi + 1
        while True:
            i += 1
            while True:
                m.comps += 1
                if not (a[i] < pivot):
                    break
                i += 1
            j -= 1
            while True:
                m.comps += 1
                if not (a[j] > pivot):
                    break
                j -= 1
            if i >= j:
                return j
            a[i], a[j] = a[j], a[i]
            m.swaps += 1

    def _qs(lo, hi):
        while lo < hi:
            p = part(lo, hi)
            if p - lo < hi - (p + 1):
                _qs(lo, p)
                lo = p + 1
            else:
                _qs(p + 1, hi)
                hi = p

    _qs(0, len(a) - 1)
    return a, m


# Pruebas simples
if __name__ == "__main__":

    v = [5, 2, 4, 6, 1, 3, 4]
    sorted_arr, metrics = quicksort_hoare_instrumented(v)
    print("Array original:", v)
    print("Array ordenado:", sorted_arr)
    print("Comparaciones:", metrics.comps)
    print("Swaps:", metrics.swaps)
    print("\nOtras versiones de Quicksort:")

    A = [5, 2, 4, 6, 1, 3, 4]
    print(quicksort_hoare(A))
    print(quicksort_lomuto(A))
    print(quicksort_3way(A))
    print(quicksort_hoare_instrumented(A))
