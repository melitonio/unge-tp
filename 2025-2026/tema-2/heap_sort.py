from dataclasses import dataclass


@dataclass
class Metrics:
    comps: int = 0
    swaps: int = 0


def heapsort(a):
    a = list(a)
    n = len(a)

    def sift_down(i, end):
        # end es límite exclusivo: heap en [0..end-1]
        while True:
            left = 2*i + 1
            right = left + 1
            if left >= end:
                break
            j = left
            if right < end and a[right] > a[left]:
                j = right
            if a[i] >= a[j]:
                break
            a[i], a[j] = a[j], a[i]
            i = j

    # heapify bottom-up (Floyd)
    for i in range((n - 2) // 2, -1, -1):
        sift_down(i, n)

    # sortdown
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(0, end)
    return a


def heapsort_instrumented(a):
    a = list(a)
    n = len(a)
    m = Metrics()

    def sift_down(i, end):
        while True:
            left = 2*i + 1
            right = left + 1
            if left >= end:
                break
            j = left
            if right < end:
                m.comps += 1
                if a[right] > a[left]:
                    j = right
            # comparación con el mayor hijo
            m.comps += 1
            if a[i] >= a[j]:
                break
            a[i], a[j] = a[j], a[i]
            m.swaps += 1
            i = j

    # heapify
    for i in range((n - 2) // 2, -1, -1):
        sift_down(i, n)
    # sortdown
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        m.swaps += 1
        sift_down(0, end)
    return a, m


# Pruebas simples
if __name__ == "__main__":

    v = [5, 2, 4, 6, 1, 3, 4]
    sorted_arr, metrics = heapsort_instrumented(v)
    print("Array original:", v)
    print("Array ordenado:", sorted_arr)
    print("Comparaciones:", metrics.comps)
    print("Swaps:", metrics.swaps)
