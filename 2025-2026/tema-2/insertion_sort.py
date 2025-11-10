from dataclasses import dataclass


@dataclass
class Metrics:
    comps: int = 0
    moves: int = 0


def insertion_sort_instrumented(a):
    a = list(a)  # copia para no mutar el original
    m = Metrics()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        # (No contamos leer key como movimiento sobre el array)
        while j >= 0:
            m.comps += 1
            if a[j] > key:
                a[j + 1] = a[j]
                m.moves += 1
                j -= 1
            else:
                break
        a[j + 1] = key
        m.moves += 1
    return a, m


def insertion_sort(a):
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        # Desplaza elementos mayores que key una posición a la derecha
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


# Pruebas simples
if __name__ == "__main__":

    v = [5, 2, 4, 6, 1, 3, 4]
    sorted_arr, metrics = insertion_sort_instrumented(v)
    print("Array original:", v)
    print("Array ordenado:", sorted_arr)
    print("Comparaciones:", metrics.comps)
    print("Movimientos:", metrics.moves)
