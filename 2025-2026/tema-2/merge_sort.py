from dataclasses import dataclass


@dataclass
class Metrics:
    comps: int = 0
    moves: int = 0   # asignaciones en array de salida


# Merge Sort básico, no instrumentado
def merge_sort(a):
    if len(a) <= 1:
        return a[:]
    mid = len(a) // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    return _merge(left, right)


# Función auxiliar de merge
def _merge(left, right):
    i = j = 0
    out = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:     # <= preserva estabilidad
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    if i < len(left):
        out.extend(left[i:])
    if j < len(right):
        out.extend(right[j:])
    return out


# Merge Sort in-place (usa buffer auxiliar)
def merge_sort_inplace(a):
    a = list(a)
    buf = [None] * len(a)
    _ms_inplace(a, buf, 0, len(a))
    return a


# Función auxiliar de merge in-place
def _ms_inplace(a, buf, lo, hi):
    if hi - lo <= 1:
        return
    mid = (lo + hi) // 2
    _ms_inplace(a, buf, lo, mid)
    _ms_inplace(a, buf, mid, hi)
    _merge_inplace(a, buf, lo, mid, hi)


# Función auxiliar de merge in-place
def _merge_inplace(a, buf, lo, mid, hi):
    # copia a[lo:hi] al buffer
    buf[lo:hi] = a[lo:hi]
    i, j, k = lo, mid, lo
    while i < mid and j < hi:
        if buf[i] <= buf[j]:     # estable
            a[k] = buf[i]
            i += 1
        else:
            a[k] = buf[j]
            j += 1
        k += 1
    while i < mid:
        a[k] = buf[i]
        i += 1
        k += 1
    # si quedan en j..hi ya están colocados


# Merge Sort in-place instrumentado
def merge_sort_instrumented(a):
    a = list(a)
    buf = [None] * len(a)
    m = Metrics()
    _ms_instr(a, buf, 0, len(a), m)
    return a, m


# Función auxiliar de merge in-place instrumentado
def _ms_instr(a, buf, lo, hi, m):
    if hi - lo <= 1:
        return
    mid = (lo + hi) // 2
    _ms_instr(a, buf, lo, mid, m)
    _ms_instr(a, buf, mid, hi, m)
    # merge con contadores
    buf[lo:hi] = a[lo:hi]
    i, j, k = lo, mid, lo
    while i < mid and j < hi:
        m.comps += 1
        if buf[i] <= buf[j]:
            a[k] = buf[i]
            i += 1
            m.moves += 1
        else:
            a[k] = buf[j]
            j += 1
            m.moves += 1
        k += 1
    while i < mid:
        a[k] = buf[i]
        i += 1
        k += 1
        m.moves += 1
    # elementos del lado derecho restantes ya están en su sitio


# Pruebas simples
if __name__ == "__main__":

    v = [5, 2, 4, 6, 1, 3, 4]
    sorted_arr, metrics = merge_sort_instrumented(v)
    print("Array original:", v)
    print("Array ordenado:", sorted_arr)
    print("Comparaciones:", metrics.comps)
    print("Movimientos:", metrics.moves)
