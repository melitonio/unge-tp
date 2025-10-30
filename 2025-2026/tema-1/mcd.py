import time


# Máximo Común Divisor (MCD)
def mcd_clasico(m, n):
    i = min(m, n)
    while not (m % i == 0 and n % i == 0):
        i -= 1
    return i


# MCD usando el algoritmo de Euclides
def mcd_euclides(m, n):
    while m > 0:
        m, n = n % m, m
    return n


# Medir tiempos de ejecución
def medir(f, m, n):
    t0 = time.perf_counter()
    r = f(m, n)
    t1 = time.perf_counter()
    return r, (t1 - t0)


a = 96021917
b = 80620729
r_clasico, dt1 = medir(mcd_clasico, b, a)
r_euclides, dt2 = medir(mcd_euclides, b, a)
print(f"MCD clásico: {r_clasico} | tiempo: {dt1:.10f} segundos")
print(f"MCD euclides: {r_euclides} | tiempo: {dt2:.10f} segundos")
