def traza_mcd_clasico(m, n):
    i = min(m, n)
    pasos = 0
    while not (m % i == 0 and n % i == 0):
        print(f"paso {pasos+1:>3} | i = {i} | m % i = {(m % i):>2} | n % i = {n % i}")
        i -= 1
        pasos += 1
    print(f"MCD={i} en {pasos+1} pasos")


def traza_mcd_euclides(m, n):
    k = 0
    while m > 0:
        print(f"paso {k:>5} | (m,n) = ({m},{n:>5})  | n % m = {n % m:>3}")
        m, n = n % m, m
        k += 1
    print(f"MCD={n} en {k} pasos")


c = 180
d = 40
print("\nTrazas del cálculo del MCD clásico:")
traza_mcd_clasico(c, d)

print("\nTrazas del cálculo del MCD de Euclides:")
traza_mcd_euclides(c, d)
