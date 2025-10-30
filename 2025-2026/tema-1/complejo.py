from dataclasses import dataclass
from math import sqrt

# Definición de la clase Complejo


@dataclass
class Complejo:
    real: float
    imag: float

# Representación en cadena
    def __repr__(self) -> str:
        return f"{self.real} + {self.imag}i"

# Cálculo del módulo
    def modulo(self) -> float:
        return sqrt(self.real*self.real + self.imag*self.imag)

# Suma de dos números complejos
    def sumar(self, c: "Complejo") -> "Complejo":
        return Complejo(self.real + c.real, self.imag + c.imag)


# Constantes de números complejos
a = Complejo(3.0, 4.0)
b = Complejo(1.0, 2.0)
suma_ab = a.sumar(b)

# Ejemplos de uso
print("Módulo de", a, ":", a.modulo())
print("Suma de", a, "y", b, ":", suma_ab)
print("Módulo de la suma:", suma_ab.modulo())
