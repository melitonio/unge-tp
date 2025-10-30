class Lista:
    def __init__(self):
        self._items = []

    # insertar en pos p (0..n)
    def insertar(self, x, p):
        self._items.insert(p, x)

    # suprimir en pos p (0..n-1)
    def suprimir(self, p):
        self._items.pop(p)

    # recuperar en pos p (0..n-1)
    def recuperar(self, p):
        return self._items[p]

    # primera posición o -1 si no está
    def localizar(self, x):
        return self._items.index(x) if x in self._items else -1

    # número de elementos
    def longitud(self):
        return len(self._items)

    # anular lista
    def anula(self):
        self._items.clear()

    # imprimir lista
    def imprimir(self):
        print(self._items)


# Ejemplo de uso
if __name__ == "__main__":
    lista = Lista()
    lista.insertar(10, 0)
    lista.insertar(20, 1)
    lista.insertar(15, 1)
    lista.imprimir()  # Salida: [10, 15, 20]
    print("Elemento en posición 1:", lista.recuperar(1))  # Salida: 15
    print("Longitud de la lista:", lista.longitud())  # Salida: 3
    lista.suprimir(1)
    lista.imprimir()  # Salida: [10, 20]
    print("Localizar elemento 20:", lista.localizar(20))  # Salida: 1
    lista.anula()
    lista.imprimir()  # Salida: []
