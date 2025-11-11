# pip install pandas
# pip install matplotlib
# pip install seaborn

import random
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- Generadores de datos ----------


def gen_random(n, *, seed=123):
    rnd = random.Random(seed)
    return [rnd.randint(0, 10**6) for _ in range(n)]


def gen_sorted(n):
    return list(range(n))


def gen_reverse(n):
    return list(range(n, 0, -1))


def gen_nearly_sorted(n, swaps=int):
    a = gen_sorted(n)
    # ~1% de pares intercambiados por defecto
    k = max(1, n // 100)
    for i in range(k):
        i1 = (37*i + 13) % n
        i2 = (53*i + 7) % n
        a[i1], a[i2] = a[i2], a[i1]
    return a


def gen_many_dups(n, distinct=10):
    rnd = random.Random(42)
    return [rnd.randint(0, distinct-1) for _ in range(n)]


# ---------- Métricas ----------


@dataclass
class Metrics:
    comps: int = 0
    moves: int = 0  # o swaps, según algoritmo


# ---------- Inserción ----------


def insertion_sort_instr(a: List[int]) -> Tuple[List[int], Metrics]:
    a = list(a)
    m = Metrics()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0:
            m.comps += 1
            if a[j] > key:
                a[j+1] = a[j]
                m.moves += 1
                j -= 1
            else:
                break
        a[j+1] = key
        m.moves += 1
    return a, m


# ---------- Selección ----------


def selection_sort_instr(a: List[int]) -> Tuple[List[int], Metrics]:
    a = list(a)
    m = Metrics()
    n = len(a)
    swaps = 0
    for i in range(n-1):
        min_idx = i
        for j in range(i+1, n):
            m.comps += 1
            if a[j] < a[min_idx]:
                min_idx = j
        if min_idx != i:
            a[i], a[min_idx] = a[min_idx], a[i]
            swaps += 1
    m.moves = swaps
    return a, m


# ---------- Burbuja (bandera) ----------


def bubble_sort_instr(a: List[int]) -> Tuple[List[int], Metrics]:
    a = list(a)
    m = Metrics()
    n = len(a)
    while True:
        swapped = False
        for j in range(0, n-1):
            m.comps += 1
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
                m.moves += 1
                swapped = True
        n -= 1
        if not swapped or n <= 1:
            break
    return a, m


# ---------- Shell (gaps Knuth) ----------


def knuth_gaps(n: int):
    gaps = []
    h = 1
    while h < n:
        gaps.append(h)
        h = 3*h + 1
    return list(reversed(gaps))


def shell_sort_instr(a: List[int]) -> Tuple[List[int], Metrics]:
    a = list(a)
    m = Metrics()
    n = len(a)
    for gap in knuth_gaps(n):
        for i in range(gap, n):
            key = a[i]
            j = i
            while j >= gap:
                m.comps += 1
                if a[j-gap] > key:
                    a[j] = a[j-gap]
                    m.moves += 1
                    j -= gap
                else:
                    break
            a[j] = key
            m.moves += 1
    return a, m


# ---------- Merge (in-place con buffer) ----------


def merge_sort_instr(a: List[int]) -> Tuple[List[int], Metrics]:
    a = list(a)
    m = Metrics()
    buf = [None] * len(a)

    def ms(lo, hi):
        if hi - lo <= 1:
            return
        mid = (lo + hi)//2
        ms(lo, mid)
        ms(mid, hi)
        # merge
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
    ms(0, len(a))
    return a, m


# ---------- Quick (Hoare + corte a inserción + mediana-de-tres) ----------


def insertion_tail(a, lo, hi):
    for i in range(lo+1, hi+1):
        key = a[i]
        j = i-1
        while j >= lo and a[j] > key:
            a[j+1] = a[j]
            j -= 1
        a[j+1] = key


def median3(a, lo, mid, hi):
    if a[mid] < a[lo]:
        a[lo], a[mid] = a[mid], a[lo]
    if a[hi] < a[lo]:
        a[lo], a[hi] = a[hi], a[lo]
    if a[hi] < a[mid]:
        a[mid], a[hi] = a[hi], a[mid]
    return a[mid]


def quick_hoare_instr(a: List[int], cutoff=16) -> Tuple[List[int], Metrics]:
    a = list(a)
    m = Metrics()

    def part(lo, hi):
        mid = (lo + hi)//2
        pivot = median3(a, lo, mid, hi)
        i, j = lo-1, hi+1
        while True:
            i += 1
            while a[i] < pivot:
                m.comps += 1
                i += 1
            m.comps += 1  # la comparación que rompe
            j -= 1
            while a[j] > pivot:
                m.comps += 1
                j -= 1
            m.comps += 1
            if i >= j:
                return j
            a[i], a[j] = a[j], a[i]
            m.moves += 1

    def qs(lo, hi):
        while lo < hi:
            if hi - lo + 1 <= cutoff:
                insertion_tail(a, lo, hi)
                return
            p = part(lo, hi)
            if p - lo < hi - (p + 1):
                qs(lo, p)
                lo = p + 1
            else:
                qs(p + 1, hi)
                hi = p
    if a:
        qs(0, len(a)-1)
        # "pulido" final en toda la lista por si quedó un subarray
        # pequeño sin ordenar
        insertion_tail(a, 0, len(a)-1)
    return a, m


# ---------- Quick 3-way (con comparaciones contadas grosso modo) ----------


def quick_3way_instr(a: List[int]) -> Tuple[List[int], Metrics]:
    import random
    a = list(a)
    m = Metrics()

    def qs(lo, hi):
        while lo < hi:
            p = random.randint(lo, hi)
            a[lo], a[p] = a[p], a[lo]
            m.moves += 1
            pivot = a[lo]
            lt, i, gt = lo, lo+1, hi
            while i <= gt:
                m.comps += 1
                if a[i] < pivot:
                    a[lt], a[i] = a[i], a[lt]
                    m.moves += 1
                    lt += 1
                    i += 1
                elif a[i] > pivot:
                    a[i], a[gt] = a[gt], a[i]
                    m.moves += 1
                    gt -= 1
                else:
                    i += 1
            if (lt - lo) < (hi - gt):
                qs(lo, lt-1)
                lo = gt + 1
            else:
                qs(gt + 1, hi)
                hi = lt - 1
    if a:
        qs(0, len(a)-1)
    return a, m


# ---------- Heap ----------


def heapsort_instr(a: List[int]) -> Tuple[List[int], Metrics]:
    a = list(a)
    m = Metrics()
    n = len(a)

    def sift(i, end):
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
            m.comps += 1
            if a[i] >= a[j]:
                break
            a[i], a[j] = a[j], a[i]
            m.moves += 1
            i = j
    for i in range((n-2)//2, -1, -1):
        sift(i, n)
    for end in range(n-1, 0, -1):
        a[0], a[end] = a[end], a[0]
        m.moves += 1
        sift(0, end)
    return a, m


# ---------- Banco y ejecución ----------


ALGOS: Dict[str, Callable[[List[int]], Tuple[List[int], Metrics]]] = {
    "insertion": insertion_sort_instr,
    "selection": selection_sort_instr,
    "bubble": bubble_sort_instr,
    "shell": shell_sort_instr,
    "merge": merge_sort_instr,
    "quick": quick_hoare_instr,
    "quick3": quick_3way_instr,
    "heap": heapsort_instr,
}


INPUTS = {
    "random": gen_random,
    "sorted": gen_sorted,
    "reverse": gen_reverse,
    "nearly": gen_nearly_sorted,
    "dups": gen_many_dups,
}


def run_bench(sizes=(200, 800, 3000), reps=3, algos=None, inputs=None,
              seed=123):
    if algos is None:
        algos = list(ALGOS.keys())
    if inputs is None:
        inputs = list(INPUTS.keys())
    rows = []
    for n in sizes:
        for dist in inputs:
            gen = INPUTS[dist]
            for r in range(reps):
                base = gen(n) if dist != "random" else gen(n, seed=seed + r)
                for name in algos:
                    arr = list(base)
                    fn = ALGOS[name]
                    t0 = time.perf_counter()
                    out, m = fn(arr)
                    t1 = time.perf_counter()
                    assert out == sorted(base), f"{name} falló en {dist} n={n}"
                    rows.append({
                        "algo": name, "n": n, "dist": dist, "rep": r,
                        "time_ms": (t1 - t0)*1000,
                        "comps": m.comps, "moves": m.moves
                    })
                    # Pequeña pausa para estabilizar ruido (opcional)
    return rows


if __name__ == "__main__":
    # Configuración de estilo para gráficos
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)

    # Ejecutar benchmark
    rows = run_bench(
        sizes=(200, 800, 3000),   # ajusta aquí tamaños
        reps=3,                   # repeticiones por punto
        algos=["insertion", "shell", "merge", "quick", "quick3", "heap"],
        inputs=["random", "sorted", "reverse", "nearly", "dups"]
    )

    # Crear DataFrame
    df = pd.DataFrame(rows)

    # Guardar CSV
    df.to_csv("sorting_results.csv", index=False, encoding="utf-8")
    print("✓ Resultados guardados en 'sorting_results.csv'")

    # Calcular estadísticas agrupadas
    df_stats = df.groupby(['algo', 'dist', 'n']).agg({
        'time_ms': ['mean', 'std'],
        'comps': ['mean', 'std'],
        'moves': ['mean', 'std']
    }).round(2)

    print("\n" + "="*80)
    print("RESUMEN ESTADÍSTICO")
    print("="*80)
    print(df_stats.to_string())

    # Resumen por algoritmo (promedio general)
    print("\n" + "="*80)
    print("RESUMEN POR ALGORITMO (promedio de todas las configuraciones)")
    print("="*80)
    algo_summary = df.groupby('algo').agg({
        'time_ms': 'mean',
        'comps': 'mean',
        'moves': 'mean'
    }).round(2).sort_values('time_ms')
    print(algo_summary.to_string())

    # Crear visualizaciones
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análisis de Algoritmos de Ordenación', fontsize=16, y=1.00)

    # 1. Tiempo por algoritmo y tamaño (promedio de todas las distribuciones)
    ax1 = axes[0, 0]
    df_time_size = df.groupby(['algo', 'n'])['time_ms'].mean().reset_index()
    for algo in df_time_size['algo'].unique():
        data = df_time_size[df_time_size['algo'] == algo]
        ax1.plot(data['n'], data['time_ms'], marker='o', label=algo,
                 linewidth=2)
    ax1.set_xlabel('Tamaño del array (n)')
    ax1.set_ylabel('Tiempo (ms)')
    ax1.set_title('Tiempo de ejecución vs Tamaño')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Comparaciones por algoritmo
    ax2 = axes[0, 1]
    df_comps = df.groupby('algo')['comps'].mean().sort_values()
    df_comps.plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_xlabel('Comparaciones (promedio)')
    ax2.set_title('Número de Comparaciones por Algoritmo')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Movimientos por algoritmo
    ax3 = axes[0, 2]
    df_moves = df.groupby('algo')['moves'].mean().sort_values()
    df_moves.plot(kind='barh', ax=ax3, color='coral')
    ax3.set_xlabel('Movimientos (promedio)')
    ax3.set_title('Número de Movimientos por Algoritmo')
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Heatmap: Tiempo por algoritmo y distribución (n=3000)
    ax4 = axes[1, 0]
    df_heatmap = df[df['n'] == 3000].groupby(['algo', 'dist'])[
        'time_ms'].mean().reset_index()
    df_pivot = df_heatmap.pivot(index='algo', columns='dist',
                                values='time_ms')
    sns.heatmap(df_pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax4,
                cbar_kws={'label': 'Tiempo (ms)'})
    ax4.set_title('Tiempo de ejecución (n=3000) por Distribución')
    ax4.set_xlabel('Distribución de datos')
    ax4.set_ylabel('Algoritmo')

    # 5. Comparación de tiempo por tipo de distribución
    ax5 = axes[1, 1]
    df_dist = df.groupby(['dist', 'algo'])['time_ms'].mean().reset_index()
    for dist in df_dist['dist'].unique():
        data = df_dist[df_dist['dist'] == dist]
        ax5.bar([f"{d[:3]}" for d in data['algo']], data['time_ms'],
                label=dist, alpha=0.7)
    ax5.set_xlabel('Algoritmo')
    ax5.set_ylabel('Tiempo (ms)')
    ax5.set_title('Tiempo por Distribución de Datos')
    ax5.legend(title='Distribución')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    # 6. Eficiencia: Comparaciones vs Tiempo (n=3000, random)
    ax6 = axes[1, 2]
    df_eff = df[(df['n'] == 3000) & (df['dist'] == 'random')].groupby(
        'algo').agg({'time_ms': 'mean', 'comps': 'mean'}).reset_index()
    ax6.scatter(df_eff['comps'], df_eff['time_ms'], s=200, alpha=0.6,
                c=range(len(df_eff)), cmap='viridis')
    for idx, row in df_eff.iterrows():
        ax6.annotate(row['algo'], (row['comps'], row['time_ms']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax6.set_xlabel('Comparaciones')
    ax6.set_ylabel('Tiempo (ms)')
    ax6.set_title('Eficiencia: Comparaciones vs Tiempo\n(n=3000, random)')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sorting_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráficos guardados en 'sorting_analysis.png'")
    plt.show()

    # Crear un segundo conjunto de gráficos: análisis detallado por tamaño
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('Complejidad Temporal por Tamaño de Entrada', fontsize=14)

    sizes = df['n'].unique()
    for idx, n in enumerate(sorted(sizes)):
        ax = axes2[idx]
        df_n = df[df['n'] == n].groupby(['algo', 'dist'])[
            'time_ms'].mean().reset_index()
        df_n_pivot = df_n.pivot(index='algo', columns='dist',
                                values='time_ms')
        df_n_pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'n = {n}')
        ax.set_xlabel('Algoritmo')
        ax.set_ylabel('Tiempo (ms)')
        ax.legend(title='Distribución', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('sorting_by_size.png', dpi=300, bbox_inches='tight')
    print("✓ Gráficos por tamaño guardados en 'sorting_by_size.png'")
    plt.show()

    print("\n" + "="*80)
    print("Análisis completado exitosamente!")
    print("="*80)
