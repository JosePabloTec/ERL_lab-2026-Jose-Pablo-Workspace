#!pip install pqdict

import numpy as np
import matplotlib.pyplot as plt
import random
from pqdict import pqdict

# Algoritmo A* para encontrar la ruta más corta
def calcular_ruta_a_star(mapa_grid, inicio, meta):
    def heuristica(nodo_actual, meta):
        return abs(nodo_actual[0] - meta[0]) + abs(nodo_actual[1] - meta[1])

    open_list = pqdict()
    nodos = {}
    closed_list = set()
    movimientos = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    nodos[inicio] = {'g': 0, 'padre': None}
    open_list[inicio] = heuristica(inicio, meta)
    llegamos = False

    while len(open_list) > 0:
        actual, costo_f = open_list.popitem()

        if actual == meta:
            llegamos = True
            break

        closed_list.add(actual)

        for mov in movimientos:
            vecino = (actual[0] + mov[0], actual[1] + mov[1])

            if 0 <= vecino[0] < len(mapa_grid) and 0 <= vecino[1] < len(mapa_grid[0]):
                if mapa_grid[vecino[0], vecino[1]] == 1 or vecino in closed_list:
                    continue

                nuevo_costo_g = nodos[actual]['g'] + 1

                if vecino not in nodos or nuevo_costo_g < nodos[vecino]['g']:
                    nodos[vecino] = {'g': nuevo_costo_g, 'padre': actual}
                    costo_f_vecino = nuevo_costo_g + heuristica(vecino, meta)
                    open_list[vecino] = costo_f_vecino

    # Reconstrucción de la ruta encontrada
    if llegamos:
        ruta = []
        nodo_actual = meta

        while nodo_actual is not None:
            ruta.append(nodo_actual)
            nodo_actual = nodos[nodo_actual]['padre']

        return ruta[::-1]
    else:
        return []

# Generación del mapa aleatorio
dimension = 15
mapa_aleatorio = np.zeros((dimension, dimension))

inicio = (0, 0)
meta = (dimension - 1, dimension - 1)

cantidad_muros = 45

# Se eligen posiciones aleatorias para colocar los obstáculos
celdas_posibles = [(f, c) for f in range(dimension) for c in range(dimension)
                   if (f, c) != inicio and (f, c) != meta]

muros_elegidos = random.sample(celdas_posibles, cantidad_muros)

for f, c in muros_elegidos:
    mapa_aleatorio[f, c] = 1

# Ejecutar A* y mostrar el resultado
ruta_optima = calcular_ruta_a_star(mapa_aleatorio, inicio, meta)

plt.figure(figsize=(7, 7))
plt.imshow(mapa_aleatorio, cmap='Greys', origin='upper')

plt.scatter(inicio[1], inicio[0], color='blue', s=150, label='Inicio', zorder=5)
plt.scatter(meta[1], meta[0], color='green', s=150, label='Meta', zorder=5)

if len(ruta_optima) > 0:
    print(f"¡Ruta encontrada! Longitud: {len(ruta_optima)} pasos.")

    x_ruta = [coord[1] for coord in ruta_optima]
    y_ruta = [coord[0] for coord in ruta_optima]

    # Dibujar la ruta óptima encontrada
    plt.plot(x_ruta, y_ruta, color='red', linewidth=3, label='Ruta A*')
else:
    print("El algoritmo determinó que no hay camino posible.")

plt.title("Prueba de Estrés: A* en Laberinto Aleatorio")
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()