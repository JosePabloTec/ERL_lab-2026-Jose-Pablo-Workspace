import numpy as np
import matplotlib.pyplot as plt
import random
import math

def colision_segmento(f_inicio, c_inicio, f_fin, c_fin, mapa_grid, radio_robot=0.35):

    dist = math.sqrt((f_fin - f_inicio)**2 + (c_fin - c_inicio)**2)
    pasos = int(dist / 0.1) + 1 
    
    for i in range(pasos + 1):
        t = i / pasos
        f_actual = f_inicio + t * (f_fin - f_inicio)
        c_actual = c_inicio + t * (c_fin - c_inicio)
        
        f_min = max(0, int(math.floor(f_actual - radio_robot)))
        f_max = min(mapa_grid.shape[0] - 1, int(math.ceil(f_actual + radio_robot)))
        c_min = max(0, int(math.floor(c_actual - radio_robot)))
        c_max = min(mapa_grid.shape[1] - 1, int(math.ceil(c_actual + radio_robot)))
        
        for f in range(f_min, f_max + 1):
            for c in range(c_min, c_max + 1):
                if mapa_grid[f, c] == 1: # Si hay un muro cerca
                    # Calculamos la distancia exacta del rayo al centro del muro
                    dist_obstaculo = math.sqrt((f_actual - f)**2 + (c_actual - c)**2)
                    
                    # El centro del cuadro al borde mide 0.5. Le sumamos el radio de nuestro robot.
                    if dist_obstaculo <= (0.5 + radio_robot):
                        return True # Colisión confirmada
                        
    return False

def calcular_ruta_rrt(mapa_grid, inicio, meta, iteraciones=3000, paso=1.0, animar=False):
    
    arbol = [{'f': float(inicio[0]), 'c': float(inicio[1]), 'padre': None}]
    llegamos = False
    nodo_final = None

    if animar:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(mapa_grid, cmap='Greys', origin='upper')
        ax.scatter(inicio[1], inicio[0], color='blue', s=50, zorder=5, label="Inicio")
        ax.scatter(meta[1], meta[0], color='green', s=50, zorder=5, label="Meta")
        ax.set_title("RRT Con Areas")

    for i in range(iteraciones):
        if random.random() < 0.1:
            rand_f, rand_c = meta[0], meta[1]
        else:
            rand_f = random.uniform(0, mapa_grid.shape[0] - 1)
            rand_c = random.uniform(0, mapa_grid.shape[1] - 1)

        idx_cercano = 0
        dist_min = float('inf')
        for idx, nodo in enumerate(arbol):
            dist = math.sqrt((nodo['f'] - rand_f)**2 + (nodo['c'] - rand_c)**2)
            if dist < dist_min:
                dist_min = dist
                idx_cercano = idx

        nodo_cercano = arbol[idx_cercano]

        theta = math.atan2(rand_f - nodo_cercano['f'], rand_c - nodo_cercano['c'])
        nuevo_f = nodo_cercano['f'] + paso * math.sin(theta)
        nuevo_c = nodo_cercano['c'] + paso * math.cos(theta)

        
        if not colision_segmento(nodo_cercano['f'], nodo_cercano['c'], nuevo_f, nuevo_c, mapa_grid, radio_robot=0.35):
            
            arbol.append({'f': nuevo_f, 'c': nuevo_c, 'padre': idx_cercano})
            idx_nuevo = len(arbol) - 1

            if animar:
                ax.plot([nodo_cercano['c'], nuevo_c], [nodo_cercano['f'], nuevo_f], 
                        color='pink', alpha=0.5, linewidth=1.5)
                if i % 10 == 0:
                    plt.pause(0.001)

            dist_meta = math.sqrt((nuevo_f - meta[0])**2 + (nuevo_c - meta[1])**2)
            if dist_meta <= paso:
                if not colision_segmento(nuevo_f, nuevo_c, meta[0], meta[1], mapa_grid, radio_robot=0.35):
                    llegamos = True
                    nodo_final = {'f': meta[0], 'c': meta[1], 'padre': idx_nuevo}
                    if animar:
                        ax.plot([nuevo_c, meta[1]], [nuevo_f, meta[0]], color='pink', alpha=0.5, linewidth=1.5)
                    break

    ruta = []
    if llegamos:
        nodo_actual = nodo_final
        while nodo_actual is not None:
            ruta.append((nodo_actual['f'], nodo_actual['c']))
            if nodo_actual['padre'] is not None:
                nodo_actual = arbol[nodo_actual['padre']]
            else:
                nodo_actual = None
        ruta = ruta[::-1]

        if animar:
            x_ruta = [coord[1] for coord in ruta]
            y_ruta = [coord[0] for coord in ruta]
            ax.plot(x_ruta, y_ruta, color='red', linewidth=3, label='Ruta RRT')
            ax.legend(loc="lower left")
            plt.ioff()
            plt.show()

        return arbol, ruta
    else:
        if animar:
            plt.ioff()
            plt.show()
        return arbol, []

if __name__ == "__main__":
    dimension = 50
    mapa_prueba = np.zeros((dimension, dimension))

    inicio_prueba = (2, 2)
    meta_prueba = (47, 47)
    cantidad_muros = 200 # Subí los muros para forzar cuellos de botella

    celdas_posibles = [(f, c) for f in range(dimension) for c in range(dimension)
                       if (f, c) != inicio_prueba and (f, c) != meta_prueba]
    
    muros_elegidos = random.sample(celdas_posibles, cantidad_muros)

    for f, c in muros_elegidos:
        mapa_prueba[f, c] = 1

    print("Iniciando algoritmo RRT con radio de colisión blindado...")
    arbol_explorado, ruta_optima = calcular_ruta_rrt(
        mapa_grid=mapa_prueba, 
        inicio=inicio_prueba, 
        meta=meta_prueba, 
        paso=0.8, 
        iteraciones=4000, 
        animar=True
    )

    if len(ruta_optima) > 0:
        print(f"¡Ruta encontrada con {len(ruta_optima)} nodos continuos!")
    else:
        print("El algoritmo se quedó sin iteraciones (el mapa generó un bloqueo total).")