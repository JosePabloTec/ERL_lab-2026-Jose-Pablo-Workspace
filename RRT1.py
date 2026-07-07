import numpy as np
import matplotlib.pyplot as plt
import random
import math

class Nodo:
    def __init__(self, fila, columna, nodo_padre=None):
        self.fila = fila
        self.columna = columna
        self.nodo_padre = nodo_padre

def rrt(mapa_cuadricula, coordenada_inicio, coordenada_meta, limite_iteraciones=5000, tamano_paso=0.7):
    
    arbol_exploracion = [Nodo(coordenada_inicio[0], coordenada_inicio[1])]
    
    for iteracion in range(limite_iteraciones):
        
        probabilidad_hacia_meta = random.random()
        
        if probabilidad_hacia_meta < 0.1:
            fila_aleatoria = coordenada_meta[0]
            columna_aleatoria = coordenada_meta[1]
        else:
            fila_aleatoria = random.uniform(0, len(mapa_cuadricula) - 1)
            columna_aleatoria = random.uniform(0, len(mapa_cuadricula[0]) - 1)
            
        nodo_mas_cercano = arbol_exploracion[0]
        distancia_minima = float('inf')
        
        for nodo_actual in arbol_exploracion:
            distancia_calculada = math.sqrt((nodo_actual.fila - fila_aleatoria)**2 + (nodo_actual.columna - columna_aleatoria)**2)
            
            if distancia_calculada < distancia_minima:
                distancia_minima = distancia_calculada
                nodo_mas_cercano = nodo_actual
                
        angulo_direccion = math.atan2(fila_aleatoria - nodo_mas_cercano.fila, columna_aleatoria - nodo_mas_cercano.columna)
        nueva_fila = nodo_mas_cercano.fila + tamano_paso * math.sin(angulo_direccion)
        nueva_columna = nodo_mas_cercano.columna + tamano_paso * math.cos(angulo_direccion)
        
        dentro_del_mapa = (0 <= nueva_fila < len(mapa_cuadricula)) and (0 <= nueva_columna < len(mapa_cuadricula[0]))
        
        if dentro_del_mapa:
            fila_redondeada = int(round(nueva_fila))
            columna_redondeada = int(round(nueva_columna))
            
            if mapa_cuadricula[fila_redondeada, columna_redondeada] == 0:
                nuevo_nodo = Nodo(nueva_fila, nueva_columna, nodo_padre=nodo_mas_cercano)
                arbol_exploracion.append(nuevo_nodo)
                
                distancia_a_meta = math.sqrt((nueva_fila - coordenada_meta[0])**2 + (nueva_columna - coordenada_meta[1])**2)
                
                if distancia_a_meta <= tamano_paso:
                    nodo_final_meta = Nodo(coordenada_meta[0], coordenada_meta[1], nodo_padre=nuevo_nodo)
                    arbol_exploracion.append(nodo_final_meta)
                    
                    ruta_final = []
                    nodo_rastreo = nodo_final_meta
                    
                    while nodo_rastreo is not None:
                        ruta_final.append((nodo_rastreo.fila, nodo_rastreo.columna))
                        nodo_rastreo = nodo_rastreo.nodo_padre
                        
                    ruta_final.reverse()
                    return arbol_exploracion, ruta_final
                    
    return arbol_exploracion, [] 


if __name__ == "__main__":
    dimension_mapa = 10
    mapa_prueba = np.zeros((dimension_mapa, dimension_mapa))
    coordenada_inicio = (0, 0)
    coordenada_meta = (9, 9)
    cantidad_obstaculos = 10

    celdas_disponibles = []
    for fila in range(dimension_mapa):
        for columna in range(dimension_mapa):
            if (fila, columna) != coordenada_inicio and (fila, columna) != coordenada_meta:
                celdas_disponibles.append((fila, columna))
                
    muros_elegidos = random.sample(celdas_disponibles, cantidad_obstaculos)
    
    for fila_muro, columna_muro in muros_elegidos:
        mapa_prueba[fila_muro, columna_muro] = 1


    arbol_final, ruta_final = rrt(mapa_prueba, coordenada_inicio, coordenada_meta)

    plt.figure(figsize=(7, 7))
    plt.imshow(mapa_prueba, cmap='Greys', origin='upper')

    for nodo_actual in arbol_final:
        if nodo_actual.nodo_padre is not None:
            plt.plot([nodo_actual.nodo_padre.columna, nodo_actual.columna], 
                     [nodo_actual.nodo_padre.fila, nodo_actual.fila], 
                     color='pink', alpha=0.5)

    if ruta_final:
        columnas_ruta = [coordenada[1] for coordenada in ruta_final]
        filas_ruta = [coordenada[0] for coordenada in ruta_final]
        plt.plot(columnas_ruta, filas_ruta, color='red', linewidth=3)

    plt.scatter(coordenada_inicio[1], coordenada_inicio[0], color='blue', s=100, label='Inicio')
    plt.scatter(coordenada_meta[1], coordenada_meta[0], color='green', s=100, label='Meta')
    plt.title("RRT")
    plt.legend()
    plt.show()