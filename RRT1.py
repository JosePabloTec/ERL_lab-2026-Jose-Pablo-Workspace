import numpy as np
import matplotlib.pyplot as plt
import random
import math

class Nodo:
    def __init__(self, fila, columna, nodo_padre=None):
        self.fila = fila
        self.columna = columna
        self.nodo_padre = nodo_padre

def rrt(mapa_cuadricula, coordenada_inicio, coordenada_meta, limite_iteraciones=5000, tamano_paso=0.5):
    
    arbol_exploracion = [Nodo(coordenada_inicio[0], coordenada_inicio[1])]
    
    
    plt.ion()
    figura, ejes = plt.subplots(figsize=(7, 7))
    ejes.imshow(mapa_cuadricula, cmap='Greys', origin='upper')
    ejes.scatter(coordenada_inicio[1], coordenada_inicio[0], color='blue', s=100, label='Inicio', zorder=5)
    ejes.scatter(coordenada_meta[1], coordenada_meta[0], color='green', s=100, label='Meta', zorder=5)
    ejes.set_title("Animación RRT (Con tolerancia de obstáculos)")
    
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
            tolerancia = 0.15
            
            fila_arriba = int(round(nueva_fila - tolerancia))
            fila_abajo = int(round(nueva_fila + tolerancia))
            columna_izq = int(round(nueva_columna - tolerancia))
            columna_der = int(round(nueva_columna + tolerancia))
            
            limites_seguros = (0 <= fila_arriba) and (fila_abajo < len(mapa_cuadricula)) and \
                              (0 <= columna_izq) and (columna_der < len(mapa_cuadricula[0]))
            
            if limites_seguros:
                choques = (
                    mapa_cuadricula[fila_arriba, columna_izq] + 
                    mapa_cuadricula[fila_arriba, columna_der] + 
                    mapa_cuadricula[fila_abajo, columna_izq] + 
                    mapa_cuadricula[fila_abajo, columna_der]
                )
                
                if choques == 0:
                    nuevo_nodo = Nodo(nueva_fila, nueva_columna, nodo_padre=nodo_mas_cercano)
                    arbol_exploracion.append(nuevo_nodo)
                    
                    
                    ejes.plot([nodo_mas_cercano.columna, nueva_columna], 
                              [nodo_mas_cercano.fila, nueva_fila], 
                              color='pink', alpha=0.5)
                    
                    
                    if iteracion % 5 == 0:
                        plt.pause(0.001)
                    
                    distancia_a_meta = math.sqrt((nueva_fila - coordenada_meta[0])**2 + (nueva_columna - coordenada_meta[1])**2)
                    
                    if distancia_a_meta <= tamano_paso:
                        nodo_final_meta = Nodo(coordenada_meta[0], coordenada_meta[1], nodo_padre=nuevo_nodo)
                        arbol_exploracion.append(nodo_final_meta)
                        
                        
                        ejes.plot([nueva_columna, coordenada_meta[1]], 
                                  [nueva_fila, coordenada_meta[0]], 
                                  color='pink', alpha=0.5)
                        
                        ruta_final = []
                        nodo_rastreo = nodo_final_meta
                        
                        while nodo_rastreo is not None:
                            ruta_final.append((nodo_rastreo.fila, nodo_rastreo.columna))
                            nodo_rastreo = nodo_rastreo.nodo_padre
                            
                        ruta_final.reverse()
                        
                        
                        columnas_ruta = [coordenada[1] for coordenada in ruta_final]
                        filas_ruta = [coordenada[0] for coordenada in ruta_final]
                        ejes.plot(columnas_ruta, filas_ruta, color='red', linewidth=3, label='Ruta Final')
                        ejes.legend()
                        
                        plt.ioff()
                        plt.show()
                        
                        return arbol_exploracion, ruta_final
                    
    plt.ioff()
    plt.show()
    return arbol_exploracion, [] 


if __name__ == "__main__":
    dimension_mapa = 20
    mapa_prueba = np.zeros((dimension_mapa, dimension_mapa))
    coordenada_inicio = (0, 0)
    coordenada_meta = (19, 19)
    cantidad_obstaculos = 50

    celdas_disponibles = []
    for fila in range(dimension_mapa):
        for columna in range(dimension_mapa):
            if (fila, columna) != coordenada_inicio and (fila, columna) != coordenada_meta:
                celdas_disponibles.append((fila, columna))
                
    muros_elegidos = random.sample(celdas_disponibles, cantidad_obstaculos)
    
    for fila_muro, columna_muro in muros_elegidos:
        mapa_prueba[fila_muro, columna_muro] = 1

    print("Iniciando algoritmo RRT")
    arbol_final, ruta_final = rrt(mapa_prueba, coordenada_inicio, coordenada_meta)
    
    if len(ruta_final) > 0:
        print("Ruta encontrada con éxito")
    else:
        print("No se encontró una ruta posible en las iteraciones dadas.")