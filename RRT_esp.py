import numpy as np # Manejo de matrices numéricas
import matplotlib.pyplot as plt # Visualización de gráficas y animación
import random # Generación de valores estocásticos
import math # Operaciones trigonométricas (seno, coseno, atan2)

# =====================================================================
# CLASE NODO: Estructura de datos para los vértices del árbol
# =====================================================================
class Nodo:
    def __init__(self, x, y, padre=None):
        self.x = x # Coordenada horizontal (Columna)
        self.y = y # Coordenada vertical (Fila)
        self.padre = padre # Puntero al nodo generador


# =====================================================================
# FUNCIÓN PRINCIPAL RRT
# =====================================================================
def rrt(grid, inicio, meta, max_iter=5000, paso=0.5):
    
    # Desempaquetamos las tuplas (fila=y, columna=x) para adaptarlas al plano cartesiano
    y_ini, x_ini = inicio
    y_meta, x_meta = meta
    
    # Inicializamos el árbol con el nodo raíz
    arbol = [Nodo(x_ini, y_ini)]
    
    ### Configuración del entorno de visualización (Animación)
    plt.ion() 
    fig, ax = plt.subplots(figsize=(7, 7)) 
    ax.imshow(grid, cmap='Greys', origin='upper') 
    ax.scatter(x_ini, y_ini, color='blue', s=100, label='Inicio', zorder=5) 
    ax.scatter(x_meta, y_meta, color='green', s=100, label='Meta', zorder=5) 
    ax.set_title("RRT (Nomenclatura Estándar)") 
    
    ### Ciclo de exploración principal
    for i in range(max_iter):
        
        # p = Variable estocástica para el Goal Bias (0.0 a 1.0)
        p = random.random() 
        
        # 10% de probabilidad: Muestreo directo a la meta
        if p < 0.1:
            x_rand, y_rand = x_meta, y_meta 
        # 90% de probabilidad: Muestreo uniforme en el espacio de configuración
        else: 
            y_rand = random.uniform(0, grid.shape[0] - 1) 
            x_rand = random.uniform(0, grid.shape[1] - 1) 
            
        # Búsqueda del vecino más cercano (Nearest Neighbor)
        nodo_cercano = arbol[0] 
        dist_min = float('inf') 
        
        for nodo in arbol: 
            # Distancia Euclidiana
            dist = math.sqrt((nodo.x - x_rand)**2 + (nodo.y - y_rand)**2)
            
            if dist < dist_min:
                dist_min = dist 
                nodo_cercano = nodo 

        # Cálculo de cinemática: Dirección y avance (Steer)       
        theta = math.atan2(y_rand - nodo_cercano.y, x_rand - nodo_cercano.x)
        x_nuevo = nodo_cercano.x + paso * math.cos(theta)
        y_nuevo = nodo_cercano.y + paso * math.sin(theta)
        
        # Validación de fronteras del mapa (Boundary Check)
        en_limites = (0 <= y_nuevo < grid.shape[0]) and (0 <= x_nuevo < grid.shape[1])
        
        if en_limites: 
            # tol = Tolerancia física del robot (Bounding Box)
            tol = 0.15 
            
            # Discretización espacial de las 4 esquinas del robot
            y_min = int(round(y_nuevo - tol))
            y_max = int(round(y_nuevo + tol))
            x_min = int(round(x_nuevo - tol))
            x_max = int(round(x_nuevo + tol))
            
            # Verificación de que la caja de colisión no exceda la matriz
            limites_seguros = (0 <= y_min) and (y_max < grid.shape[0]) and \
                              (0 <= x_min) and (x_max < grid.shape[1])
            
            if limites_seguros: 
                # Detección de colisiones: Suma de la grilla de ocupación
                colisiones = (
                    grid[y_min, x_min] + 
                    grid[y_min, x_max] + 
                    grid[y_max, x_min] + 
                    grid[y_max, x_max]
                )
                
                # Cero colisiones = Espacio libre (Free Space)
                if colisiones == 0: 
                    # Instanciación y guardado del nuevo vértice
                    nuevo_nodo = Nodo(x_nuevo, y_nuevo, padre=nodo_cercano)
                    arbol.append(nuevo_nodo) 
                    
                    ### Renderizado de la arista (Edge rendering)
                    ax.plot([nodo_cercano.x, x_nuevo], 
                            [nodo_cercano.y, y_nuevo], 
                            color='pink', alpha=0.5)
                    
                    # Tasa de refresco de la interfaz gráfica
                    if i % 5 == 0: 
                        plt.pause(0.001)
                    
                    # Verificación de condición de término (Goal Check)
                    dist_meta = math.sqrt((x_nuevo - x_meta)**2 + (y_nuevo - y_meta)**2)
                    
                    if dist_meta <= paso: 
                        # Inserción del nodo terminal
                        nodo_final = Nodo(x_meta, y_meta, padre=nuevo_nodo)
                        arbol.append(nodo_final) 
                        
                        ax.plot([x_nuevo, x_meta], 
                                [y_nuevo, y_meta], 
                                color='pink', alpha=0.5)
                        
                        # Reconstrucción de trayectoria (Path Backtracking)
                        path = [] 
                        nodo_actual = nodo_final 
                        
                        while nodo_actual is not None: 
                            # Se invierte de vuelta a formato matriz (fila, columna) para consistencia externa
                            path.append((nodo_actual.y, nodo_actual.x)) 
                            nodo_actual = nodo_actual.padre 
                            
                        path.reverse() 
                        
                        # Renderizado del Path resultante
                        x_path = [coord[1] for coord in path]
                        y_path = [coord[0] for coord in path]
                        ax.plot(x_path, y_path, color='red', linewidth=3, label='Path')
                        ax.legend() 
                        
                        plt.ioff()
                        plt.show()
                        
                        return arbol, path
    
    # Cierre por agotamiento de iteraciones (Time-out)
    plt.ioff()
    plt.show()
    return arbol, [] 


### Bloque de Ejecución (Main)
if __name__ == "__main__":
    dim = 20 # Dimensión de la cuadrícula
    grid_map = np.zeros((dim, dim)) # Inicialización del Free Space
    inicio = (0, 0) # Punto Start (y, x)
    meta = (19, 19) # Punto Goal (y, x)
    num_obs = 50 # Densidad de obstáculos

    celdas_libres = [] 
    for y in range(dim): 
        for x in range(dim): 
            if (y, x) != inicio and (y, x) != meta:
                celdas_libres.append((y, x)) 
    
    # Muestreo de obstáculos
    obs_random = random.sample(celdas_libres, num_obs)
    
    # Asignación de obstáculos en la cuadrícula
    for obs_y, obs_x in obs_random:
        grid_map[obs_y, obs_x] = 1

    print("Inicializando planeación de trayectorias (RRT)...")
    arbol_resultante, trayectoria = rrt(grid_map, inicio, meta)
    
    if len(trayectoria) > 0:
        print("Path Planning exitoso.")
    else: 
        print("Fallo en resolución: Espacio de configuración obstruido o límite de iteraciones excedido.")