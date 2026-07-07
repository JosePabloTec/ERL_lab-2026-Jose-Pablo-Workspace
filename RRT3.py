import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Nod:
    f: float
    c: float
    pad: Optional[int] = None

def colSeg(fIni: float, cIni: float, fFin: float, cFin: float, 
           mapGri: np.ndarray, radRob: float = 1.0) -> bool:
    """Valida colisiones del segmento"""
    dis = math.dist((fIni, cIni), (fFin, cFin))
    pas = int(dis / 0.1) + 1 
    
    filMap, colMap = mapGri.shape

    for i in range(pas + 1):
        t = i / pas if pas > 0 else 0
        fAct = fIni + t * (fFin - fIni)
        cAct = cIni + t * (cFin - cIni)
        
        fMin = max(0, int(math.floor(fAct - radRob)))
        fMax = min(filMap - 1, int(math.ceil(fAct + radRob)))
        cMin = max(0, int(math.floor(cAct - radRob)))
        cMax = min(colMap - 1, int(math.ceil(cAct + radRob)))
        
        for f in range(fMin, fMax + 1):
            for c in range(cMin, cMax + 1):
                if mapGri[f, c] == 1: 
                    disObs = math.dist((fAct, cAct), (f, c))
                    if disObs <= (0.5 + radRob):
                        return True 
                        
    return False 

def calRutRrt(mapGri: np.ndarray, ini: Tuple[float, float], met: Tuple[float, float], 
              ite: int = 20000, pas: float = 1.0, ani: bool = False) -> Tuple[List[Nod], List[Tuple[float, float]]]:
    """Calcula ruta usando algoritmo RRT"""
    arb = [Nod(f=float(ini[0]), c=float(ini[1]))]
    nodFinId: Optional[int] = None 
    filMap, colMap = mapGri.shape

    if ani:
        plt.ion() 
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(mapGri, cmap='Greys', origin='upper') 
        ax.scatter(ini[1], ini[0], color='blue', s=50, zorder=5, label="Ini") 
        ax.scatter(met[1], met[0], color='green', s=50, zorder=5, label="Met") 
        ax.set_title("RRT Plan")

    for i in range(ite):
        if i % 1000 == 0 and i > 0:
            print(f"Ite {i}/{ite}...")

        if random.random() < 0.1:
            rndF, rndC = met[0], met[1]
        else:
            rndF = random.uniform(0, filMap - 1)
            rndC = random.uniform(0, colMap - 1)

        # Buscar nodo más cercano
        lstDis = [math.dist((nod.f, nod.c), (rndF, rndC)) for nod in arb]
        idCer = lstDis.index(min(lstDis))
        nodCer = arb[idCer] 

        the = math.atan2(rndF - nodCer.f, rndC - nodCer.c)
        nueF = nodCer.f + pas * math.sin(the)
        nueC = nodCer.c + pas * math.cos(the)

        if not colSeg(nodCer.f, nodCer.c, nueF, nueC, mapGri, radRob=0.35):
            
            arb.append(Nod(f=nueF, c=nueC, pad=idCer))
            idNue = len(arb) - 1 

            if ani:
                ax.plot([nodCer.c, nueC], [nodCer.f, nueF], color='pink', alpha=0.5, linewidth=1.5)
                if i % 100 == 0:
                    plt.pause(0.001)

            disMet = math.dist((nueF, nueC), met)
            
            if disMet <= pas:
                if not colSeg(nueF, nueC, met[0], met[1], mapGri, radRob=0.35):
                    arb.append(Nod(f=met[0], c=met[1], pad=idNue))
                    nodFinId = len(arb) - 1
                    
                    if ani:
                        ax.plot([nueC, met[1]], [nueF, met[0]], color='pink', alpha=0.5, linewidth=1.5)
                    break 

    rut = []
    if nodFinId is not None:
        idAct = nodFinId
        while idAct is not None:
            nod = arb[idAct]
            rut.append((nod.f, nod.c))
            idAct = nod.pad
        
        rut = rut[::-1] 

        if ani:
            xRut = [cor[1] for cor in rut]
            yRut = [cor[0] for cor in rut]
            ax.plot(xRut, yRut, color='red', linewidth=3, label='Rut RRT')
            ax.legend(loc="lower left")
            plt.ioff() 
            plt.show() 

        return arb, rut 

    print(f"Lim ite alcanzado ({ite}). Sin rut.")
    if ani:
        plt.ioff()
        plt.show()
    return arb, []


if __name__ == "__main__":
    dim = 50 
    mapPru = np.zeros((dim, dim)) 

    iniPru = (2, 2)
    metPru = (47, 47)
    canMur = 500

    celPos = [(f, c) for f in range(dim) for c in range(dim)
              if (f, c) != iniPru and (f, c) != metPru]
    
    murEle = random.sample(celPos, canMur)

    for f, c in murEle:
        mapPru[f, c] = 1

    print("Ini RRT...")
    
    arbExp, rutOpt = calRutRrt(
        mapGri=mapPru, 
        ini=iniPru, 
        met=metPru, 
        pas=0.8,        
        ite=20000, 
        ani=True      
    )

    if rutOpt:
        print(f"Ruta OK: {len(rutOpt)} nodos.")
    else:
        print("Fin sin rut.")