import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import pybullet as p
import pybullet_data
import time

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



# --------------------------
# GAUSSIAN GRID MAP
# --------------------------

EXTEND_AREA = 10.0

xyreso = 0.5
STD = 5.0



def calc_grid_map_config(ox, oy, xyreso):

    minx = round(min(ox) - EXTEND_AREA/2.0)
    miny = round(min(oy) - EXTEND_AREA/2.0)

    maxx = round(max(ox) + EXTEND_AREA/2.0)
    maxy = round(max(oy) + EXTEND_AREA/2.0)


    xw = int(round((maxx-minx)/xyreso))
    yw = int(round((maxy-miny)/xyreso))


    return minx,miny,maxx,maxy,xw,yw




def generate_gaussian_grid_map(ox, oy, xyreso, std):


    minx,miny,maxx,maxy,xw,yw = calc_grid_map_config(
        ox,oy,xyreso
    )


    gmap = [
        [0.0 for i in range(yw)]
        for i in range(xw)
    ]


    for ix in range(xw):

        for iy in range(yw):


            x = ix*xyreso + minx
            y = iy*xyreso + miny


            mindis = float("inf")


            for iox,ioy in zip(ox,oy):

                d = math.hypot(
                    iox-x,
                    ioy-y
                )

                if d < mindis:
                    mindis=d



            pdf = 1.0 - norm.cdf(
                mindis,
                0.0,
                std
            )


            gmap[ix][iy]=pdf



    return gmap,minx,maxx,miny,maxy




def draw_heatmap(data,minx,maxx,miny,maxy,xyreso):

    x,y=np.mgrid[
        slice(minx-xyreso/2,
              maxx+xyreso/2,
              xyreso),

        slice(miny-xyreso/2,
              maxy+xyreso/2,
              xyreso)
    ]


    plt.pcolor(
        x,
        y,
        data,
        vmax=1.0,
        cmap=plt.cm.Blues
    )


    plt.axis("equal")





# --------------------------
# PYBULLET
# SAME CAMERA
# SAME MOTION
# --------------------------

cid = p.connect(p.SHARED_MEMORY)

if cid < 0:
    p.connect(p.GUI)



p.resetDebugVisualizerCamera(
    cameraDistance=15,
    cameraYaw=-0,
    cameraPitch=-55,
    cameraTargetPosition=[0,0,0]
)


p.resetSimulation()

p.setGravity(0,0,-10)

p.setRealTimeSimulation(1)



p.loadSDF(
    os.path.join(
        pybullet_data.getDataPath(),
        "stadium.sdf"
    )
)



car = p.loadURDF(
    os.path.join(
        pybullet_data.getDataPath(),
        "racecar/racecar.urdf"
    )
)




# --------------------------
# VELOCITY
# SAME
# --------------------------

linear_speed = 2
angular_speed = 1




# --------------------------
# MAPPING MEMORY
# --------------------------

ox=[]
oy=[]



plt.ion()




# --------------------------
# LOOP
# --------------------------

while True:


    # CAR POSE

    pos,orn = p.getBasePositionAndOrientation(car)


    x=pos[0]
    y=pos[1]



    # store observations

    ox.append(x)
    oy.append(y)



    # ----------------------
    # SAME MOTION
    # ----------------------

    rotation=p.getMatrixFromQuaternion(orn)


    forward=[
        rotation[0],
        rotation[3],
        rotation[6]
    ]


    linear_velocity=[
        forward[0]*linear_speed,
        forward[1]*linear_speed,
        0
    ]



    p.resetBaseVelocity(
        car,
        linearVelocity=linear_velocity,
        angularVelocity=[
            0,
            0,
            angular_speed
        ]
    )



    # ----------------------
    # UPDATE GAUSSIAN MAP
    # ----------------------

    gmap,minx,maxx,miny,maxy = generate_gaussian_grid_map(
        ox,
        oy,
        xyreso,
        STD
    )



    plt.cla()


    draw_heatmap(
        gmap,
        minx,
        maxx,
        miny,
        maxy,
        xyreso
    )


    plt.plot(
        ox,
        oy,
        "xr"
    )


    plt.plot(
        x,
        y,
        "ob"
    )


    plt.title("RaceCar Gaussian Grid Map")

    plt.pause(0.01)



    time.sleep(0.01)