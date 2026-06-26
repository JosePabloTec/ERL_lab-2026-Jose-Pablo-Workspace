import os
import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# PYBULLET SETUP
# -------------------------

p.connect(p.GUI)

p.resetSimulation()

p.setGravity(0,0,-10)

p.setRealTimeSimulation(1)


p.resetDebugVisualizerCamera(
    cameraDistance=15,
    cameraYaw=0,
    cameraPitch=-55,
    cameraTargetPosition=[0,0,0]
)


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



# -------------------------
# MOTION
# -------------------------

linear_speed = 2
angular_speed = 1



# -------------------------
# OBSTACLES
# -------------------------

def create_obstacle(size,x,y):

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            size,
            size,
            size
        ]
    )


    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            size,
            size,
            size
        ],
        rgbaColor=[
            1,
            0,
            0,
            1
        ]
    )


    p.createMultiBody(

        baseMass=0,

        baseCollisionShapeIndex=collision,

        baseVisualShapeIndex=visual,

        basePosition=[
            x,
            y,
            size
        ]

    )


create_obstacle(1,5,5)
create_obstacle(1,-5,5)
create_obstacle(1,8,-3)
create_obstacle(1,-8,-4)
create_obstacle(0.5,3,-6)
create_obstacle(1.5,-2,7)



# -------------------------
# LIDAR
# -------------------------

NUM_RAYS = 360
LIDAR_RANGE = 15



def simulate_lidar(pos,orn):

    yaw = p.getEulerFromQuaternion(orn)[2]


    ray_from=[]
    ray_to=[]


    for i in range(NUM_RAYS):

        angle = yaw + (
            -np.pi +
            2*np.pi*i/NUM_RAYS
        )


        start=[

            pos[0],
            pos[1],
            pos[2]+0.3

        ]


        end=[

            pos[0]+LIDAR_RANGE*np.cos(angle),

            pos[1]+LIDAR_RANGE*np.sin(angle),

            pos[2]+0.3

        ]


        ray_from.append(start)
        ray_to.append(end)



    results=p.rayTestBatch(
        ray_from,
        ray_to
    )


    points=[]


    for r in results:

        if r[0]!=-1:

            points.append(
                [
                    r[3][0],
                    r[3][1]
                ]
            )


    return points,ray_from,ray_to,results





# -------------------------
# GRID MAP
# -------------------------

resolution=0.5

GRID_SIZE=60


grid=np.ones(
    (GRID_SIZE,GRID_SIZE)
)*0.5



def world_to_grid(x,y):

    return (

        int(x/resolution+GRID_SIZE/2),

        int(y/resolution+GRID_SIZE/2)

    )



def bresenham(x0,y0,x1,y1):

    points=[]

    dx=abs(x1-x0)
    dy=abs(y1-y0)


    sx=1 if x0<x1 else -1
    sy=1 if y0<y1 else -1


    err=dx-dy


    while True:

        points.append(
            (x0,y0)
        )

        if x0==x1 and y0==y1:
            break


        e2=2*err


        if e2>-dy:

            err-=dy
            x0+=sx


        if e2<dx:

            err+=dx
            y0+=sy


    return points




def update_grid(
        rays_start,
        rays_end,
        results):


    for s,e,r in zip(
        rays_start,
        rays_end,
        results
    ):


        sx,sy=world_to_grid(
            s[0],
            s[1]
        )


        if r[0]!=-1:

            hit=r[3]

            ex,ey=world_to_grid(
                hit[0],
                hit[1]
            )


        else:

            ex,ey=world_to_grid(
                e[0],
                e[1]
            )


        cells=bresenham(
            sx,
            sy,
            ex,
            ey
        )


        for c in cells[:-1]:

            x,y=c

            if 0<=x<GRID_SIZE and 0<=y<GRID_SIZE:

                grid[x,y]=0



        if r[0]!=-1:

            x,y=cells[-1]

            if 0<=x<GRID_SIZE and 0<=y<GRID_SIZE:

                grid[x,y]=1




# -------------------------
# PLOT
# -------------------------

plt.ion()

fig,ax=plt.subplots()



# -------------------------
# LOOP
# -------------------------

while True:


    pos,orn=p.getBasePositionAndOrientation(car)


    rotation=p.getMatrixFromQuaternion(orn)


    forward=[

        rotation[0],
        rotation[3],
        rotation[6]

    ]


    p.resetBaseVelocity(

        car,

        linearVelocity=[

            forward[0]*linear_speed,

            forward[1]*linear_speed,

            0

        ],


        angularVelocity=[

            0,
            0,
            angular_speed

        ]

    )



    lidar,rays1,rays2,results = simulate_lidar(
        pos,
        orn
    )


    update_grid(
        rays1,
        rays2,
        results
    )



    # -------------------------
    # DRAW GRID
    # -------------------------

    ax.clear()


    ax.imshow(

        grid.T,

        origin="lower",

        cmap="gray",

        vmin=0,

        vmax=1

    )


    # lidar points

    if len(lidar)>0:

        pts=np.array(lidar)

        gx=[]
        gy=[]


        for x,y in pts:

            ix,iy=world_to_grid(x,y)

            gx.append(ix)
            gy.append(iy)



        ax.scatter(

            gx,
            gy,

            c="black",

            s=15

        )



    # robot

    rx,ry=world_to_grid(
        pos[0],
        pos[1]
    )


    ax.scatter(

        rx,
        ry,

        c="red",

        s=80

    )



    ax.set_title(
        "Racecar LiDAR Occupancy Grid"
    )


    plt.pause(0.01)

    time.sleep(0.01)