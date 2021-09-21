import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import os
import math

#plt.rcParams.update({'font.size': 16})

class Animate():
    def __init__(self,Masses,frames,title,snapshots,trail):
        self.patches = []
        self.circlePatches =[]
        self.Crosspatches = []
        self.pathPatches = []
        self.path = []
        self.frames = frames
        self.m1 = Masses[0]
        self.m2 = Masses[1]
        self.m3 = Masses[2]
        self.time_text = ""
        self.masses = Masses
        self.xlim = 2
        self.ylim = 2
        self.radius = 1/14
        self.time = 0
        self.mcolour = ['g','r','b']
        self.title = title
        if snapshots != None:
            self.snapshots = snapshots
        self.trail = trail

    def animate(self,i):
        # Everytime a new frame is drawn this function is called
        # For every patch on the figure it gets centered at its next location, with index i
        for x in range(3):
            self.circlePatches[x].center = (self.masses[x].rxs[i],self.masses[x].rys[i])
        #self.ax.clear()
        scale = 4
        for x in range(len(self.masses)):
            patch = plt.arrow(x=self.masses[x].rxs[i],y=self.masses[x].rys[i],dx=self.masses[x].vxs[i]/scale,dy=self.masses[x].vys[i]/scale,color=self.mcolour[x],
            animated=True,head_width=0.2, head_length=0.1,width=0.02)
            self.patches.append(patch)
            self.ax.add_patch(patch)
            self.patches.pop(0)


        if self.trail:
            if (self.time % 200 == 0):
                for x in range(len(self.masses)):
                    traj = plt.Circle((self.masses[x].rxs[self.time],self.masses[x].rys[self.time]),self.radius/2,color=self.mcolour[x],label="Body 1",animated=True)
                    self.pathPatches.append(traj)
                    self.ax.add_patch(traj)


        self.time_text.set_text('Frame = %.1f' % (self.time))
        self.time += 1
        """
        for x in range(len(self.snapshots)):
            if(self.snapshots[x] == self.time):
                self.fig.savefig('./'+self.title+"frame"+str(self.time)+'.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
        """

        return tuple(self.patches) + (self.time_text,) + tuple(self.Crosspatches) + tuple(self.circlePatches) + tuple(self.pathPatches)

    def init(self):
        return tuple(self.patches)  + (self.time_text,) + tuple(self.Crosspatches) + tuple(self.circlePatches) + tuple(self.pathPatches)

    def run(self):
        # Function called by the user, creates the figure and axes
        self.fig = plt.figure(figsize=(6,4),dpi=150,facecolor='w', edgecolor='k')
        self.ax = plt.axes()
        self.patches = []
        self.circlePatches = []
        #self.radius = 1/20
        radiusStart = 1/20
        # Bodies
        self.Patch1 = plt.Circle((self.m1.rxs[0],self.m1.rys[0]),self.radius,color=self.mcolour[0],label="Body 1",animated=True)
        self.circlePatches.append(self.Patch1)
        self.Patch2 = plt.Circle((self.m2.rxs[0],self.m2.rys[0]),self.radius,color=self.mcolour[1],label="Body 2",animated=True)
        self.circlePatches.append(self.Patch2)
        self.Patch3 = plt.Circle((self.m3.rxs[0],self.m3.rys[0]),self.radius,color=self.mcolour[2],label="Body 3",animated=True)
        self.circlePatches.append(self.Patch3)

        #Starting Position
        self.Patch11 = plt.Circle((self.m1.rxs[0],self.m1.rys[0]),radiusStart,color=self.mcolour[0],animated=True)
        self.circlePatches.append(self.Patch11)
        self.Patch22 = plt.Circle((self.m2.rxs[0],self.m2.rys[0]),radiusStart,color=self.mcolour[1],animated=True)
        self.circlePatches.append(self.Patch22)
        self.Patch33 = plt.Circle((self.m3.rxs[0],self.m3.rys[0]),radiusStart,color=self.mcolour[2],animated=True)
        self.circlePatches.append(self.Patch33)

        # Velocity Arrows
        self.mass1Patch = plt.arrow(x=self.m1.rxs[0],y=self.m1.rys[0],dx=5,dy=5,color=self.mcolour[0],animated=True)
        self.patches.append(self.mass1Patch)
        self.mass2Patch = plt.arrow(x=self.m2.rxs[0],y=self.m2.rys[0],dx=5,dy=5,color=self.mcolour[1],animated=True)
        self.patches.append(self.mass2Patch)
        self.mass3Patch = plt.arrow(x=self.m3.rxs[0],y=self.m3.rys[0],dx=5,dy=5,color=self.mcolour[2],animated=True)
        self.patches.append(self.mass3Patch)

        # Crosshair
        length = 1
        self.cross1Patch = plt.arrow(x=-length/2,y=0,dx=length,dy=0,color='black',animated=True)
        self.Crosspatches.append(self.cross1Patch)
        self.cross2Patch = plt.arrow(x=0,y=-length/2,dx=0,dy=length,color='black',animated=True)
        self.Crosspatches.append(self.cross2Patch)

        #pathPatches



        for x in range (len(self.Crosspatches)):
            self.ax.add_patch(self.Crosspatches[x])
        for x in range (len(self.patches)):
            self.ax.add_patch(self.patches[x])
        for x in range (len(self.circlePatches)):
            self.ax.add_patch(self.circlePatches[x])

        if self.trail:
            for x in range (len(self.pathPatches)):
                self.ax.add_patch(self.pathPatches[x])

        self.ax.axis('scaled')
        self.ax.set_xlim(-self.xlim, self.xlim)
        self.ax.set_ylim(-self.ylim, self.ylim)
        self.ax.set_title("Simulation of Orbits",fontsize='x-small')
        self.ax.legend(loc=4,fontsize='small',shadow=True)
        self.time_text = self.ax.text(-self.xlim,-self.ylim,s='frame = ',fontsize=16)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title(self.title)
        anim = FuncAnimation(self.fig, self.animate, init_func = self.init, frames = self.frames, repeat = False, interval = 1, blit = True)
        plt.show()
