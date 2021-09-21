import math as m
import numpy as np
import matplotlib.pyplot as plt
from Animator import Animate
from scipy.integrate import odeint
from odeintw import odeintw
import scipy as sp

plt.rcParams.update({'font.size': 14})

class Body():
    #Class to store position and velocity data to send to animator
    def __init__(self,id,r0,v0,mass):
        self.id = id
        self.rxs = []
        self.rys = []
        self.vxs = []
        self.vys = []
        self.mass = mass

    def setPositions(self,rx,ry,vx,vy):
        self.rxs = rx
        self.rys = ry
        self.vxs = vx
        self.vys = vy

G = 1
def Leapfrog(f , y , t , h , w0 , Masses):
    # Leapfrog Iterative Method in 2D
    rs = y[:6]
    vs = y[6:]

    hh = h/2.0
    r1 = rs + hh* np.asarray(f(0,rs,vs,t, w0, Masses))
    v1 = vs + h* np.asarray(f(1,r1,vs,t+hh ,w0, Masses))
    r1 = r1 + hh* np.asarray(f(0,rs,v1,t+h , w0, Masses))
    y = [r1[0],r1[1],r1[2],r1[3],r1[4],r1[5],v1[0],v1[1],v1[2],v1[3],v1[4],v1[5]]
    return y

def rk4(f,x,t,h,w0,Masses):
    # RK4 method
    k_1 = h * np.asarray(f(t,x,w0,Masses))
    k_2 = h * np.asarray(f((t+(h/2)),(x+(k_1/2)),w0,Masses))
    k_3 = h * np.asarray(f((t + (h/2)),(x+(k_2/2)),w0,Masses))
    k_4 = h * np.asarray(f(t + h, x + k_3,w0,Masses))
    x_next = x + (1/6)*(k_1 + 2 * k_2 + 2 * k_3 + k_4)
    return x_next

def ODE(a,rs,vs,t,w0, Masses):
    # ODE solver to be used for Leapfrog method
        dy = np.zeros(len(rs)+len(vs))
        rx1 = rs[0]
        ry1 = rs[1]
        rx2 = rs[2]
        ry2 = rs[3]
        rx3 = rs[4]
        ry3 = rs[5]
        r1 = [rx1,ry1]
        r2 = [rx2,ry2]
        r3 = [rx3,ry3]
        rs = [r1,r2,r3]
        dy[0] = vs[0]
        dy[1] = vs[1]
        dy[2] = vs[2]
        dy[3] = vs[3]
        dy[4] = vs[4]
        dy[5] = vs[5]
        dy[6] = 0
        dy[7] = 0
        dy[8] = 0
        dy[9] = 0
        dy[10] = 0
        dy[11] = 0

        r12 , r12_bold = findDist(rs[0],rs[1])
        r13 , r13_bold = findDist(rs[0],rs[2])
        r23 , r23_bold = findDist(rs[1],rs[2])
        r31 , r31_bold = findDist(rs[2],rs[0])
        r32 , r32_bold = findDist(rs[2],rs[1])
        r21 , r21_bold = findDist(rs[1],rs[0])
        # Force on body 1
        dy[6] -= G*Masses[1].mass*r12_bold[0]*r12**(-3)
        dy[7] -= G*Masses[1].mass*r12_bold[1]*r12**(-3)
        dy[6] -= G*Masses[2].mass*r13_bold[0]*r13**(-3)
        dy[7] -= G*Masses[2].mass*r13_bold[1]*r13**(-3)
        # Force on body 2
        dy[8] -= G*Masses[0].mass*r21_bold[0]*r21**(-3)
        dy[9] -= G*Masses[0].mass*r21_bold[1]*r21**(-3)
        dy[8] -= G*Masses[2].mass*r23_bold[0]*r23**(-3)
        dy[9] -= G*Masses[2].mass*r23_bold[1]*r23**(-3)
        # Force on body 3
        dy[10] -= G*Masses[0].mass*r31_bold[0]*r31**(-3)
        dy[11] -= G*Masses[0].mass*r31_bold[1]*r31**(-3)
        dy[10] -= G*Masses[1].mass*r32_bold[0]*r32**(-3)
        dy[11] -= G*Masses[1].mass*r32_bold[1]*r32**(-3)
        #print(dy)
        if a == 1:
            return dy[6:]
        else:
            return dy[:6]

def findDist(r1,r2):
    # Finds the distance between two masses and the vector too
    r1x = r1[0]
    r1y = r1[1]
    r2x = r2[0]
    r2y = r2[1]
    comp1 = -r2x + r1x
    comp2 = -r2y+ r1y
    rij_bold = [comp1,comp2]
    rij = np.abs(m.sqrt(rij_bold[0]**2 + rij_bold[1]**2))
    return rij,rij_bold


def ODE_sci(t,y,w0,Masses):
    # ODE to be used with runge kutta 4 method
    dy = np.zeros(len(y))
    rx1 = y[0]
    ry1 = y[1]
    rx2 = y[2]
    ry2 = y[3]
    rx3 = y[4]
    ry3 = y[5]

    r1 = [rx1,ry1]
    r2 = [rx2,ry2]
    r3 = [rx3,ry3]

    rs = [r1,r2,r3]
    dy[0] = y[6]
    dy[1] = y[7]
    dy[2] = y[8]
    dy[3] = y[9]
    dy[4] = y[10]
    dy[5] = y[11]
    dy[6] = 0
    dy[7] = 0
    dy[8] = 0
    dy[9] = 0
    dy[10] = 0
    dy[11] = 0

    r12 , r12_bold = findDist(rs[0],rs[1])
    r13 , r13_bold = findDist(rs[0],rs[2])
    r23 , r23_bold = findDist(rs[1],rs[2])
    r31 , r31_bold = findDist(rs[2],rs[0])
    r32 , r32_bold = findDist(rs[2],rs[1])
    r21 , r21_bold = findDist(rs[1],rs[0])
    # Force on body 1
    dy[6] -= G*Masses[1].mass*r12_bold[0]*r12**(-3)
    dy[7] -= G*Masses[1].mass*r12_bold[1]*r12**(-3)
    dy[6] -= G*Masses[2].mass*r13_bold[0]*r13**(-3)
    dy[7] -= G*Masses[2].mass*r13_bold[1]*r13**(-3)
    # Force on body 2
    dy[8] -= G*Masses[0].mass*r21_bold[0]*r21**(-3)
    dy[9] -= G*Masses[0].mass*r21_bold[1]*r21**(-3)
    dy[8] -= G*Masses[2].mass*r23_bold[0]*r23**(-3)
    dy[9] -= G*Masses[2].mass*r23_bold[1]*r23**(-3)
    # Force on body 3
    dy[10] -= G*Masses[0].mass*r31_bold[0]*r31**(-3)
    dy[11] -= G*Masses[0].mass*r31_bold[1]*r31**(-3)
    dy[10] -= G*Masses[1].mass*r32_bold[0]*r32**(-3)
    dy[11] -= G*Masses[1].mass*r32_bold[1]*r32**(-3)
    return dy


def QuintEq(m1,m2,m3):
    # finds the roots for the quintic equation
    l5 = m2 + m3
    l4 = 2*m2 + 3*m3
    l3 = m2 + 3*m3
    l2 = -3*m1 - m2
    l1 = -3*m1-2*m2
    l0 = -m1 - m2
    coeffs = [l5,l4,l3,l2,l1,l0]
    roots = np.roots(coeffs)
    return roots

def Energy(xs,Masses):
    # Finds the energy of the system
    m1 = Masses[0].mass
    m2 =  Masses[1].mass
    m3 =  Masses[2].mass
    b1 = [xs[0],xs[1],xs[6],xs[7]]
    b2 = [xs[2],xs[3],xs[8],xs[9]]
    b3 = [xs[4],xs[5],xs[10],xs[11]]
    bs = [b1,b2,b3]
    v = [[b1[2],b1[3]], [b2[2],b2[3]] ,[b3[2],b3[3]] ]
    r = [[b1[0],b1[1]], [b2[0],b2[1]] ,[b3[0],b3[1]] ]
    r12 = [b1[0] - b2[0] , b1[1] - b2[1]]
    r13 = [b1[0] - b3[0] , b1[1] - b3[1]]
    r23 = [b2[0] - b3[0] , b2[1] - b3[1]]
    r12v=np.asarray(r12)

    r13v=np.asarray(r13)
    r23v=np.asarray(r23)
    s12 = np.sqrt(np.dot(r12v,r12v))
    s13 = np.sqrt(np.dot(r13v,r13v))
    s23 = np.sqrt(np.dot(r23v,r23v))
    KE = 0.5 * (m1*np.dot(v[0],v[0]) + m2*np.dot(v[1],v[1]) + m3*np.dot(v[2],v[2]))
    denom = (-m1*m2/s12 - m1*m3/s13 - m2*m3/s23)
    return (KE+denom)

def BodyReset(x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3,m1,m2,m3,npts):
    # Sets the initial bodies to their starting conditions
    m1 = Body(1,[[x1],[y1]],[[vx1],[vy1]],m1)
    m2 = Body(2,[[x2],[y2]],[[vx2],[vy2]],m2)
    m3 = Body(3,[[x3],[y3]],[[vx3],[vy3]],m3)
    x_1 = np.zeros((npts,12))
    x = [x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3]
    x_1[0,:]=x
    return [m1,m2,m3] , x_1 ,x

def TrajPlot(x,tlist,E,title,arrows=True,circles=True):
    # Plotting function that plots energy plot and contour plot
    fig, axs = plt.subplots(2)
    axs[0].plot(tlist,E)
    axs[1].plot(x[:,0],x[:,1],'g')
    axs[1].plot(x[:,2],x[:,3],'r')
    axs[1].plot(x[:,4],x[:,5],'b')
    factor = 5
    if arrows:
        axs[1].arrow(x=x[-1][0],y=x[-1][1],dx=x[-1][6]/factor,dy=x[-1][7]/factor,color='g',head_width=0.2, head_length=0.1,width=0.02)
        axs[1].arrow(x=x[-1][2],y=x[-1][3],dx=x[-1][8]/factor,dy=x[-1][9]/factor,color='r',head_width=0.2, head_length=0.1,width=0.02)
        axs[1].arrow(x=x[-1][4],y=x[-1][5],dx=x[-1][10]/factor,dy=x[-1][11]/factor,color='b',head_width=0.2, head_length=0.1,width=0.02)

    radius = 1/20
    if circles:
        circle1 = plt.Circle((x[-1][0],x[-1][1]),radius,color ='g')
        circle2 = plt.Circle((x[-1][2],x[-1][3]),radius,color = 'r')
        circle3 = plt.Circle((x[-1][4],x[-1][5]),radius,color ='b')

        axs[1].add_artist(circle1)
        axs[1].add_artist(circle2)
        axs[1].add_artist(circle3)

    axs[0].set(xlabel="Time ($T_0$)", ylabel="Energy (J)")
    axs[1].set(xlabel="$x\ (m)$", ylabel="$y\ (m)$")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.40)
    fig.suptitle(title)
    plt.savefig('./'+title+'.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()

def initialConditions(w):
    # Finds the intial conditions
    mass1 = 1
    mass2 = 2
    mass3 = 3
    # Find starting parameters
    roots = QuintEq(mass1,mass2,mass3)
    lam = np.real(roots[2])

    a3 = (1/w**2)*(mass2 + mass3 - ((mass1*(1+2*lam))/((lam**2)*(1+lam)**2)))
    a = a3**(1/3)
    x2 = ((1/(w**2*a**2))*((mass1)/(lam**2)-mass3))
    x1 = (x2 - lam*a)
    x3 = (-(mass1*x1+mass2*x2)/(mass3))
    v1y = w*x1
    v2y = w*x2
    v3y = w*x3
    return x1,x2,x3,v1y,v2y,v3y

def animates(y_1,Masses,npts,title,snapshots=None,trail=True):
    # Sets all the positions in the Class to be sent to the animator
    Masses[0].setPositions(y_1[:,0],y_1[:,1],y_1[:,6],y_1[:,7])
    Masses[1].setPositions(y_1[:,2],y_1[:,3],y_1[:,8],y_1[:,9])
    Masses[2].setPositions(y_1[:,4],y_1[:,5],y_1[:,10],y_1[:,11])
    ani = Animate(Masses,npts,title,snapshots,trail)
    ani.run()

def a():
    #x1,x2,x3,v1y,v2y,v3y = initialConditions()
    h = 0.001
    delta = 1E-9
    w0s = [1,1+delta,1-delta]

    for m in range(len(w0s)):
        t0 = (np.pi * 2)/w0s[m]
        x1,x2,x3,v1y,v2y,v3y = initialConditions(w0s[m]) # Sets initial conditions from root
        n = 3 # Number of time periods
        tmax =  n*t0
        tlist = np.arange(0,int(tmax),h)/t0
        npts = len(tlist)
        Masses, x_1 , x = BodyReset(x1,0,x2,0,x3,0,0,v1y,0,v2y,0,v3y,1,2,3,npts)
        MassesRk , y_1 , y = BodyReset(x1,0,x2,0,x3,0,0,v1y,0,v2y,0,v3y,1,2,3,npts)
        E = [Energy(y,MassesRk)]
        ELeap = [Energy(y,Masses)]
        for j in range(1,npts):
            x = Leapfrog(ODE,x,tlist[j-1],h,w0s[m],Masses) # Runs the step integrator
            x_1[j,:] = x
            ELeap.append(Energy(x,Masses))
            y = rk4(ODE_sci,y,tlist[j-1],h,w0s[m],MassesRk)
            y_1[j,:] = y
            E.append(Energy(y,MassesRk))
        TrajPlot(x_1,tlist,ELeap,"LeapFrog,  $ω_0$="+str(w0s[m])+ " ,Max T="+str(n)+"$T_0$")
        TrajPlot(y_1,tlist,E,"RK4,  $ω_0$="+str(w0s[m])+ " ,Max T="+str(n)+"$T_0$")
    title = "RK4"
    animates(y_1,Masses,npts,title,[0,50,500,1000])

def b():
    # Package ODE Solver solution
    h = 0.001
    delta = 1E-9
    w0s = [1,1+delta,1-delta]
    for j in range(len(w0s)):
        t0 = (np.pi * 2)/w0s[j]
        n = 10
        tmax =  n*t0
        tlist = np.arange(0,int(tmax),h)/t0
        npts = len(tlist)
        x1,x2,x3,v1y,v2y,v3y = initialConditions(w0s[j]) # Sets initial conditions
        rtol = 1E-12
        E = []
        Masses, x_1 , x= BodyReset(x1,0,x2,0,x3,0,0,v1y,0,v2y,0,v3y,1,2,3,npts)
        # Runs odeint
        x = odeint(ODE_sci,x_1[0],tlist,args=(w0s[j],Masses),tfirst=True,rtol=rtol,atol=rtol)
        for k in range(len(x)):
            E.append(Energy(x[k],Masses))
        TrajPlot(x,tlist,E,"Odeint, $ω_0 = $"+str(w0s[j]))

def c():
    h = 0.001
    delta = 1E-9
    w0 = 1+delta
    t0 = np.pi * 2 /w0
    n = 6
    tmax =  n * t0
    tlist = np.arange(0,tmax,h)/t0
    npts = len(tlist)
    x1,x2,x3,v1y,v2y,v3y = initialConditions(w0)
    MassesRk, y_1 , y= BodyReset(x1,0,x2,0,x3,0,0,v1y,0,v2y,0,v3y,1,2,3,npts)
    Masses, x_1 , x= BodyReset(x1,0,x2,0,x3,0,0,v1y,0,v2y,0,v3y,1,2,3,npts)
    E = [Energy(y,MassesRk)]
    ELeap = [Energy(y,Masses)]
    for j in range(1,npts):
        if(j==npts/2):
            for m in range(6,len(y)):
                y[m] *= -1
                x[m] *= -1
        x = Leapfrog(ODE,x,tlist[j-1],h,w0,Masses)
        x_1[j,:] = x
        ELeap.append(Energy(x,Masses))
        y = Leapfrog(ODE,y,tlist[j-1],h,w0,MassesRk)
        y_1[j,:] = y
        E.append(Energy(y,MassesRk))

    title = "RK4,  ω="+str(w0) +", Reversed at t = $3T_0$"
    titleLeap = "Leapfrog ω="+str(w0) +", Reversed at t = $3T_0$"
    TrajPlot(y_1,tlist,E,title)
    TrajPlot(x_1,tlist,ELeap,titleLeap)
    animates(y_1,MassesRk,npts,title,[0,50,500,1000])

def di():
    #Initial Conditions
    x1 = -0.30805788
    v1y = -1.015378093
    x2 = 0.15402894
    y2 = -0.09324743
    v2x =0.963502817
    v2y = 0.507689046
    x3 = x2
    y3=-y2
    v3x=-v2x
    v3y=v2y
    delta = 1E-9
    w0 = 3.3
    h = 0.001
    t0 = np.pi * 2 /w0
    n = 3
    tmax =  n * t0
    tlist = np.arange(0,tmax,h)/t0
    npts = len(tlist)
    MassesRk, y_1 , y= BodyReset(x1,0,x2,y2,x3,y3,0,v1y,v2x,v2y,v3x,v3y,1/3,1/3,1/3,npts)
    E = [Energy(y,MassesRk)]
    for j in range(1,npts):
        y = rk4(ODE_sci,y,tlist[j-1],h,w0,MassesRk)
        E.append(Energy(y,MassesRk))
        y_1[j,:] = y
    title = "Initial Conditions Set 1"
    TrajPlot(y_1,tlist,E,title,arrows=False,circles=False)
    animates(y_1,MassesRk,npts,title,trail=False)

def dii():
    #Initial Conditions
    x1 = 0.97000436
    y1 = -0.24308753
    v3x=-0.93240737
    v3y=-0.86473146
    v1x = -v3x/2
    v1y = -v3y/2
    x2 = -x1
    y2 = -y1
    v2x = v1x
    v2y = v1y

    w0 = 2.47
    h = 0.001
    t0 = np.pi * 2 /w0
    n = 3
    tmax =  n * t0
    tlist = np.arange(0,tmax,h)/t0
    npts = len(tlist)
    MassesRk, y_1 , y= BodyReset(x1,y1,x2,y2,0,0,v1x,v1y,v2x,v2y,v3x,v3y,1/3,1/3,1/3,npts)
    E = [Energy(y,MassesRk)]
    for j in range(1,npts):
        y = rk4(ODE_sci,y,tlist[j-1],h,w0,MassesRk)
        E.append(Energy(y,MassesRk))
        y_1[j,:] = y
    title = "Initial Conditions Set 2"
    TrajPlot(y_1,tlist,E,title)
    animates(y_1,MassesRk,npts,title,trail=False)
a()
b()
c()
di()
dii()
