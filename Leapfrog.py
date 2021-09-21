# Assignment 3 Josh Dykstra
import math as m
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
w0 = 1

def Leapfrog(f , r0, v0 , t , h , w0):
    # Leapfrog integrator method
    hh = h/2.0
    r1 = r0 + hh*f(0,r0,v0,t, w0)
    v1 = v0 + h*f(1,r1,v0,t+hh ,w0)
    r1 = r1 + hh*f(0,r0,v1,t+h , w0)
    return r1 , v1

def rk4(f,r0,v0,t,h,w0):
    # RK4 integrator method
    y = [r0,v0]
    k_1 = h * np.asarray(f(t,y,w0))
    k_2 = h * np.asarray(f((t+(h/2)),(y+(k_1/2)),w0))
    k_3 = h * np.asarray(f((t + (h/2)),(y+(k_2/2)),w0))
    k_4 = h * np.asarray(f(t + h, y + k_3,w0))
    y_next = y + (1/6)*(k_1 + 2 * k_2 + 2 * k_3 + k_4)
    return y_next[0] , y_next[1]

def ODERK(t,y,w0):
    # ODE to be used with RK4
    dy = np.zeros(len(y))
    dy[0] = y[1]
    dy[1] = -y[0]
    return dy

def ODE(a,r0,v0,t,w0):
    # ODE to be used with Leapfrog
    dx = v0
    dv = - r0
    if a == 1:
        return  dv
    else:
        return dx

def reset(npts):
    # Reset conditions
    r = np.zeros(npts)
    v = np.zeros(npts)
    E = np.zeros(npts)
    r[0] = 1
    v[0] = 0
    return r , v , E

def Energy(t,w0):
    # Returns energy of the system
    return 0.5*(m.cos(w0*t))**2 + 0.5*(-w0 * m.sin(w0*t))**2

def call(f,ODE,tlist,h,w0):
    # Calls the integrator method and ode
    npts = len(tlist)
    r , v, E= reset(npts)
    E[0] = 0.5*r[0]**2 + 0.5*v[0]**2
    for x in range(1,npts):
        r[x] , v[x] = f(ODE,r[x-1],v[x-1],tlist[x],h,w0)
        E[x] = (0.5*r[x]**2 + 0.5*v[x]**2)
    return r, v, E

def Error(E,tlist):
    # Finds the error
    npts = len(tlist)
    Error = np.zeros(npts)
    relError = np.zeros(npts)
    for x in range(1,npts):
        Error[x] = -Energy(tlist[x],w0) + E[x]
        relError[x] = Error[x]/Energy(tlist[x],w0)
        #print("Error: "+ str(Error[x]))
        #print("RelError: "+ str(relError[x]))
    return Error, relError


def Q1a():
    t0 =  2*m.pi
    h = 0.02*t0
    tmax =  50*t0
    global w0
    w0 =1

    rel = 1
    while(rel>1E-6):
        tlist = np.arange(0,tmax,h)
        npts = len(tlist)
        r , v ,E = call(Leapfrog,ODE,tlist,h,w0)
        #rRk , vRk ,Erk = call(rk4,ODERK,tlist,h,w0)
        LeapError , LeaprelError = np.abs(Error(E,tlist))
        rel = np.max(LeaprelError)
        #print(h)
        h /= 10

    print("h:" + str(h*10))
    print("RELATIVE ERROR: " + str(np.max(LeaprelError)))
    plt.semilogy(tlist,LeaprelError,'g')
    plt.xlabel("Time ($s$)")
    plt.ylabel("Energy Relative Error")
    plt.title("Energy Relative Error Temporal Evolution")
    plt.savefig('./RelError.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()

def Q1b():

    t0 = 2*m.pi
    hs = [0.02,0.04,0.1]
    for x in range(len(hs)):
        tmax =  10000
        tlist = np.arange(0,tmax,t0*hs[x])
        r , v ,E = call(Leapfrog,ODE,tlist,t0*hs[x],w0)
        rRk , vRk ,Erk = call(rk4,ODERK,tlist,t0*hs[x],w0)
        LeapError , LeaprelError = np.abs(Error(E,tlist))
        rkError , rkrelError = Error(Erk,tlist)
        ti = "$h=$" + str(hs[x])+"$T_0$"
        fig, axs = plt.subplots(2,2)#sharex = 'col')
        axs[0,0].plot(r,v,'g')
        axs[1,0].plot(rRk,vRk,'g')
        axs[0,1].plot(tlist,E,'g')
        axs[1,1].plot(tlist,Erk,'g')
        axs[1,0].set(xlabel="x (m)",ylabel="v ($ms^{-1}$)",title="RK4")
        axs[0,0].set(ylabel="v ($ms^{-1}$)",title="Leapfrog")
        axs[0,1].set(ylabel="Energy (J)",title="Leapfrog")
        axs[1,1].set(xlabel="t (s)",ylabel="Energy (J)",title="RK4")
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.40)

        fig.suptitle(ti)

        axs[0,1].set_ylim(0,0.8)
        axs[1,1].set_ylim(0,0.8)
        plt.savefig('./'+ti+'.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
        plt.show()

Q1a()
Q1b()
