import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.constants
#import os
# os.chdir("D:/Documents/4th year modules/Project/Test files") # working dir for files

""" ===== To do list ====

find and test rk4 solver


"""

"""===============================Variables================================="""
"""
L is the length of the pendulum
Ms is the mass of the swing(not including frame)
Mp is the mass of the person
cheight is the height of the COM of the person when crouching
delta is the distance between the standing and crouching COMs of the person 
n is the number of rk4 steps per unit time
l is used to denote the length of the persons COM from the pivot



"""
"""============================= Calculations =============================="""




"Need an x(t) relation"
def energy(L, Ms=70, Mp=70, cheight=0.2, delta=0.6, n=100):
    t = 1000
    T, X, Z, L_list = rk4(np.pi/6, 0, dzdt, t, L , cheight, delta, Ms, Mp, n)
    Itot=[];K=[];P=[]
    E = []
    for i in range(len(X)):
        Itot.append(1/3*Ms*L**2 + 13.00 + Mp*(L_list[i])**2)
        K.append(1/2 * Itot[i]*Z[i]**2)
        P.append((scipy.constants.g)* (Mp*L_list[i]+Ms*L/2)* (1 - np.cos(X[i])))
        E.append(K[i]+P[i])
        
        """
        Need to workout dx/dt
        need a relation between x and t
        """
    plt.plot(X,E, label=('Total Energy for length '+ str(L)))
    plt.xlabel("$Angle$")
    plt.ylabel("$Energy$")
    plt.title("Energy for fixed lengths")
    plt.legend()
    plt.show()
    plt.plot(X,K, label="Kinetic Energy")
    plt.xlabel("$Angle$")
    plt.ylabel("$Energy$")
    plt.title("Energy for fixed lengths")
    plt.legend()
    plt.show()
    plt.plot(X,P, label="Potential Energy")
    plt.xlabel("$Angle$")
    plt.ylabel("$Energy$")
    plt.title("Energy for fixed lengths")
    plt.legend()
    plt.show()
    return()
#energy(L)    
    
def PhasePlane(L, cheight=0.2, delta=0.6, Ms=70, Mp=70 ):
    T, X, Z, L = rk4(np.pi/6, 0, dzdt, 1000,  L , cheight, delta, Ms, Mp)

    plt.plot(X,Z)
    plt.xlabel("Angle")
    plt.ylabel("Angle velocity")
    plt.title("Phase plane for "+ str(L))
    #plt.legend()
    plt.show()
    return()

    
"""============================ Test functions ============================="""

"""============================ Plot template =============================="""

"""
    con3 = pyplot.figure(figsize = (12, 4)) # subplots
    
    y1rk3 = con3.add_subplot(131) 
    y1rk3.plot(X,Y,'kx',label="First equation") 
    
    y1exact = con3.add_subplot(131)
    y1exact.plot(X,Y,'r-',label="Second equation (on same graph)")
    
    pyplot.xlabel("$x$")
    pyplot.ylabel("$y$")
    pyplot.title("Title")
    pyplot.legend()
    
    
    # 2nd subplot
    
    y2rk3 = con3.add_subplot(132)
    y2rk3.plot(X,Y[1,:],'kx',label="First equation")
    
    y2exact = con3.add_subplot(132)
    y2exact.plot(X,y2(X),'r-',label="Second equation (on same graph)")
          
    pyplot.xlabel("$x$")
    pyplot.ylabel("$y$")
    pyplot.title("Title")
    pyplot.legend()
"""
    
"""============================== RK4 Solver ==============================="""
    # (convert second order in to first order using change of var)
    # y'' = u' y' = u
    # https://sam-dolan.staff.shef.ac.uk/mas212/notebooks/ODE_Example.html
    # https://stackoverflow.com/questions/52334558/runge-kutta-4th-order-method-to-solve-second-order-odes
    # https://smath.com/wiki/GetFile.aspx?File=Examples/RK4-2ndOrderODE.pdf

def dzdt(t, z , x, L, l, Ms, Mp): 
    # z' = -g/l * sin(x)
    # x' = z
    return ((-(scipy.constants.g)*(Mp*l+Ms*L/2)/(1/3*Ms*L**2+13.00+Mp*l**2))* \
        np.sin(x))
  
# Finds value of y for a given x using step size h 
# and initial value y0 at x0.


def rk4(x0, z0, fn, t, L, cheight, delta, Ms, Mp, n): 
    """
    x0 - intial value of x
    z0 - intial value of z (x')
    fn - callable function for dz/dt
    t  - max time
    n  - number of steps in a second
    L  - crouching height
    """
    num = t*n
    h = 1/n
    X = [x0]
    T = [0]
    Z = [z0]
    
    IntialLength = L - cheight # crouching is slightly higher than the bar
    l = IntialLength
    
    L_list = [l] # bug fixing
    
    # Iterate for number of iterations 
    for i in range(num): 
        "Apply Runge Kutta Formulas to find next value"
        "System of differentials - solves 2nd ODE"
        m1 = h * Z[i]
        k1 = h * fn(T[i], Z[i], X[i], L, l, Ms, Mp) 
        
        m2 = h * (Z[i] + 0.5 * k1)
        k2 = h * fn( T[i]+0.5*h , Z[i]+0.5*k1 , X[i]+0.5*m1,L, l, Ms, Mp) 
        
        m3 = h * (Z[i] + 0.5 * k2)
        k3 = h * fn( T[i]+0.5*h , Z[i]+0.5*k2 , X[i]+0.5*m2,L, l, Ms, Mp) 
        
        m4 = h * (Z[i] + k3)
        k4 = h * fn( T[i]+h, Z[i]+k3 , X[i]+m3,L, l, Ms, Mp) 
  
        'update points'
        # Update next value of x 
        
        newx = X[i] + (1/6)*(m1 + 2 * m2 + 2 * m3 + m4) 

        
        # Update next value of z 
        newz = Z[i] + (1/6)*(k1 + 2 * k2 + 2 * k3 + k4)
 
        # Update next value of t 
        newt = T[i] + h
        
        #Check if values require switch and implement if yes
        if np.sign(newx)==np.sign(X[i]):
            if np.sign(newz)==np.sign(Z[i]):
                X = np.append(X,newx)
                Z = np.append(Z,newz)
                T = np.append(T,newt)
            elif np.sign(newz)!=np.sign(Z[i]):
                l =  IntialLength
                X = np.append(X,newx)
                Z = np.append(Z,newz)
                T = np.append(T,newt)
        elif np.sign(newx)!=np.sign(X[i]):
            lold = l
            l = l-delta 
            newz=(lold**2/l**2)*newz  
            X = np.append(X,newx)
            Z = np.append(Z,newz)
            T = np.append(T,newt)

        
        L_list = np.append(L_list,l) # bug fixing
        if abs(newx) > np.pi:
            
            break
        
        
    return T, X, Z, L_list



def output(t,L, cheight=0.2, delta=0.6, Ms=70, Mp=70):
    T, X, Z, L_list = rk4(np.pi/6, 0, dzdt, t, L, cheight, delta, Ms, Mp, 20)
    X = X/np.pi
    plt.plot(T,X)
    
    plt.xlabel("$Time$")
    plt.ylabel("Angle / pi")
    plt.title("Basic model of bar of length "+ str(L))
    #plt.legend()
    plt.show()
    print("Time taken to reach angle pi",T[-1])
    #print(L_list) # bug fixing

#output(1000, 3)
#output(1000, 4)
#output(1000, 5)
#output(1000, 6)
#output(1000, 7)
#output(1000, 8)

energy(8, n=200)
#PhasePlane(8)

#output(1000, 2)
