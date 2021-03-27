import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.constants
#import os
# os.chdir("D:/Documents/4th year modules/Project/Test files") # working dir for files

""" ===== To do list ====

find and test rk4 solver


"""

"""============================= Calculations =============================="""

def length(x, dx, L):
    """
    x - angle
    dx - angular velocity
    L - Intial lenght
    """
    # picks whether the kiiker is standing or crouching 
    if abs(dx) <= 0.1:  # speed is (near) zero
        'crouch stance'
        newL = L
    if abs(x) <= 0.1:  # angle is (near) zero'
        'stand stance'
        newL = L + 0.5
    return newL








"Need an x(t) relation"
def energy(L):
    
    X = np.linspace(0,2*np.pi, 1000)
    for i in range(len(L)):
        E = lambda x: 1/2 * L[i]**2 * dx**2 + (scipy.constants.g)* L[i]* (1 - np.cos(x))
        """
        Need to workout dx/dt
        need a relation between x and t
        """
        plt.plot(X,E(X), label=('Energy for a fixed length '+ str(L[i])).format(i=i))
    

    plt.xlabel("$Angle$")
    plt.ylabel("$Energy$")
    plt.title("Energy for fixed lengths")
    plt.legend()
    plt.show()
    return()
    

#L = np.array([4,5,6,7,8])
#energy(L)
    
"""============================ Test funcitons ============================="""

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

def dzdt(t, z , x, l): 
    # z' = -g/l * sin(x)
    # x' = z
    return (-(scipy.constants.g)/l) * np.sin(x)
  
# Finds value of y for a given x using step size h 
# and initial value y0 at x0.


def rk4(x0, z0, fn, t, n, L): 
    """
    x0 - intial value of x
    z0 - intial value of z (x')
    fn - callable function for dz/dt
    t  - total time
    n  - number of steps in a second
    L  - crouching height
    """
    num = t*n
    h = 1/n
    X = [x0]
    T = [0]
    Z = [z0]
    
    IntialLength = L - 0.2
    l = IntialLength
    # Iterate for number of iterations 
    for i in range(num): 
        "Apply Runge Kutta Formulas to find next value"
        
        m1 = h * Z[i]
        k1 = h * fn(T[i], Z[i], X[i], l) 
        
        m2 = h * (Z[i] + 0.5 * k1)
        k2 = h * fn( T[i]+0.5*h , Z[i]+0.5*k1 , X[i]+0.5*m1, l) 
        
        m3 = h * (Z[i] + 0.5 * k2)
        k3 = h * fn( T[i]+0.5*h , Z[i]+0.5*k2 , X[i]+0.5*m2, l) 
        
        m4 = h * (Z[i] + k3)
        k4 = h * fn( T[i]+h, Z[i]+k3 , X[i]+m3, l) 
  
        'update points'
        # Update next value of x 
        newx = X[i] + (1/6)*(m1 + 2 * m2 + 2 * m3 + m4) 
        X = np.append(X,newx)
        
        # Update next value of z 
        newz = Z[i] + (1/6)*(k1 + 2 * k2 + 2 * k3 + k4)
        Z = np.append(Z,newz)
        
        # Update next value of t 
        newt = T[i] + h
        T = np.append(T,newt)
        
        if abs(newz) < 0.1 or abs(newx) < 0.1:
            l = length(newx, newz, IntialLength)        

        if abs(newx) > np.pi:
            
            break
    return T, X, Z



def output(t,L):
    T, X, Z = rk4(np.pi/6, 0, dzdt, t, 20, L)
    X = X/np.pi
    plt.plot(T,X)
    
    plt.xlabel("$Time$")
    plt.ylabel("Angle / pi")
    plt.title("Basic model of bar of length "+ str(L))
    #plt.legend()
    plt.show()
    print("Time taken to reach angle pi",T[-1])

output(1000, 3)
output(1000, 4)
output(1000, 5)
output(1000, 6)
output(1000, 7)
output(1000, 8)



#t0 = 0
#x0 = np.pi/6