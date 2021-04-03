#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 13:46:59 2021

@author: isaacwalter
"""

"""
solve flash flood equation using Godunov's method

Equation: \partial_t A + \partial_s \kappa(s) A^p(s) = 0

"""

import numpy
from matplotlib import pyplot 

def kappa(s):
    """Channel topography in this variable"""
    return numpy.ones_like(s)

def flux(A, s):
    """ Mass flux """
    return kappa(s) * A**(3/2)

def max_char_speed(A, s):
    """ Stable timestep for max lambda = dQ/dA """
    return max(3 / 2*kappa(s) * A**(1/2))

def godunov_step(A, s, dt):
    """Godunovs method """
    ds = s[1] - s[0]
    f = flux(A, s)
    dA = numpy.zeros_like(A)
    """ not general line """
    dA[1:] = f[:-1] - f[1:]
    return A + dt / ds * dA

def bcs(A):
    """ Boundarys conditions for a simple outflow """
    A[0] = A[1]
    A[-1] - A[-2]
    
    return A 

def init_data(domain, N):
    """ setup arrays """
    
    s = numpy.linspace(domain[0], domain[1], N+2)
    """step function"""
    A = numpy.tanh(-s*5) + 2
    return s, A

def godunov1(t_end, domain=[-1, 1], N=100, cfl = 0.9):
    """Run godunovs method"""
    s, A = init_data(domain, N)
    ds = (s[1] - s[0])
    t = 0 
    while t < t_end:
        dt = cfl * ds / max_char_speed(A, s)
        if t + dt > t_end:
            dt = t_end - t + 1e-10
        A = godunov_step(A, s, dt)
        A = bcs(A)
        t += dt
    return s, A

s, A = godunov1(0.2, N=20)
pyplot.plot(s, A, "bx-")
pyplot.xlabel(r"$x$")


#
#s, A = godunov1(0.5, domain=[-5, 5], N=1000)
#pyplot.figure()
#pyplot.plot(s, A, "bx-")
#pyplot.xlabel(r"$x$")
#
#
#












