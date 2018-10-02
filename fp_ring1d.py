from fenics import *
import numpy as np
from matplotlib import pyplot as plt

T = 0.1
num_steps = 100
dt = T / num_steps

class PeriodicBoundary(SubDomain):
   def inside(self, x, on_boundary):
      return (near(x[0], 0.0) and on_boundary)

   def map(self, x, y):
      y[0] = x[0] - 1.0

nx = 10000
mesh = UnitIntervalMesh(nx)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())


# Initial condition defined by parts
p1 = Expression('4*(x[0]-0.25)', degree=1)
p2 = Expression('4*(0.75-x[0])', degree=1)
u_0 = Expression('x[0] < 0.25 ? 0 : '
                 '(x[0] < 0.5 ? p1 : '
                 '(x[0] < 0.75 ? p2 : 0))',
                 p1=p1, p2=p2, degree=1)

# Define the potential
phi = Expression('pow(sin(8*pi*(x[0]-0.25)), 2) + 0.1', degree=1)
phi = interpolate(phi, V)

# Initialize u_n sequence
u_n = interpolate(u_0, V)


# Weak functional formulation a(u_{n+1}, v) = L(u_{n})
u = TrialFunction(V)
v = TestFunction(V)

a = u*v*dx + dt*dot(grad(phi*u), grad(v))*dx
L = u_n*v*dx


u = Function(V)
t = 0

vtkfile = File('FP1D/solution.pvd')

vtkfile << u_n
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    solve(a == L, u)
    
    vtkfile << u

    # Update previous solution
    u_n.assign(u)