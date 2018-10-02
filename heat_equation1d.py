from fenics import *
import numpy as np
from matplotlib import pyplot as plt

class PeriodicBoundary(SubDomain):
   def inside(self, x, on_boundary):
      return (near(x[0], 0.0) and on_boundary)

   def map(self, x, y):
      y[0] = x[0] - 1.0

T = 1
num_steps = 100
dt = T / num_steps

nx = 10000
mesh = UnitIntervalMesh(nx)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())
#V = FunctionSpace(mesh, 'P', 1)
p1 = Expression('4*(x[0]-0.25)', degree=1)
p2 = Expression('4*(0.75-x[0])', degree=1)
u_0 = Expression('x[0] < 0.25 ? 0 : (x[0] < 0.5 ? p1 : (x[0] < 0.75 ? p2 : 0))',
                 p1=p1, p2=p2, degree=1)

def boundary(x, on_boundary):
    return on_boundary

#bc = DirichletBC(V, Constant(0), boundary)
# u_0 = Expression('exp(-pow(x[0] - 0.5, 2)*16) - exp(-pow(0.5, 2)*16)',
#                  degree=2)

phi = Expression('pow(sin(8*pi*(x[0]-0.25)), 2)', degree=1)
# phi = Constant(1)
u_n = interpolate(u_0, V)
phi = interpolate(phi, V)

u = TrialFunction(V)
v = TestFunction(V)

a = u*v*dx + dt*dot(grad(phi*u), grad(v))*dx
L = u_n*v*dx

# Time-stepping
u = Function(V)
t = 0

vtkfile = File('heat1d/solution.pvd')

vtkfile << u_n
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    solve(a == L, u)
    
    vtkfile << u

    # Update previous solution
    u_n.assign(u)