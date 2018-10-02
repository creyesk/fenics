from fenics import *
import numpy as np
from matplotlib import pyplot as plt

class PeriodicBoundary(SubDomain):
   def inside(self, x, on_boundary):
      return (near(x[0], 0.0) and on_boundary)

   def map(self, x, y):
      y[0] = x[0] - 1.0

T = 0.01
num_steps = 10
dt = T / num_steps

nx = 20000
mesh = UnitIntervalMesh(nx)
V = FunctionSpace(mesh, "CG", 2, constrained_domain=PeriodicBoundary())

# p1 = Expression('4*(x[0]-0.25)', degree=1)
# p2 = Expression('4*(0.75-x[0])', degree=1)
# u_0 = Expression('x[0] < 0.25 ? 0 : (x[0] < 0.5 ? p1 : (x[0] < 0.75 ? p2 : 0))',
#                  p1=p1, p2=p2, degree=2)
u_0 = Expression('exp(-pow(x[0] - 0.5, 2)*16) - exp(-pow(0.5, 2)*16)',
                 degree=2)

# u_0 = Expression('x[0]<=0.5 ? 2*x[0] : 2 - 2*x[0]', degree=1)


phi = Expression('sin(2*pi*(x[0]-0.25))', degree=2)
# phi = Constant(1)
u_n = interpolate(u_0, V)
phi = interpolate(phi, V)

u = TrialFunction(V)
v = TestFunction(V)

# F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n)*v*dx
# a, L = lhs(F), rhs(F)

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

    # Plot solution
    # plot(u)

    # Update previous solution
    u_n.assign(u)


