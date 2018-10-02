from fenics import *
import numpy as np
from matplotlib import pyplot as plt

class PeriodicBoundary(SubDomain):
   def inside(self, x, on_boundary):
      return (near(x[0], 0.0) and on_boundary)

   def map(self, x, y):
      y[0] = x[0] - 1.0

T = 2.0
num_steps = 20
dt = T / num_steps

nx = 40
mesh = UnitIntervalMesh(nx)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())

u_0 = Expression('exp(-pow(x[0] - 0.5, 2))', degree=1)

u = TrialFunction(V)
v = TestFunction(V)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n)*v*dx
a, L = lhs(F), rhs(F)

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    solve(a == L, u, bc)

    # Plot solution
    plot(u)

    # Compute error at vertices
    u_e = interpolate(u_0, V)
    error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
    print('t = %.2f: error = %.3g' % (t, error))

    # Update previous solution
    u_n.assign(u)

plt.show()

