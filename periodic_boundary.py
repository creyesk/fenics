from fenics import *
from matplotlib import pyplot as plt


class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and 
                (not ((near(x[0], 0) and near(x[1], 1)) or 
                        (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
        elif near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.

pbc = PeriodicBoundary()

mesh = UnitSquareMesh(1000, 1000)
V = FunctionSpace(mesh, 'CG', 1, constrained_domain=pbc)

u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)

a = dot(grad(u), grad(v))*dx
L = f*v*dx

u = Function(V)
solve(a == L, u)


plot(u)
plt.show()