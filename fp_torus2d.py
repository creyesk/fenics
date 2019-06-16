from dolfin import *

# Create mesh and define function space
mesh = UnitSquareMesh(100, 100)
V2 = FunctionSpace(mesh, "CG", 2)

n_eigen = 25

# Define trial and test functions
u = TrialFunction(V2)
v = TestFunction(V2)

# Define bilinear form
mu = 0.0

a = (inner(div(grad(u)), div(grad(v)))
  + (1-mu)*(u.dx(0).dx(1)*v.dx(0).dx(1)
          - u.dx(0).dx(0)*v.dx(1).dx(1)
          - u.dx(1).dx(1)*v.dx(0).dx(0)))*dx

b = u*v*dx
A, B = PETScMatrix(), PETScMatrix()
assemble(a, A)
assemble(b, B)
# bc.apply(A)

eigensolver = SLEPcEigenSolver(A, B)
eigensolver.parameters['solver'] = "krylov-schur"

print('Finding eigenvalues this might take a while.')
eigensolver.solve(n_eigen)



# Save solution to file
file = File("BH_eigen/biharmonic.pvd")
for i in range(n_eigen):
    r, _, rx, _ = eigensolver.get_eigenpair(i)
    u = Function(V2)
    u.vector()[:] = rx
    file << u

# Save solution for use with Fokker-Planck
File('BH_eigen/mesh.xml') << mesh
r, _, rx, _ = eigensolver.get_eigenpair(22)

dofmap = V2.dofmap()
dofs = dofmap.dofs(mesh, 0)


#############################################
from fenics import *
from matplotlib import pyplot as plt

T = 2
num_steps = 100
dt = T / num_steps

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

# mesh = Mesh('BH_eigen/mesh.xml')
mesh = UnitSquareMesh(100, 100)
V1 = FunctionSpace(mesh, 'CG', 1, constrained_domain=pbc)
# pdb.set_trace()

# phi = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=2)
# phi = interpolate(phi, V1)

phi = Function(V2)
phi.vector()[:] = rx.get_local()*100
phi = interpolate(phi, V1)

u_0 = Constant(1)
u_n = interpolate(u_0, V1)


# Weak functional formulation a(u_{n+1}, v) = L(u_{n})
u = TrialFunction(V1)
v = TestFunction(V1)
alpha = 0.0
epsilon = 0.01
a = u*v*dx + dt*dot(-alpha*grad(abs(phi))*u + grad(u*(pow(phi,2)+epsilon)), grad(v))*dx
L = u_n*v*dx

u = Function(V1)
t = 0

vtkfile = File('FP2D/solution.pvd')

vtkfile << u_n
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    solve(a == L, u)
    
    vtkfile << u

    # Update previous solution
    u_n.assign(u)