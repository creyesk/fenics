from dolfin import *
import numpy as np
# Optimization options for the form compiler
# parameters["form_compiler"]["cpp_optimize"] = True
# parameters["form_compiler"]["optimize"] = True

# Create mesh and define function space
mesh = UnitSquareMesh(50, 50)
V = FunctionSpace(mesh, "CG", 2)

n_eigen = 40

# Define Dirichlet boundary
# bc = DirichletBC(V, Constant(0.0), 'on_boundary')

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define bilinear form
mu = 0.0

# a = (u.dx(0).dx(0)*v.dx(0).dx(0) + u.dx(1).dx(1)*v.dx(1).dx(1)
#    + mu*(u.dx(1).dx(1)*v.dx(0).dx(0) + u.dx(0).dx(0)*v.dx(1).dx(1))
#    + 2*(1-mu)*u.dx(0).dx(1)*v.dx(0).dx(1))*dx
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
    print('Eigenvalue {}: {}'.format(i, r**0.25))
    u = Function(V)
    u.vector()[:] = rx
    file << u

# Save solution for use with Fokker-Planck
File('BH_eigen/mesh.xml') << mesh
r, _, rx, _ = eigensolver.get_eigenpair(20)

# e_file = HDF5File(mesh.mpi_comm(), "BH_eigen/eigenfunction20.h5", "w")
# e_file.write(rx, "solution")
output_file = HDF5File(mesh.mpi_comm(), "u.h5", "w")
output_file.write(rx, "solution")
output_file.close()