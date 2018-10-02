from dolfin import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Create mesh and define function space
mesh = UnitSquareMesh(200, 200)
V = FunctionSpace(mesh, "CG", 2)

# Define Dirichlet boundary
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class Source(Expression):
    def eval(self, values, x):
        values[0] = 4.0*pi**4*sin(pi*x[0])*sin(pi*x[1])

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define normal component, mesh size and right-hand side
h = 2*Circumradius(mesh)
h_avg = (h('+') + h('-'))/2.0
n = FacetNormal(mesh)
f = Expression('2000*(x[0]*x[0] + x[1]*x[1])', degree=2)

# Penalty parameter
alpha = Constant(8.0)

# Define bilinear form
a = inner(div(grad(u)), div(grad(v)))*dx \
  - inner(avg(div(grad(u))), jump(grad(v), n))*dS \
  - inner(jump(grad(u), n), avg(div(grad(v))))*dS \
  + alpha('+')/h_avg*inner(jump(grad(u),n), jump(grad(v),n))*dS

# Define linear form
L = f*v*dx

# Solve variational problem
u = Function(V)
solve(a == L, u, bc)

# Save solution to file
file = File("biharmonic.pvd")
file << u

# Plot solution
# plot(u, interactive=True)