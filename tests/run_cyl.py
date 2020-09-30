import sys
sys.path.append('../')
import numpy as np
import time

from mesh.read_starcd import Mesh

import solver.solver as Boltzmann
import pickle

log = open('log.txt', 'w') #log file (w+)
log.close()

# compute parameters for flow around cylinder

# Parameters for argon (default)
gas_params = Boltzmann.GasParams()

Mach = 10.
Kn = 0.564
delta = 8.0 / (5 * np.pi**0.5 * Kn)
n_l = 2e+23
T_l = 200.
u_l = Mach * ((gas_params.g * gas_params.Rg * T_l) ** 0.5)
T_w = 5.0 * T_l
r = 1e-7

n_s = n_l
T_s = T_l

p_s = gas_params.m * n_s * gas_params.Rg * T_s

v_s = np.sqrt(2. * gas_params.Rg * T_s)
mu_s = gas_params.mu(T_s)

l_s = delta * mu_s * v_s / p_s

#print 'l_s = ', l_s

#print 'v_s = ', v_s

nv = 44
vmax = 22 * v_s

hv = 2. * vmax / nv
vx_ = np.linspace(-vmax+hv/2, vmax-hv/2, nv) # coordinates of velocity nodes

v = Boltzmann.VelocityGrid(vx_, vx_, vx_)

f_init = lambda x, y, z, v: Boltzmann.f_maxwell(v, n_l, u_l, 0., 0., T_l, gas_params.Rg)
f_bound = Boltzmann.f_maxwell(v, n_l, u_l, 0., 0., T_l, gas_params.Rg)
#print(f_bound)
fmax = Boltzmann.f_maxwell(v, 1., 0., 0., 0., T_w, gas_params.Rg)
#print(fmax)
problem = Boltzmann.Problem(bc_type_list = ['sym-z', 'in', 'out', 'wall', 'sym-y'],
                                bc_data = [[],
                                           [f_bound],
                                           [f_bound],
                                           [fmax],
                                           []], f_init = f_init)


#print 'vmax =', vmax

CFL = 5e+1
tol = 1e-3

solver = 'impl'

config = Boltzmann.Config(solver, CFL, tol, tec_save_step = 1)

path = '../mesh/mesh-cyl/'
mesh = Mesh()
mesh.read_starcd(path)

# =============================================================================
# f = open('../mesh/mesh-cyl/mesh-cyl.pickle', 'rb')
#
# mesh = pickle.load(file = f)
#
# f.close()
# =============================================================================

log = open('log.txt', 'a')
log.write('Mach = ' + str(Mach) + '\n')
log.close()

print 'Initialization...'
t1 = time.clock()
S = Boltzmann.Solution(gas_params, problem, mesh, v, config)
t2 = time.clock()
print 'Complete! Took', str(t2 - t1), 'seconds'

nt = 30
t1 = time.time()
S.make_time_steps(config, nt)
t2 = time.time()

log = open('log.txt', 'a')
log.write('Time  = ' + str(t2 - t1) + '\n')
log.close()

S.save_macro()

log = open('log.txt', 'a')
log.write('Residual = ' + str('{0:5.2e}'.format(S.frob_norm_iter[-1]/S.frob_norm_iter[0])) + '\n')
log.close()
