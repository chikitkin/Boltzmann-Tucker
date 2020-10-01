import sys
sys.path.append('../')
import numpy as np
import time
from datetime import datetime
import os
import tucker.tucker as tuck
import solver.solver_tucker as Boltzmann

nv = 44
vmax = 22 * 1000.

hv = 2. * vmax / nv
vx_ = np.linspace(-vmax+hv/2, vmax-hv/2, nv) # coordinates of velocity nodes

v = Boltzmann.VelocityGrid(vx_, vx_, vx_)

print(v.v2)

print('Sum')
print((tuck.sum(v.v2) - np.sum(v.vx*v.vx + v.vy*v.vy + v.vz*v.vz)) / tuck.sum(v.v2))

print(np.sum(v.vx*v.vx + v.vy*v.vy + v.vz*v.vz))
print(tuck.sum(v.v2))


print('Division')
print(np.linalg.norm(tuck.div_1r(v.vx_tt*v.vz_tt, v.vy_tt).full() - (v.vx*v.vz / v.vy)))


print('Reflect')
print(np.linalg.norm(tuck.reflect_tuck(v.vx_tt + v.vz_tt, 'x').full() - (v.vx + v.vz)[::-1, :, :]))

# =============================================================================
#
# print('Rmul')
# print(v.vx_tt * np.float64(5) * v.vx_tt)
#
# =============================================================================
print('Round', '2')
print(v.v2.round(1e-7, rmax = 1))