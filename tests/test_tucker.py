import sys
sys.path.append('../')
import numpy as np
import time
from datetime import datetime
import os
import tucker.tucker as tuck
import solver.solver_tucker as Boltzmann
import solver.solver as Boltzmann_full

gas_params = Boltzmann.GasParams()

nv = 44
vmax = 22 * 200.

hv = 2. * vmax / nv
vx_ = np.linspace(-vmax+hv/2, vmax-hv/2, nv) # coordinates of velocity nodes

v = Boltzmann.VelocityGrid(vx_, vx_, vx_)
v_full = Boltzmann_full.VelocityGrid(vx_, vx_, vx_)

print(v.v2)

print('Sum')
print((tuck.sum(v.v2) - np.sum(v.vx*v.vx + v.vy*v.vy + v.vz*v.vz)) / tuck.sum(v.v2))

print(np.sum(v.vx*v.vx + v.vy*v.vy + v.vz*v.vz))
print(tuck.sum(v.v2))
# =============================================================================
print('Division')
print(np.linalg.norm(tuck.div_1r(v.vx_tt*v.vz_tt, v.vy_tt).full() - (v.vx*v.vz / v.vy)))
# =============================================================================
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
# =============================================================================
# TODO: Test multiplication when ranks are large

# TODO: Test creation of a rank-one tensor from factors

# TODO: test copy

t1 = v.v2.copy()

t1 = 2 * t1

print("Test copy: ", tuck.norm(t1 - 2 * v.v2)/tuck.norm(t1))

# TODO: create rank-one f_maxwell from macroparams, integrate it and
# check that macroparameters are reconstructed
ux = v.hvx
uy = v.hvy
uz = v.hvz
fmax = Boltzmann.f_maxwell_tuck(v, 1e+23, ux, uy, uz, 300., gas_params.Rg)
fmax_full = Boltzmann_full.f_maxwell(v_full, 1e+23, ux, uy, uz, 300., gas_params.Rg)
fmax_tuck_full = fmax.full()
print( 'delta Fmax = ', np.linalg.norm( fmax_full - fmax_tuck_full ) / np.linalg.norm( fmax_full ) )
# TODO: compute J with old function for full tensors and with new for Tucker

Macro = Boltzmann.comp_j(fmax, v, gas_params)

#Macro = Boltzmann_full.comp_j(fmax_full, v_full, gas_params)
print( 'n = ', Macro[1] )
print( 'dux = ', (Macro[2] - ux) / ux )
print( 'duy = ', (Macro[3] - uy) / uy )
print( 'duz = ', (Macro[4] - uz) / uz )
print( 'T = ', Macro[5] )
#
v2_tuck_full = tuck.tensor(v.vx*v.vx + v.vy*v.vy + v.vz*v.vz).round(1e-7, rmax = 1).full()
print (np.linalg.norm(v2_tuck_full - v.vx*v.vx + v.vy*v.vy + v.vz*v.vz) / np.linalg.norm(v.vx*v.vx + v.vy*v.vy + v.vz*v.vz))

print( np.min([1, 2]) )

print( (1, 1, 1) == [1, 1, 1] )