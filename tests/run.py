import sys
sys.path.append('../')
import numpy as np
import time

from mesh.read_starcd import Mesh
import pickle

tensor_format = 'full'

if tensor_format == 'full':
    import solver.solver as Boltzmann

elif tensor_format == 'tt':
    import tt
    import solver.solver_tt as Boltzmann

elif tensor_format == 'tuck':
    import tucker.tucker as tuck
    import solver.solver_tuck as Boltzmann

else:
    raise Exception('Wrong arguments!')

# compute parameters for flow around cylinder

# Parameters for argon (default)

def main():

    gas_params = Boltzmann.GasParams()

    Mach = 6.5
    Kn = 0.564
    delta = 8.0 / (5 * np.pi**0.5 * Kn)
    n_l = 2e+23
    T_l = 200.
    u_l = Mach * ((gas_params.g * gas_params.Rg * T_l) ** 0.5)
    T_w = 5.0 * T_l

    n_s = n_l
    T_s = T_l

    p_s = gas_params.m * n_s * gas_params.Rg * T_s

    v_s = np.sqrt(2. * gas_params.Rg * T_s)
    mu_s = gas_params.mu(T_s)

    l_s = delta * mu_s * v_s / p_s

    n_r = (gas_params.g + 1.) * Mach * Mach / ((gas_params.g - 1.) * Mach * Mach + 2.) * n_l
    u_r = ((gas_params.g - 1.) * Mach * Mach + 2.) / ((gas_params.g + 1.) * Mach * Mach) * u_l
    T_r = (2. * gas_params.g * Mach * Mach - (gas_params.g - 1.)) * ((gas_params.g - 1.) * Mach * Mach + 2.) / ((gas_params.g + 1) ** 2 * Mach * Mach) * T_l

    #print 'l_s = ', l_s

    #print 'v_s = ', v_s

    nv = 64
    vmax = 22 * v_s

    hv = 2. * vmax / nv
    vx_ = np.linspace(-vmax+hv/2, vmax-hv/2, nv) # coordinates of velocity nodes

    v = Boltzmann.VelocityGrid(vx_, vx_, vx_)

    f_in = Boltzmann.f_maxwell_t(v, n_l, u_l, 0., 0., T_l, gas_params.Rg)
    f_out = Boltzmann.f_maxwell_t(v, n_r, u_r, 0., 0., T_r, gas_params.Rg)

    def f_init(x, y, z, v):
        if (x <= 0.):
            return Boltzmann.f_maxwell_t(v, n_l, u_l, 0., 0., T_l, gas_params.Rg)
        else:
            return Boltzmann.f_maxwell_t(v, n_r, u_r, 0., 0., T_r, gas_params.Rg)

    #print(f_bound)
    fmax = Boltzmann.f_maxwell_t(v, 1., 0., 0., 0., T_w, gas_params.Rg)
    #print(fmax)
    problem = Boltzmann.Problem(bc_type_list = ['sym-z', 'in', 'out', 'wall', 'sym-y'],
                                bc_data = [[],
                                           [f_in],
                                           [f_out],
                                           [fmax],
                                           []], f_init = f_init)


    #print 'vmax =', vmax

    solver = 'impl'
    CFL = 50.
    tol = 1e-7

    config = Boltzmann.Config(solver, CFL, tol, tec_save_step = 10)

    path = '../mesh/mesh-shock/'
    #mesh = Mesh()
    #mesh.read_starcd(path, l_s)

    #f = open('../mesh/mesh-shock/mesh-shock.pickle', 'wb')
    #pickle.dump(mesh, f)
    #f.close()

    f = open('../mesh/mesh-shock/mesh-shock.pickle', 'rb')
    mesh = pickle.load(file = f)
    f.close()

    S = Boltzmann.Solution(gas_params, problem, mesh, v, config)

    log = open(S.path + 'log.txt', 'w') #log file (w+)
    log.close()

    log = open(S.path + 'log.txt', 'a')
    log.write('mesh = ' + path + '\n')
    log.write('Mach = ' + str(Mach) + '\n')
    log.write('nv = ' + str(nv) + '\n')
    log.write('solver = ' + str(solver) + '\n')
    log.write('CFL = ' + str(CFL) + '\n')
    log.write('tol = ' + str(tol) + '\n')
    log.close()

    nt = 10 #3000
    t1 = time.time()
    S.make_time_steps(config, nt)
    t2 = time.time()

    log = open(S.path + 'log.txt', 'a')
    log.write('Time  = ' + time.strftime('%H:%M:%S', time.gmtime(t2 - t1)) + '\n')
    log.close()

    S.save_macro()

    S.plot_macro()

    log = open(S.path + 'log.txt', 'a')
    log.write('Residual = ' + str('{0:5.2e}'.format(S.frob_norm_iter[-1]/S.frob_norm_iter[0])) + '\n')
    log.close()

if __name__ == '__main__':
    main()
