import sys
sys.path.append('../')
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from mesh.read_starcd import write_tecplot
import tucker.tucker as tuck
import os
from datetime import datetime

def f_maxwell(v, n, ux, uy, uz, T, Rg):
    """Compute maxwell distribution function on cartesian velocity mesh

    vx, vy, vz - 3d numpy arrays with x, y, z components of velocity mesh
    in each node
    T - float, temperature in K
    n - float, numerical density
    ux, uy, uz - floats, x,y,z components of equilibrium velocity
    Rg - gas constant for specific gas
    """
    return n * ((1. / (2. * np.pi * Rg * T)) ** (3. / 2.)) * (np.exp(-((v.vx - ux)**2 + (v.vy - uy)**2 + (v.vz - uz)**2) / (2. * Rg * T)))

def f_maxwell_t(v, n, ux, uy, uz, T, Rg):
    """Compute maxwell distribution function on cartesian velocity mesh
    in tensor format

    vx, vy, vz - 3d numpy arrays with x, y, z components of velocity mesh
    in each node
    T - float, temperature in K
    n - float, numerical density
    ux, uy, uz - floats, x,y,z components of equilibrium velocity
    Rg - gas constant for specific gas
    """
    return n * ((1. / (2. * np.pi * Rg * T)) ** (3. / 2.)) * tuck.tuck_from_factors(np.exp(-((v.vx_ - ux) ** 2) / (2. * Rg * T)),
                                                                             np.exp(-((v.vy_ - uy) ** 2) / (2. * Rg * T)),
                                                                             np.exp(-((v.vz_ - uz) ** 2) / (2. * Rg * T)))

class VelocityGrid:
    """Class of velocity grid
    Contains full and tucker tensors of \xi, parameters of the grid and
    other auxillary tensors
   
    vx_, vy_, vz_ - 1d numpy arrays for each dimention
    """
    def __init__(self, vx_, vy_, vz_):

        self.vx_ = vx_
        self.vy_ = vy_
        self.vz_ = vz_

        self.nvx = np.size(vx_)
        self.nvy = np.size(vy_)
        self.nvz = np.size(vz_)

        self.hvx = vx_[1] - vx_[0]
        self.hvy = vy_[1] - vy_[0]
        self.hvz = vz_[1] - vz_[0]
        self.hv3 = self.hvx * self.hvy * self.hvz

        self.vx, self.vy, self.vz = np.meshgrid(vx_, vy_, vz_, indexing='ij')

        self.vx_t = tuck.tuck_from_factors(vx_, np.ones(self.nvy), np.ones(self.nvz))
        self.vy_t = tuck.tuck_from_factors(np.ones(self.nvx), vy_, np.ones(self.nvz))
        self.vz_t = tuck.tuck_from_factors(np.ones(self.nvx), np.ones(self.nvy), vz_)

        self.v2 = (self.vx_t*self.vx_t + self.vy_t*self.vy_t + self.vz_t*self.vz_t).round(1e-7) 

        self.zero = tuck.zeros((self.nvx, self.nvy, self.nvz))
        self.ones = tuck.ones((self.nvx, self.nvy, self.nvz))

class GasParams:
    """Class of gas parameters
    """
    Na = 6.02214129e+23 # Avogadro constant
    kB = 1.381e-23 # Boltzmann constant, J / K
    Ru = 8.3144598 # Universal gas constant

    def __init__(self, Mol = 40e-3, Pr = 2. / 3., g = 5. / 3., d = 3418e-13):
        self.Mol = Mol # molar mass
        self.Rg = self.Ru  / self.Mol # J / (kg * K) # gas constant
        self.m = self.Mol / self.Na # kg # mass of the molecule

        self.Pr = Pr # Prandtl number

        self.C = 144.4
        self.T_0 = 273.11
        self.mu_0 = 2.125e-05
        self.mu_suth = lambda T: self.mu_0 * ((self.T_0 + self.C) / (T + self.C)) * ((T / self.T_0) ** (3. / 2.))
        self.mu = lambda T: self.mu_suth(200.) * (T/200.)**0.734 # viscosity formula
        self.g = g # specific heat ratio
        self.d = d # diameter of molecule

class Problem:
    """Class of initial and boundary conditions of the problem
    """
    def __init__(self, bc_type_list = None, bc_data = None, f_init = None):
        # list of boundary conditions' types
        # acording to order in starcd '.bnd' file
        # list of strings
        self.bc_type_list = bc_type_list
        # data for b.c.: wall temperature, inlet n, u, T and so on.
        # list of lists
        self.bc_data = bc_data
        # Function to set initial condition
        self.f_init = f_init

def set_bc(gas_params, bc_type, bc_data, f, v, vn, vn_abs, tol):
    """Set boundary condition
    """
    if (bc_type == 'sym-x'): # symmetry in x
        return tuck.reflect_tuck(f, 'x')
    elif (bc_type == 'sym-y'): # symmetry in y
        return tuck.reflect_tuck(f, 'y')
    elif (bc_type == 'sym-z'): # symmetry in z
        return tuck.reflect_tuck(f, 'z')
    elif (bc_type == 'sym'): # zero derivative
        return f.copy()
    elif (bc_type == 'in'): # inlet
        # unpack bc_data
        return bc_data[0]
    elif (bc_type == 'out'): # outlet
        # unpack bc_data
        return bc_data[0]
    elif (bc_type == 'wall'): # wall
        # unpack bc_data
        fmax = bc_data[0]
        Ni = v.hv3 * tuck.sum((0.5 * f * (vn + vn_abs)).round(tol))
        Nr = v.hv3 * tuck.sum((0.5 * fmax * (vn - vn_abs)).round(tol))
        n_wall = - Ni/ Nr
        return n_wall * fmax

def comp_macro_params(f, v, gas_params):
    """Computes macroscopic parameters of the gas
    for given distribution function

    f - d.f. tensor
    v - VelocityGrid
    gas_params - GasParams
    """
    n = v.hv3 * tuck.sum(f)
    if n <= 0.:
        n = 1e+10

    ux = (1. / n) * v.hv3 * tuck.sum(v.vx_t * f)
    uy = (1. / n) * v.hv3 * tuck.sum(v.vy_t * f)
    uz = (1. / n) * v.hv3 * tuck.sum(v.vz_t * f)

    u2 = ux*ux + uy*uy + uz*uz

    T = (1. / (3. * n * gas_params.Rg)) * (v.hv3 * tuck.sum(v.v2 * f) - n * u2)
    if T <= 0.:
        T = 1.

    rho = gas_params.m * n
    p = rho * gas_params.Rg * T
    mu = gas_params.mu(T)
    nu = p / mu

    return n, ux, uy, uz, T, rho, p, nu

def comp_j(f, v, gas_params):
    """Computes S-model collision integral and macroscopic parameters of the gas
    for given distribution function

    f - d.f. tensor
    v - VelocityGrid
    gas_params - GasParams
    """
    n, ux, uy, uz, T, rho, p, nu = comp_macro_params(f, v, gas_params)

    cx = tuck.tuck_from_factors((1. / ((2. * gas_params.Rg * T) ** (1. / 2.))) * (v.vx_ - ux), np.ones(v.nvy), np.ones(v.nvz))
    cy = tuck.tuck_from_factors(np.ones(v.nvx), (1. / ((2. * gas_params.Rg * T) ** (1. / 2.))) * (v.vy_ - uy), np.ones(v.nvz))
    cz = tuck.tuck_from_factors(np.ones(v.nvx), np.ones(v.nvy), (1. / ((2. * gas_params.Rg * T) ** (1. / 2.))) * (v.vz_ - uz))

    c2 = ((cx*cx) + (cy*cy) + (cz*cz)).round(1e-7) #, rmax = 2)

    Sx = (1. / n) * v.hv3 * tuck.sum(cx * c2 * f)
    Sy = (1. / n) * v.hv3 * tuck.sum(cy * c2 * f)
    Sz = (1. / n) * v.hv3 * tuck.sum(cz * c2 * f)

    fmax = f_maxwell_t(v, n, ux, uy, uz, T, gas_params.Rg)

    f_plus = fmax * (v.ones + ((4. / 5.) * (1. - gas_params.Pr) * (Sx*cx + Sy*cy + Sz*cz) * ((c2 - (5. / 2.) * v.ones))))
    J = nu * (f_plus - f)
    J = J.round(1e-7) #, rmax = 2)

    return J, n, ux, uy, uz, T, rho, p, nu

class Config:
    def __init__(self, solver, CFL, tol, init_type = 'default', init_filename = None, tec_save_step = 1e+5):

        self.solver = solver # type of the solver (expl or impl)

        self.CFL = CFL
        self.tol = tol # relative accuracy of the tensor rounding

        self.init_type = init_type
        self.init_filename = init_filename

        self.tec_save_step = tec_save_step

class Solution:

    def __init__(self, gas_params, problem, mesh, v, config):

        self.gas_params = gas_params
        self.problem = problem
        self.mesh = mesh
        self.v = v
        self.config = config

        self.path = './' + 'job_tuck_' + config.solver + '_' + datetime.now().strftime("%Y.%m.%d_%H:%M:%S") + '/'
        os.mkdir(self.path)

        self.vn = [None] * mesh.nf # list of tensors of normal velocities at each mesh face
        self.vn_tmp = np.zeros((v.nvx, v.nvy, v.nvz))
#        self.vnm = [None] * mesh.nf # negative part of vn: 0.5 * (vn - |vn|)
#        self.vnp = [None] * mesh.nf # positive part of vn: 0.5 * (vn + |vn|)
        self.vn_abs = [None] * mesh.nf # approximations of |vn|

        for jf in range(mesh.nf):
            self.vn_tmp = mesh.face_normals[jf, 0] * v.vx + mesh.face_normals[jf, 1] * v.vy + mesh.face_normals[jf, 2] * v.vz
            self.vn[jf] = (mesh.face_normals[jf, 0] * v.vx_t + mesh.face_normals[jf, 1] * v.vy_t + mesh.face_normals[jf, 2] * v.vz_t).round(1e-3)
#            self.vnp[jf] = tuck.tensor(np.where(self.vn_tmp > 0, self.vn_tmp, 0.), eps = config.tol)
#            self.vnm[jf] = tuck.tensor(np.where(self.vn_tmp < 0, self.vn_tmp, 0.), eps = config.tol)
            if (mesh.isbound[jf] != -1) and (mesh.bound_face_info[mesh.isbound[jf], 1] == 3): # increase rank if wall
                self.vn_abs[jf] = tuck.tensor(np.abs(self.vn_tmp)).round(1e-14, rmax = 6) 
            else:
                self.vn_abs[jf] = tuck.tensor(np.abs(self.vn_tmp)).round(1e-14, rmax = 6) # TODO WAS 4

        self.h = np.min(mesh.cell_diam)
        self.tau = self.h * config.CFL / (np.max(np.abs(v.vx_)) * (3.**0.5))

        self.diag = [None] * mesh.nc # part of diagonal coefficient in implicit scheme
        self.diag_r1 = [None] * mesh.nc
        # precompute diag
        # simple approximation for v_abs
        self.vn_abs_r1 = tuck.tensor((v.vx**2 + v.vy**2 + v.vz**2)**0.5).round(1e-7, rmax = 1)
        for ic in range(mesh.nc):
            diag_temp = np.zeros((v.nvx, v.nvy, v.nvz))
            diag_sc = 0.
            for j in range(6):
                jf = mesh.cell_face_list[ic, j]
                vn_full = (mesh.face_normals[jf, 0] * v.vx + mesh.face_normals[jf, 1] * v.vy \
                           + mesh.face_normals[jf, 2] * v.vz) * mesh.cell_face_normal_direction[ic, j]
                vnp_full = np.where(vn_full > 0, vn_full, 0.)
                vn_abs_full = np.abs(vn_full)
                diag_temp += (mesh.face_areas[jf] / mesh.cell_volumes[ic]) * vnp_full
                diag_sc += 0.5 * (mesh.face_areas[jf] / mesh.cell_volumes[ic])
            self.diag_r1[ic] = diag_sc * self.vn_abs_r1
#            diag_t_full = tuck.tensor(diag_temp).round(1e-7, rmax = 1).full()
            diag_t_full = self.diag_r1[ic].full()
            ind_min = np.unravel_index(np.argmin(diag_t_full / diag_temp), diag_temp.shape)
#            diag_t_full = (diag_temp[ind_max] / diag_t_full[ind_max]) * diag_t_full
            self.diag_r1[ic] = (diag_temp[ind_min] / diag_t_full[ind_min]) * self.diag_r1[ic]

        # set initial condition
        self.f = [None] * mesh.nc # RENAME f!

        if (config.init_type == 'default'):
            for i in range(mesh.nc):
                x = mesh.cell_center_coo[i, 0]
                y = mesh.cell_center_coo[i, 1]
                z = mesh.cell_center_coo[i, 2]
                self.f[i] = problem.f_init(x, y, z, v)
        elif (config.init_type == 'restart'):
            # restart from distribution function
            self.f = self.load_restart()
        elif (config.init_type == 'macro_restart'):
            # restart form macroparameters array
            init_data = np.loadtxt(config.init_filename)
            for ic in range(mesh.nc):
                self.f[ic] = f_maxwell_t(v, init_data[ic, 0], init_data[ic, 1], init_data[ic, 2], init_data[ic, 3], init_data[ic, 5], gas_params.Rg)

        self.f_plus = [None] * mesh.nf # Reconstructed values on the right
        self.f_minus = [None] * mesh.nf # reconstructed values on the left
        self.flux = [None] * mesh.nf # Flux values
        self.rhs = [None] * mesh.nc
        self.df = [None] * mesh.nc

        # Arrays for macroparameters
        self.n = np.zeros(mesh.nc)
        self.rho = np.zeros(mesh.nc)
        self.ux = np.zeros(mesh.nc)
        self.uy = np.zeros(mesh.nc)
        self.uz = np.zeros(mesh.nc)
        self.p = np.zeros(mesh.nc)
        self.T = np.zeros(mesh.nc)
        self.nu = np.zeros(mesh.nc)
        self.rank = np.zeros(mesh.nc)
        self.data = np.zeros((mesh.nc, 10))

        self.frob_norm_iter = np.array([])

        self.create_res()

    def create_res(self):
        """Creates file for RHS
        """
        resfile = open(self.path + 'res.txt', 'w')
        resfile.close()

    def update_res(self):
        """Write RHS in a file
        """
        resfile = open(self.path + 'res.txt', 'a')
        resfile.write('%10.5E \n'% (self.frob_norm_iter[-1]))
        resfile.close()

    def write_tec(self):
        """Creates a tecplot data file 
        """
        fig, ax = plt.subplots(figsize = (20,10))
        line, = ax.semilogy(self.frob_norm_iter/self.frob_norm_iter[0])
        ax.set(title='$Steps =$' + str(self.it))
        plt.savefig(self.path + 'norm_iter.png')
        plt.close()

        self.data[:, 0] = self.n[:]
        self.data[:, 1] = self.ux[:]
        self.data[:, 2] = self.uy[:]
        self.data[:, 3] = self.uz[:]
        self.data[:, 4] = self.p[:]
        self.data[:, 5] = self.T[:]
        for ic in range(self.mesh.nc):
            self.data[ic, 6] = self.f[ic].r[0]
            self.data[ic, 7] = self.f[ic].r[1]
            self.data[ic, 8] = self.f[ic].r[2]
            self.data[ic, 9] = 1. * (self.f[ic].core.size + self.f[ic].u[0].size + self.f[ic].u[1].size + self.f[ic].u[2].size) / self.v.vx.size

        write_tecplot(self.mesh, self.data, self.path + 'tec.dat', ('n', 'ux', 'uy', 'uz', 'p', 'T', 'rank1', 'rank2', 'rank3', 'compression'))

    def save_macro(self):
        """Save macro restart
        """        
        np.savetxt(self.path + 'macro.txt', self.data)

    def save_restart(self):
        """ Save the solution into a file
        """
        m = np.max([self.f[i].r for i in range(self.mesh.nc)])
        
        F = np.zeros((m * m * m + self.v.nvx * m + self.v.nvy * m + self.v.nvz * m + 4, self.mesh.nc))
        
        for i in range(self.mesh.nc):
            F[-1:, i] = m
            F[-4:-1, i] = np.array(self.f[i].r)
            F[:self.f[i].core.size, i] = self.f[i].core.ravel()
            index = m * m * m
            F[index : index + self.f[i].u[0].size, i] = self.f[i].u[0].ravel()
            index = m * m * m + self.v.nvx * m
            F[index : index + self.f[i].u[1].size, i] = self.f[i].u[1].ravel()
            index = m * m * m + self.v.nvx * m + self.v.nvy * m
            F[index : index + self.f[i].u[2].size, i] = self.f[i].u[2].ravel()

        np.save(self.path + 'restart.npy', F)#, fmt='%s')

    def load_restart(self):
        """ Load the solution from a file
        """
        F = np.load(self.config.init_filename)

        f = list()

        m = int(F[-1, 0])

        for i in range(self.mesh.nc):

            t = tuck.tensor()

            t.n = [self.v.nvx, self.v.nvy, self.v.nvz]
            
            t.r = F[-4:-1, i].astype(np.int)

            t.core = F[:t.r[0]*t.r[1]*t.r[2], i].reshape((t.r[0], t.r[1], t.r[2]))

            index = m * m * m
            t.u[0] = F[index : index + self.v.nvx * t.r[0], i].reshape((self.v.nvx, t.r[0]))
            index = m * m * m + self.v.nvx * m
            t.u[1] = F[index : index + self.v.nvy * t.r[1], i].reshape((self.v.nvy, t.r[1]))
            index = m * m * m + self.v.nvx * m + self.v.nvy * m
            t.u[2] = F[index : index + self.v.nvz * t.r[2], i].reshape((self.v.nvz, t.r[2]))

            f.append(t)

        return f

    def plot_macro(self):

        fig, ax = plt.subplots(figsize = (12,6))
        line, = ax.plot(self.mesh.cell_center_coo[:, 0], (self.n - self.n[0]) / (self.n[-1] - self.n[0]), 'k-', linewidth=4)
        line.set_label('Density')
        line, = ax.plot(self.mesh.cell_center_coo[:, 0], (self.ux - self.ux[-1]) / (self.ux[0] - self.ux[-1]), 'b-', linewidth=4)
        line.set_label('Velocity')
        line, = ax.plot(self.mesh.cell_center_coo[:, 0], (self.T - self.T[0]) / (self.T[-1] - self.T[0]), 'r-', linewidth=4)
        line.set_label('Temperature')

        plt.grid()
        ax.legend()
        ax.set_xlabel('x')
        plt.savefig(self.path + 'plot.png', dpi = 200)
        plt.close()

    def make_time_steps(self, config, nt):
        """Makes nt time steps of the solution
        """
        self.config = config
        self.tau = self.h * config.CFL / (np.max(np.abs(self.v.vx_)) * (3.**0.5))

        self.it = 0
        while(self.it < nt):
            self.it += 1
            # reconstruction for inner faces
            # 1st order
            for ic in range(self.mesh.nc):
                for j in range(6):
                    jf = self.mesh.cell_face_list[ic, j]
                    if (self.mesh.cell_face_normal_direction[ic, j] == 1):
                        self.f_minus[jf] = self.f[ic].copy()
                    else:
                        self.f_plus[jf] = self.f[ic].copy()

            # boundary condition
            # loop over all boundary faces
            for j in range(self.mesh.nbf):
                jf = self.mesh.bound_face_info[j, 0] # global face index
                bc_num = self.mesh.bound_face_info[j, 1]
                bc_type = self.problem.bc_type_list[bc_num]
                bc_data = self.problem.bc_data[bc_num]
                if (self.mesh.bound_face_info[j, 2] == 1):
                    self.f_plus[jf] =  set_bc(self.gas_params, bc_type, bc_data, self.f_minus[jf], self.v, self.vn[jf], self.vn_abs[jf], config.tol)
                else:
                    self.f_minus[jf] = set_bc(self.gas_params, bc_type, bc_data, self.f_plus[jf], self.v, -self.vn[jf], self.vn_abs[jf], config.tol)

            # riemann solver - compute fluxes
            for jf in range(self.mesh.nf):
                self.flux[jf] = 0.5 * self.mesh.face_areas[jf] * \
                ((self.f_plus[jf] + self.f_minus[jf]) * self.vn[jf]  - (self.f_plus[jf] - self.f_minus[jf]) * self.vn_abs[jf])
                self.flux[jf] = self.flux[jf].round(config.tol)

            # computation of the right-hand side
            for ic in range(self.mesh.nc):
                self.rhs[ic] = self.v.zero.copy()
                # sum up fluxes from all faces of this cell
                for j in range(6):
                    jf = self.mesh.cell_face_list[ic, j]
                    self.rhs[ic] += -(self.mesh.cell_face_normal_direction[ic, j]) * (1. / self.mesh.cell_volumes[ic]) * self.flux[jf]
                    self.rhs[ic] = self.rhs[ic].round(config.tol)
                # Compute macroparameters and collision integral
                J, self.n[ic], self.ux[ic], self.uy[ic], self.uz[ic], self.T[ic], self.rho[ic], self.p[ic], self.nu[ic] = \
                comp_j(self.f[ic], self.v, self.gas_params)
                self.rhs[ic] += J
                self.rhs[ic] = self.rhs[ic].round(config.tol)

            self.frob_norm_iter = np.append(self.frob_norm_iter, np.sqrt(sum([(self.rhs[ic].norm())**2 for ic in range(self.mesh.nc)])) / self.mesh.nc)

            self.update_res()
            #
            # update values, expclicit scheme
            #
            if (self.config.solver == 'expl'):
                for ic in range(self.mesh.nc):
                    self.f[ic] = (self.f[ic] + self.tau * self.rhs[ic]).round(config.tol)
            #
            # LU-SGS iteration
            #
            elif (self.config.solver == 'impl'):
                for ic in range(self.mesh.nc - 1, -1, -1):
                    self.df[ic] = self.rhs[ic].copy()
                #
                # Backward sweep
                #
                for ic in range(self.mesh.nc - 1, -1, -1):
                    # loop over neighbors of cell ic
                    for j in range(6):
                        jf = self.mesh.cell_face_list[ic, j]
                        icn = self.mesh.cell_neighbors_list[ic, j] # index of neighbor
                        if self.mesh.cell_face_normal_direction[ic, j] == 1:
                            vnm_loc = 0.5 * (self.vn[jf] - self.vn_abs_r1) # vnm[jf]
                        else:
                            vnm_loc = - 0.5 * (self.vn[jf] + self.vn_abs_r1) # -vnp[jf]
                        if (icn >= 0 ) and (icn > ic):
                            self.df[ic] += -(self.mesh.face_areas[jf] / self.mesh.cell_volumes[ic]) \
                            * vnm_loc * self.df[icn]
                            self.df[ic] = self.df[ic].round(config.tol)
                    # divide by diagonal coefficient
                    diag_temp = ((1./self.tau + self.nu[ic]) * self.v.ones + self.diag_r1[ic]).round(1e-3, rmax = 1)
                    self.df[ic] = tuck.div_1r(self.df[ic], diag_temp)
                    self.df[ic] = self.df[ic].round(config.tol) # TODO
                #
                # Forward sweep
                #
                for ic in range(self.mesh.nc):
                    # loop over neighbors of cell ic
                    incr = self.v.zero.copy()
                    for j in range(6):
                        jf = self.mesh.cell_face_list[ic, j]
                        icn = self.mesh.cell_neighbors_list[ic, j] # index of neighbor
                        if self.mesh.cell_face_normal_direction[ic, j] == 1:
                            vnm_loc = 0.5 * (self.vn[jf] - self.vn_abs_r1) # vnm[jf]
                        else:
                            vnm_loc = - 0.5 * (self.vn[jf] + self.vn_abs_r1) # -vnp[jf]
                        if (icn >= 0 ) and (icn < ic):
                            incr+= -(self.mesh.face_areas[jf] / self.mesh.cell_volumes[ic]) \
                            * vnm_loc * self.df[icn]
                            incr = incr.round(config.tol)
                    # divide by diagonal coefficient
                    diag_temp = ((1./self.tau + self.nu[ic]) * self.v.ones + self.diag_r1[ic]).round(1e-3, rmax = 1)
                    self.df[ic] += tuck.div_1r(incr, diag_temp)
                    self.df[ic] = self.df[ic].round(config.tol)
                #
                # Update values
                #
                for ic in range(self.mesh.nc):
                    self.f[ic] += self.df[ic]
                    self.f[ic] = self.f[ic].round(config.tol)
                #
                # end of LU-SGS iteration
                #
            # save rhs norm and tec tile
            if ((self.it % config.tec_save_step) == 0):
                self.write_tec()
            if ((self.it % 25) == 0):
                self.save_restart()

        self.save_restart()
        self.write_tec()
