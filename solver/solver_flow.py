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

    return f_maxwell(v, n, ux, uy, uz, T, Rg)

class VelocityGrid:
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

        self.v2 = self.vx*self.vx + self.vy*self.vy + self.vz*self.vz

        self.zero = np.zeros((self.nvx, self.nvy, self.nvz))
        self.ones = np.ones((self.nvx, self.nvy, self.nvz))

class GasParams:
    Na = 6.02214129e+23 # Avogadro constant
    kB = 1.381e-23 # Boltzmann constant, J / K
    Ru = 8.3144598 # Universal gas constant

    def __init__(self, Mol = 40e-3, Pr = 2. / 3., g = 5. / 3., d = 3418e-13):
        self.Mol = Mol
        self.Rg = self.Ru  / self.Mol  # J / (kg * K)
        self.m = self.Mol / self.Na # kg

        self.Pr = Pr

        self.C = 144.4
        self.T_0 = 273.11
        self.mu_0 = 2.125e-05
        self.mu_suth = lambda T: self.mu_0 * ((self.T_0 + self.C) / (T + self.C)) * ((T / self.T_0) ** (3. / 2.))
        self.mu = lambda T: self.mu_suth(200.) * (T/200.)**0.734
        self.g = g # specific heat ratio
        self.d = d # diameter of molecule

class Problem:
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

def set_bc(gas_params, bc_type, bc_data, f, v, vn, vn_abs):
    """Set boundary condition
    """
    if (bc_type == 'sym-x'): # symmetry in x
        return f[::-1, :, :]
    elif (bc_type == 'sym-y'): # symmetry in y
        return f[:, ::-1, :]
    elif (bc_type == 'sym-z'): # symmetry in z
        return f[:, :, ::-1]
    elif (bc_type == 'sym'): # zero derivative
        return f[:, :, :]
    elif (bc_type == 'in'): # inlet
        # unpack bc_data
        return bc_data[0]
    elif (bc_type == 'out'): # outlet
        # unpack bc_data
        return bc_data[0]
    elif (bc_type == 'wall'): # wall
        # unpack bc_data
        fmax = bc_data[0]
        Ni = v.hv3 * np.sum(0.5 * f * (vn + vn_abs))
        Nr = v.hv3 * np.sum(0.5 * fmax * (vn - vn_abs))
        n_wall = - Ni/ Nr
        return n_wall * fmax

def comp_macro_params(f, v, gas_params):

    n = v.hv3 * np.sum(f)
    if n <= 0.:
        n = 1e+10

    ux = (1. / n) * v.hv3 * np.sum(v.vx * f)
    uy = (1. / n) * v.hv3 * np.sum(v.vy * f)
    uz = (1. / n) * v.hv3 * np.sum(v.vz * f)

    u2 = ux*ux + uy*uy + uz*uz

    T = (1. / (3. * n * gas_params.Rg)) * (v.hv3 * np.sum(v.v2 * f) - n * u2)
    if T <= 0.:
        T = 1.

    rho = gas_params.m * n
    p = rho * gas_params.Rg * T
    mu = gas_params.mu(T)
    nu = p / mu

    return n, ux, uy, uz, T, rho, p, nu

def comp_j(f, v, gas_params):

    n, ux, uy, uz, T, rho, p, nu = comp_macro_params(f, v, gas_params)

    Vx = v.vx - ux
    Vy = v.vy - uy
    Vz = v.vz - uz

    cx = Vx / ((2. * gas_params.Rg * T) ** (1. / 2.))
    cy = Vy / ((2. * gas_params.Rg * T) ** (1. / 2.))
    cz = Vz / ((2. * gas_params.Rg * T) ** (1. / 2.))

    c2 = cx*cx + cy*cy + cz*cz

    Sx = (1. / n) * v.hv3 * np.sum(cx * c2 * f)
    Sy = (1. / n) * v.hv3 * np.sum(cy * c2 * f)
    Sz = (1. / n) * v.hv3 * np.sum(cz * c2 * f)

    fmax = f_maxwell(v, n, ux, uy, uz, T, gas_params.Rg)

    f_plus = fmax * (1. + (4. / 5.) * (1. - gas_params.Pr) * (cx*Sx + cy*Sy + cz*Sz) * (c2 - (5. / 2.)))

    J = nu * (f_plus - f)

    return J, n, ux, uy, uz, T, rho, p, nu

class Config:

    def __init__(self, solver, CFL, tol, init_type = 'default', init_filename = None, tec_save_step = 1e+5):

        self.solver = solver

        self.CFL = CFL
        self.tol = tol

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

        self.path = './' + 'job_flow_' + config.solver + '_' + datetime.now().strftime("%Y.%m.%d_%H:%M:%S") + '/'
        os.mkdir(self.path)

        self.h = np.min(mesh.cell_diam)
        self.tau = self.h * config.CFL / (np.max(np.abs(v.vx_)) * (3.**0.5))

        self.f = np.zeros((mesh.nc, v.nvx, v.nvy, v.nvz))

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
                self.f[ic, :, :, :] = f_maxwell(v, init_data[ic, 0], init_data[ic, 1], init_data[ic, 2], init_data[ic, 3], init_data[ic, 5], gas_params.Rg)

        self.f_plus = np.zeros((mesh.nf, v.nvx, v.nvy, v.nvz)) # Reconstructed values on the right
        self.f_minus = np.zeros((mesh.nf, v.nvx, v.nvy, v.nvz)) # reconstructed values on the left
        self.flux = np.zeros((mesh.nf, v.nvx, v.nvy, v.nvz)) # Flux values
        self.rhs = np.zeros((mesh.nc, v.nvx, v.nvy, v.nvz))
        self.df = np.zeros((mesh.nc, v.nvx, v.nvy, v.nvz)) # Array for increments \Delta f
        self.vn = np.zeros((mesh.nf, v.nvx, v.nvy, v.nvz))
        self.vn_abs = [None] * mesh.nf # approximations of |vn|

        self.vn_abs_r1 = (tuck.tensor((v.vx**2 + v.vy**2 + v.vz**2)**0.5).round(1e-7, rmax = 1)).full()

        for jf in range(mesh.nf):
            self.vn[jf, :, :, :] = mesh.face_normals[jf, 0] * v.vx + mesh.face_normals[jf, 1] * v.vy + mesh.face_normals[jf, 2] * v.vz
            self.vn_abs[jf] = (tuck.tensor(np.abs(self.vn[jf, :, :, :])).round(1e-14, rmax = 6)).full()

        self.diag = np.zeros((mesh.nc, v.nvx, v.nvy, v.nvz)) # part of diagonal coefficient in implicit scheme
        self.incr = np.zeros((v.nvx, v.nvy, v.nvz), dtype = np.double)
        # precompute diag
        for ic in range(mesh.nc):
            for j in range(6):
                jf = mesh.cell_face_list[ic, j]
                vnp = np.where(mesh.cell_face_normal_direction[ic, j] * self.vn[jf, :, :, :] > 0,
                                        mesh.cell_face_normal_direction[ic, j] * self.vn[jf, :, :, :], 0.)
                self.diag[ic, :, :, :] += (mesh.face_areas[jf] / mesh.cell_volumes[ic]) * vnp

        # Arrays for macroparameters
        self.n = np.zeros(mesh.nc)
        self.rho = np.zeros(mesh.nc)
        self.ux = np.zeros(mesh.nc)
        self.uy = np.zeros(mesh.nc)
        self.uz = np.zeros(mesh.nc)
        self.p =  np.zeros(mesh.nc)
        self.T = np.zeros(mesh.nc)
        self.nu = np.zeros(mesh.nc)
        self.data = np.zeros((mesh.nc, 6))

        self.frob_norm_iter = np.array([])

        self.create_res()

    def create_res(self):

        resfile = open(self.path + 'res.txt', 'w')
        resfile.close()

    def update_res(self):

        resfile = open(self.path + 'res.txt', 'a')
        resfile.write('%10.5E \n'% (self.frob_norm_iter[-1]))
        resfile.close()

    def write_tec(self):

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

        write_tecplot(self.mesh, self.data, self.path + 'tec.dat', ('n', 'ux', 'uy', 'uz', 'p', 'T'))

    def save_macro(self):

        np.savetxt(self.path + 'macro.txt', self.data)

    def save_restart(self):
        """ Save the solution into a file
        """
        np.save(self.path + 'restart.npy', self.f)#, fmt='%s')

    def load_restart(self):
        """ Load the solution from a file
        """
        return np.reshape(np.load(self.config.init_filename), (self.mesh.nc, self.v.nvx, self.v.nvy, self.v.nvz))

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
                        self.f_minus[jf, :, :, :] = self.f[ic, :, :, :]
                    else:
                        self.f_plus[jf, :, :, :] = self.f[ic, :, :, :]

            # boundary condition
            # loop over all boundary faces
            for j in range(self.mesh.nbf):
                jf = self.mesh.bound_face_info[j, 0] # global face index
                bc_num = self.mesh.bound_face_info[j, 1]
                bc_type = self.problem.bc_type_list[bc_num]
                bc_data = self.problem.bc_data[bc_num]
                if (self.mesh.bound_face_info[j, 2] == 1):
                    self.f_plus[jf, :, :, :] =  set_bc(self.gas_params, bc_type, bc_data, self.f_minus[jf, :, :, :], self.v, self.vn[jf, :, :, :], self.vn_abs[jf])
                else:
                    self.f_minus[jf, :, :, :] = set_bc(self.gas_params, bc_type, bc_data, self.f_plus[jf, :, :, :], self.v, -self.vn[jf, :, :, :], self.vn_abs[jf])

            # riemann solver - compute fluxes
            for jf in range(self.mesh.nf):
                self.flux[jf, :, :, :] = 0.5 * self.mesh.face_areas[jf] * self.vn[jf, :, :, :] * \
                ((self.f_plus[jf, :, :, :] + self.f_minus[jf, :, :, :]) * self.vn[jf, :, :, :]  - (self.f_plus[jf, :, :, :] - self.f_minus[jf, :, :, :]) * self.vn_abs[jf])
                # flux[jf] = (1. / 2.) * mesh.face_areas[jf] * ((vn * (f_plus[jf, :, :, :] + f_minus[jf, :, :, :])) - (vn_abs * (f_plus[jf, :, :, :] - f_minus[jf, :, :, :])))

            # computation of the right-hand side
            self.rhs[:] = 0.
            for ic in range(self.mesh.nc):
                # sum up fluxes from all faces of this cell
                for j in range(6):
                    jf = self.mesh.cell_face_list[ic, j]
                    self.rhs[ic, :, :, :] += - (self.mesh.cell_face_normal_direction[ic, j]) * (1. / self.mesh.cell_volumes[ic]) * self.flux[jf, :, :, :]
                    # Compute macroparameters and collision integral
                J, self.n[ic], self.ux[ic], self.uy[ic], self.uz[ic], self.T[ic], self.rho[ic], self.p[ic], self.nu[ic] = \
                comp_j(self.f[ic, :, :, :], self.v, self.gas_params)
                self.rhs[ic, :, :, :] += J

            self.frob_norm_iter = np.append(self.frob_norm_iter, np.linalg.norm(self.rhs) / self.mesh.nc)

            self.update_res()
            #
            # update values, expclicit scheme
            #
            if (self.config.solver == 'expl'):
                for ic in range(self.mesh.nc):
                    self.f[ic] = self.f[ic] + self.tau * self.rhs[ic]
            #
            # LU-SGS iteration
            #
            #
            # Backward sweep
            #
            elif (self.config.solver == 'impl'):
                for ic in range(self.mesh.nc - 1, -1, -1):
                    self.df[ic, :, :, :] = self.rhs[ic, :, :, :]
                for ic in range(self.mesh.nc - 1, -1, -1):
                    # loop over neighbors of cell ic
                    for j in range(6):
                        jf = self.mesh.cell_face_list[ic, j]
                        icn = self.mesh.cell_neighbors_list[ic, j] # index of neighbor
                        vnm = np.where(self.mesh.cell_face_normal_direction[ic, j] * self.vn[jf, :, :, :] < 0,
                                            self.mesh.cell_face_normal_direction[ic, j] * self.vn[jf, :, :, :], 0.)
                        if (icn >= 0 ) and (icn > ic):
                            self.df[ic, :, :, :] += -(self.mesh.face_areas[jf] / self.mesh.cell_volumes[ic]) \
                            * vnm * self.df[icn, : , :, :]
                    # divide by diagonal coefficient
                    self.df[ic, :, :, :] = self.df[ic, :, :, :] / ((1. / self.tau + self.nu[ic]) + self.diag[ic])
                #
                # Forward sweep
                #
                for ic in range(self.mesh.nc):
                    # loop over neighbors of cell ic
                    self.incr[:] = 0.
                    for j in range(6):
                        jf = self.mesh.cell_face_list[ic, j]
                        icn = self.mesh.cell_neighbors_list[ic, j] # index of neighbor
                        vnm = np.where(self.mesh.cell_face_normal_direction[ic, j] * self.vn[jf, :, :, :] < 0,
                                            self.mesh.cell_face_normal_direction[ic, j] * self.vn[jf, :, :, :], 0.)
                        if (icn >= 0 ) and (icn < ic):
                            self.incr += -(self.mesh.face_areas[jf] / self.mesh.cell_volumes[ic]) \
                            * vnm * self.df[icn, : , :, :]
                    # divide by diagonal coefficient
                    self.df[ic, :, :, :] += self.incr / ((1. / self.tau + self.nu[ic]) + self.diag[ic])
                self.f += self.df
                #
                # end of LU-SGS iteration
                #
            if ((self.it % config.tec_save_step) == 0):
                self.write_tec()

        self.save_restart()
        self.write_tec()
