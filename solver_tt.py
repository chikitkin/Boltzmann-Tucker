import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
import time
from read_starcd import write_tecplot
import tt

class MacroParams:
    def __init__(self, n, ux, uy, uz, T, par):
        self.n = n
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.T = T
        self.rho =  par.m * self.n
        self.p = self.rho * par.Rg * self.T

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

def tt_from_factors(u0, u1, u2):

    F = tt.rand([np.size(u0), np.size(u1), np.size(u2)], 3, [1, 1, 1, 1])
    F_list = F.to_list(F)
    F_list[0][0, :, 0] = u0
    F_list[1][0, :, 0] = u1
    F_list[2][0, :, 0] = u2
    F = F.from_list(F_list)

    return F

def f_maxwell_tt(v, n, ux, uy, uz, T, Rg):

    return n * ((1. / (2. * np.pi * Rg * T)) ** (3. / 2.)) * tt_from_factors(np.exp(-((v.vx_ - ux) ** 2) / (2. * Rg * T)),
                                                                             np.exp(-((v.vy_ - uy) ** 2) / (2. * Rg * T)),
                                                                             np.exp(-((v.vz_ - uz) ** 2) / (2. * Rg * T)))

def div_tt(a, b):

    a_list = a.to_list(a)
    b_list = a.to_list(b)

    c = tt.rand(a.n, 3, a.r)

    c_list = c.to_list(c)

    c_list[0] = a_list[0] / b_list[0]
    c_list[1] = a_list[1] / b_list[1]
    c_list[2] = a_list[2] / b_list[2]

    c = c.from_list(c_list)

    return c

class VelocityGrid:
    def __init__(vx_, vy_, vz_):
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

        self.vx_tt = tt_from_factors(vx_, np.ones(self.nvy), np.ones(self.nvz))
        self.vy_tt = tt_from_factors(np.ones(self.nvx), vy_, np.ones(self.nvz))
        self.vz_tt = tt_from_factors(np.ones(self.nvx), np.ones(self.nvy), vz_)

        self.v2 = (self.vx_tt*self.vx_tt + self.vy_tt*self.vy_tt + self.vz_tt*self.vz_tt).round(1e-7, rmax = 2)

        self.zero = 0. * tt.ones((self.nvx, self.nvy, self.nvz))
        self.ones = tt.ones((self.nvx, self.nvy, self.nvz))

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

def set_bc(gas_params, bc_type, bc_data, f, v, vn, vnp, vnm, tol):
    """Set boundary condition
    """
    # TODO: create general function for symmetrical reflection of a tensor in one dimesnion
    if (bc_type == 'sym-x'): # symmetry in x
        l = f.to_list(f)
        l[0] = l[0][:,::-1,:]
        return f.from_list(l)
    elif (bc_type == 'sym-y'): # symmetry in y
        l = f.to_list(f)
        l[1] = l[1][:,::-1,:]
        return f.from_list(l)
    elif (bc_type == 'sym-z'): # symmetry in z
        l = f.to_list(f)
        l[2] = l[2][:,::-1,:]
        return f.from_list(l)
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
        Ni = v.hv3 * tt.sum((f * vnp).round(tol))
        Nr = v.hv3 * tt.sum((fmax * vnm).round(tol))
        n_wall = - Ni/ Nr
        return n_wall * fmax

def comp_macro_params(f, v, gas_params, tol):
    # takes "F" with (1, 1, 1, 1) ranks to perform to_list function
    # takes precomputed "ones_tt" tensor
    # "ranks" array for mean ranks
    # much less rounding
    Rg = gas_params.Rg
    n = v.hv3 * tt.sum(f)
    if n <= 0.:
        n = 1e+10

    ux = (1. / n) * v.hv3 * tt.sum(vx_tt * f)
    uy = (1. / n) * v.hv3 * tt.sum(vy_tt * f)
    uz = (1. / n) * v.hv3 * tt.sum(vz_tt * f)

    u2 = ux*ux + uy*uy + uz*uz

    T = (1. / (3. * n * Rg)) * (v.hv3 * tt.sum((v2 * f)) - n * u2)
    if T <= 0.:
        T = 1.

    rho = gas_params.m * n
    p = rho * Rg * T
    mu = gas_params.mu(T)
    nu = p / mu

    return n, ux, uy, uz, T, rho, p, nu

def comp_j(f, v, gas_params, tol):

    n, ux, uy, uz, T, rho, p, nu = comp_macro_params(f, v, gas_params, tol)

    cx = tt_from_factors((1. / ((2. * Rg * T) ** (1. / 2.))) * (vx_ - ux), np.ones(self.nvy), np.ones(self.nvz))
    cy = tt_from_factors(np.ones(self.nvx), (1. / ((2. * Rg * T) ** (1. / 2.))) * (vy_ - uy), np.ones(self.nvz))
    cz = tt_from_factors(np.ones(self.nvx), np.ones(self.nvy), (1. / ((2. * Rg * T) ** (1. / 2.))) * (vz_ - uz))

    c2 = ((cx*cx) + (cy*cy) + (cz*cz)).round(1e-7, rmax = 2)

    Sx = (1. / n) * (hv ** 3) * tt.sum(cx * c2 * f)
    Sy = (1. / n) * (hv ** 3) * tt.sum(cy * c2 * f)
    Sz = (1. / n) * (hv ** 3) * tt.sum(cz * c2 * f)

    fmax = f_maxwell_tt(v, n, ux, uy, uz, T, Rg)

    f_plus = fmax * (v.ones_tt + ((4. / 5.) * (1. - gas_params.Pr) * (cx*Sx + cy*Sy + cz*Sz) * ((c2 - (5. / 2.) * v.ones_tt))))
    f_plus = f_plus.round(tol)
    J = nu * (f_plus - f)
    J = J.round(tol)

    return J, n, ux, uy, uz, T, rho, p, nu

def save(filename, f, L):
    """ Save the solution into a file
    """

    m = max(f[i].core.size for i in range(L))

    F = np.zeros((m+4, L))

    for i in range(L):
        F[:4, i] = f[i].r.ravel()
        F[4:f[i].core.size+4, i] = f[i].core.ravel()

    np.save(filename, F)#, fmt='%s')

def load(filename, L, n0, n1, n2):
    """ Load the solution from a file
    """

    F = np.load(filename)

    f = list()

    for i in range(L):

        f.append(tt.rand([n0, n1, n2], 3, F[:4, i]))
        f[i].core = F[4:f[i].core.size+4, i]

    return f

class Config:

    def __init__(self, CFL, tol, filename, init_type = 'default', init_filename = '0', res_filename, tec_save_step):

        self.CFL = CFL
        self.tol = tol

        self.filename = filename

        self.init_type = init_type
        self.init_filename = init_filename

        self.res_filename = res_filename

        self.tec_save_step = tec_save_step

class Solution:

    def __init__(self, gas_params, problem, mesh, v, config):

        self.gas_params = gas_params
        self.problem = problem
        self.mesh = mesh
        self.v = v
        self.config = config


        self.vn = [None] * mesh.nf # list of tensors of normal velocities at each mesh face
        self.vn_tmp = np.zeros((v.nvx, v.nvy, v.nvz))
        self.vnm = [None] * mesh.nf # negative part of vn: 0.5 * (vn - |vn|)
        self.vnp = [None] * mesh.nf # positive part of vn: 0.5 * (vn + |vn|)
        self.vn_abs = [None] * mesh.nf # approximations of |vn|

        for jf in range(mesh.nf):
            self.vn_tmp = mesh.face_normals[jf, 0] * v.vx + mesh.face_normals[jf, 1] * v.vy + mesh.face_normals[jf, 2] * v.vz
            self.vn[jf] = mesh.face_normals[jf, 0] * v.vx_tt + mesh.face_normals[jf, 1] * v.vy_tt + mesh.face_normals[jf, 2] * v.vz_tt
            self.vnp[jf] = tt.tensor(np.where(self.vn_tmp > 0, self.vn_tmp, 0.), eps = config.tol)
            self.vnm[jf] = tt.tensor(np.where(self.vn_tmp < 0, self.vn_tmp, 0.), eps = config.tol)
            self.vn_abs[jf] = tt.tensor(np.abs(self.vn_tmp), rmax = 4)
            self.vn_error = max(self.vn_error, np.linalg.norm(self.vn_abs[jf].full() - np.abs(self.vn_tmp))/
                           np.linalg.norm(np.abs(self.vn_tmp)))

        self.h = np.min(mesh.cell_diam)
        self.tau = self.h * config.CFL / (np.max(np.abs(v.vx_)) * (3.**0.5))

        self.diag = [None] * mesh.nc # part of diagonal coefficient in implicit scheme
        self.diag_r1 = [None] * mesh.nc
        # precompute diag
        # simple approximation for v_abs
        self.vn_abs_r1 = tt.tensor((v.vx**2 + v.vy**2 + v.vz**2)**0.5, rmax = 1)
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
            diag_tt_full = tt.tensor(diag_temp, 1e-7, rmax = 1).full()
            if (np.amax(diag_temp - diag_tt_full) > 0.):
                ind_max = np.unravel_index(np.argmax(diag_temp - diag_tt_full), diag_temp.shape)
                diag_tt_full = (diag_temp[ind_max] / diag_tt_full[ind_max]) * diag_tt_full
            self.diag[ic] = tt.tensor(diag_tt_full)


        # set initial condition
        self.f = [None] * self.mesh.nc # RENAME f!
        if (config.init_type == 'default'):
            for i in range(mesh.nc):
                x = mesh.cell_center_coo[i, 0]
                y = mesh.cell_center_coo[i, 1]
                z = mesh.cell_center_coo[i, 2]
                self.f[i] = problem.f_init(x, y, z, vx, vy, vz)
        elif (config.init_type == 'restart'):
            # restart from distribution function
            self.f = load(config.init_filename, mesh.nc, v.nvx, v.nvy, v.nvz)
        elif (config.init_type == 'macro_restart'):
            # restart form macroparameters array
            init_data = np.loadtxt(config.init_filename)
            for ic in range(mesh.nc):
                self.f[ic] = f_maxwell_tt(v, init_data[ic, 5], init_data[ic, 0], init_data[ic, 1], init_data[ic, 2], init_data[ic, 3], gas_params.Rg)

        # TODO: may be join f_plus and f_minus in one array
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
        self.data = np.zeros((mesh.nc, 7))

        self.frob_norm_iter = np.array([])

    def update_res(self, res_filename):

        resfile = open(res_filename, 'a+')
        resfile.write('%10.5E \n'% (self.frob_norm_iter[-1]))
        resfile.close()

    def write_tec(self):

        fig, ax = plt.subplots(figsize = (20,10))
        line, = ax.semilogy(frob_norm_iter/frob_norm_iter[0])
        ax.set(title='$Steps =$' + str(it))
        plt.savefig('norm_iter.png')
        plt.close()

        self.data[:, 0] = self.n[:]
        self.data[:, 1] = self.ux[:]
        self.data[:, 2] = self.uy[:]
        self.data[:, 3] = self.uz[:]
        self.data[:, 4] = self.p[:]
        self.data[:, 5] = self.T[:]
        self.data[:, 6] = self.rank[:]

        write_tecplot(self.mesh, self.data, 'tec_tt.dat', ('n', 'ux', 'uy', 'uz', 'p', 'T', 'rank'))

    def save_macro(self, filename):

        np.savetxt(filename, self.data)

    def make_time_steps(self, config, nt):

        self.config = config
        self.tau = self.h * config.CFL / (np.max(np.abs(self.v.vx_)) * (3.**0.5))

        it = 0
        while(it < nt):
            it += 1
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
                    self.f_plus[jf] =  set_bc(self.gas_params, bc_type, bc_data, self.f_minus[jf], self.v, self.vn[jf], self.vnp[jf], self.vnm[jf], config.tol)
                else:
                    self.f_minus[jf] = set_bc(self.gas_params, bc_type, bc_data, self.f_plus[jf], self.v, -self.vn[jf], -self.vnm[jf], -self.vnp[jf], config.tol)

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
                comp_j(self.f[ic], self.v, self.gas_params, config.tol)
                self.rhs[ic] += J
                self.rhs[ic] = self.rhs[ic].round(tol)

            self.frob_norm_iter = np.append(self.frob_norm_iter, np.sqrt(sum([(self.rhs[ic].norm())**2 for ic in range(self.mesh.nc)])))

            self.update_res('res.txt')
            #
            # update values, expclicit scheme
            #
    #        for ic in range(mesh.nc):
    #            f[ic] = (f[ic] + tau * rhs[ic]).round(tol)
            '''
            LU-SGS iteration
            '''
            #
            # Backward sweep
            #
            for ic in range(self.mesh.nc - 1, -1, -1):
                self.df[ic] = self.rhs[ic].copy()
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
    #                    df[ic] += -(0.5 * mesh.face_areas[jf] / mesh.cell_volumes[ic]) \
    #                        * (mesh.cell_face_normal_direction[ic, j] * vn[jf] * df[icn] + vn_abs[jf] * df[icn])
                        self.df[ic] += -(self.mesh.face_areas[jf] / self.mesh.cell_volumes[ic]) \
                        * vnm_loc * self.df[icn]
                        self.df[ic] = self.df[ic].round(config.tol)
                # divide by diagonal coefficient
                diag_temp = (self.v.ones * (1./self.tau + nu[ic]) + self.diag_r1[ic]).round(1e-3, rmax = 1)
                self.df[ic] = div_tt(df[ic], diag_temp)
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
    #                    incr+= -(0.5 * mesh.face_areas[jf] /  mesh.cell_volumes[ic]) \
    #                    * (mesh.cell_face_normal_direction[ic, j] * vn[jf] + vn_abs[jf]) * df[icn]
                        incr+= -(self.mesh.face_areas[jf] / self.mesh.cell_volumes[ic]) \
                        * vnm_loc * self.df[icn]
                        incr = incr.round(config.tol)
                # divide by diagonal coefficient
                diag_temp = (self.v.ones * (1./self.tau + self.nu[ic]) + self.diag_r1[ic]).round(1e-3, rmax = 1)
                self.df[ic] += div_tt(incr, diag_temp)
                self.df[ic] = self.df[ic].round(config.tol)
            #
            # Update values
            #
            for ic in range(self.mesh.nc):
                self.f[ic] += self.df[ic]
                self.f[ic] = self.f[ic].round(config.tol)
            '''
            end of LU-SGS iteration
            '''
                    # save rhs norm and tec tile
            if ((it % config.tec_save_step) == 0):

                self.write_tec()

            save(config.filename, self.f, self.mesh.nc)
            self.write_tec()