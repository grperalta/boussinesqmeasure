"""
=============================================================================
         MEASURE-VALUED CONTROLS TO THE 2D STEADY BOUSSINESQ EQUATION
=============================================================================

This Python module approximates the solution of an optimal control problem
for the two-dimensional steady Boussinesq equation with regular Borel
measures as controls. The problem being considered is the following:

    min (1/2)|u - ud|^2 + (1/2)|z - zd|^2 + a|q| + b|y|
    subject to the state equation:
        - nu Delta u + (u.Grad) u + Grad p = zg + q     in Omega
                                     div u = 0          in Omega
              - kappa Delta z + (u.Grad) z = y          in Omega
                                         u = 0          on Gamma
                                         z = 0          on Gamma
    over all controls q in M(Omega) x M(Omega) and y in M(Omega).

For more details, please refer to the manuscript:
    Peralta, G., Optimal Borel Measure Controls for the Two-Dimensional
	Stationary Boussinesq System, Preprint 2020.

Gilbert Peralta
Department of Mathematics and Computer Science
University of the Philippines Baguio
Governor Pack Road, Baguio, Philippines 2600
Email: grperalta@up.edu.ph
Date: 2 March 2020
"""

from __future__ import division
from scipy.sparse.linalg import spsolve
from scipy import sparse as sp
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from time import time
import numpy as np
import pyfem as fem
import datetime
import platform
import sys
import os

def print_start():
    """
    Prints machine platform and python version.
    """

    print('*'*78 + '\n')
    start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start of Run: " + start + '\n')

    string = ("PYTHON VERSION: {} \nPLATFORM: {} \nPROCESSOR: {}"
        + "\nVERSION: {} \nMAC VERSION: {}")
    print(string.format(sys.version, platform.platform(),
        platform.uname()[5], platform.version()[:60]
        + '\n' + platform.version()[60:], platform.mac_ver()) + '\n')


def print_end():
    """
    Prints end datetime of execution.
    """

    end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("End of Run: " + end + '\n')
    print('*'*78 + '\n')


def desired_velocity_x(x, y):
    """
    X-component of desired velocity.
    """
    PI2 = 2 * np.pi
    return (1 - np.cos(PI2 * x)) * np.sin(PI2 * y)


def desired_velocity_y(x, y):
    """
    Y-component of desired velocity.
    """
    PI2 = 2 * np.pi
    return np.sin(PI2 * x) * (np.cos(PI2 * y) - 1)


def desired_temperature(x, y):
    """
    Desired temperature.
    """
    PI2 = 2 * np.pi
    return - np.sin(PI2 * x) * np.sin(PI2 * y)


def block_matrix_function(NU, KAPPA, K, M, Bx, By, Z,
    N, N1, N2, N3, N4, N5, N6, ACT_U, ACT_V, ACT_T):
    """
    Generates the block matrix for the coupled state-adjoint system.
    """
    return sp.bmat([
        [NU*K + N,      None,           -Bx.T,      None,        \
         ACT_U,         None,           None,       None],
        [None,          NU*K + N,       -By.T,      M,           \
         None,          ACT_V,          None,       None],
        [Bx,            By,             Z,          None,        \
         None,          None,           None,       None],
        [None,          None,           None,       KAPPA*K + N, \
         None,          None,           None,       ACT_T],
        [-M,            None,           None,       None,        \
         NU*K - N + N1, N2,             -Bx.T,      -N5],
        [None,          -M,             None,       None,        \
         N3,            NU*K - N + N4,  -By.T,      -N6],
        [None,          None,           None,       None,        \
         Bx,            By,             Z,          None],
        [None,          None,           None,       -M,          \
         None,          M,              None,       KAPPA*K - N]
        ], format='csr')


def block_matrix_gradient(NU, KAPPA, K, M, Bx, By, Z,
    N, N1, N2, N3, N4, N5, N6, ACT_U, ACT_V, ACT_T,
    pN, pN1, pN2, pN3, pN4, pN5, pN6, NTx, NTy, NZx, NZy,
    dirichlet_bc):
    """
    Generates the block matrix for the inexact Jacobian of the coupled
    state-adjoint system.
    """
    MAT_SOLVE = sp.bmat([
        [NU*K + N + N1, N3,             -Bx.T,      None,       \
         ACT_U,         None,           None,       None],
        [N2,            NU*K + N + N4,  -By.T,      M,          \
         None,          ACT_V,          None,       None],
        [Bx,            By,             Z,          None,        \
         None,          None,           None,       None],
        [NTx,           NTy,            None,       KAPPA*K + N, \
         None,          None,           None,       ACT_T],
        [-M - 2*pN1,    -pN2 - pN3,     None,       -NZx,        \
         NU*K - N + N1, N2,             -Bx.T,      -pN5],
        [-pN2 - pN3,    -M - 2*pN4,   None,       -NZy,        \
         N3,            NU*K - N + N4,  -By.T,      -pN6],
        [None,          None,           None,       None,        \
         Bx,           By,            Z,          None],
        [-NZx,          -NZy,           None,       -M,          \
         None,          M,              None,       KAPPA*K - N]
        ], format='csr')
    return fem.apply_noslip_bc(MAT_SOLVE, dirichlet_bc)


def initialize_active_sets(DOF):
    """
    Active sets initializer.
    """
    return (np.zeros(DOF), np.zeros(DOF), np.zeros(DOF),
            np.zeros(DOF), np.zeros(DOF), np.zeros(DOF))


def initialize_solution(DOF, NUMNODE):
    """
    Solution initializer.
    """
    return np.zeros(DOF), np.zeros(DOF), np.zeros(NUMNODE), np.zeros(DOF)


class BoussinesqMeasureControl():
    """
    Class for the optimal control problem.
    """
    def __init__(self, NU=1.0, KAPPA=1.0, ALPHA=1e-3, BETA=1e-3, MAXIT=100,
        FEMSPACE='p1_bubble', N=11):
        """
        Class initialization.

        Attributes
        ----------
            NU : float
                fluid viscosity
            KAPPA : float
                thermal conductivity
            ALPHA : float
                penalty parameter for velocity control
            BETA : float
                penalty parameter for thermal control
            MAXIT : int
                maximum number of iterations for Newton's method
            FEMSPACE : str
                finite element space
        """
        self.NU = NU
        self.KAPPA = KAPPA
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.MAXIT = MAXIT
        self.FEMSPACE = FEMSPACE
        self.NSPACESUBDIVION = N

        # Mesh generation
        tic = time()
        self.mesh = fem.square_uni_trimesh(N).femprocess(self.FEMSPACE)
        toc = time()
        self.TIME_MESH_GENERATION = toc - tic

        self._set_attributes()

    def __str__(self):
        """
        Class string representation.
        """
        txt = "="*78+"\n\t\tMEASURE CONTROLS TO BOUSSINESQ EQUATION\n"+"="*78
        txt += '\nAttribute\t Description \t\t\t\t Value\n'
        txt += ('-'*78 + '\nNU\t\t Viscosity \t\t\t\t {}\n'
            + 'KAPPA\t\t Thermal Conductivity \t\t\t {}\n'
            + 'ALPHA\t\t Velocity Control Parameter \t\t {}\n'
            + 'BETA\t\t Thermal Control Parameter \t\t {}\n'
            + 'MAXIT\t\t Maximum Number of Newton Iterations \t {}\n'
            + 'FEMSPACE\t Finite Element Space \t\t\t {}\n')
        txt += '-'*78 + '\n'
        return txt.format(self.NU, self.KAPPA, self.ALPHA, self.BETA,
            self.MAXIT, self.FEMSPACE)

    def _get_dirichlet_bc(self):
        """
        Returns list of indices for the boundary nodes.
        """
        if self.FEMSPACE is 'p1_bubble':
            dirichlet_bc = np.append(self.mesh.bdy_node,
                self.DOF + self.mesh.bdy_node)
            dirichlet_bc = np.append(dirichlet_bc,
                2*self.DOF + self.NUMNODE + self.mesh.bdy_node)
            dirichlet_bc = np.append(dirichlet_bc,
                3*self.DOF + self.NUMNODE + self.mesh.bdy_node)
            dirichlet_bc = np.append(dirichlet_bc,
                4*self.DOF + self.NUMNODE + self.mesh.bdy_node)
            dirichlet_bc = np.append(dirichlet_bc,
                5*self.DOF + 2*self.NUMNODE + self.mesh.bdy_node)
        else:
            dirichlet_bc = np.append(self.mesh.all_bdy_node,
                self.DOF + self.mesh.all_bdy_node)
            dirichlet_bc = np.append(dirichlet_bc,
                2*self.DOF + self.NUMNODE + self.mesh.all_bdy_node)
            dirichlet_bc = np.append(dirichlet_bc,
                3*self.DOF + self.NUMNODE + self.mesh.all_bdy_node)
            dirichlet_bc = np.append(dirichlet_bc,
                4*self.DOF + self.NUMNODE + self.mesh.all_bdy_node)
            dirichlet_bc = np.append(dirichlet_bc,
                5*self.DOF + 2*self.NUMNODE + self.mesh.all_bdy_node)

        return dirichlet_bc

    def _set_attributes(self):
        """
        Set additional attributes of the class.
        """
        self.DOF = self.mesh.dof
        self.NUMNODE = self.mesh.num_node

        # Finite element data structure
        tic = time()
        self.femstruct = fem.get_fem_data_struct(self.mesh,
            name=self.FEMSPACE)
        toc = time()
        self.TIME_FEMDATASTRUCT = toc - tic

        # Dirichlet boundary conditions
        self.dirichlet_bc = self._get_dirichlet_bc()

        # Main matrix assembly
        tic = time()
        self.K, self.M, self.Bx, self.By \
            = fem.assemble(self.mesh, self.femstruct)
        self.Z = sp.spdiags(1e-11*np.ones(self.NUMNODE), 0,
            self.NUMNODE, self.NUMNODE)
        trial_mesh \
            = fem.square_uni_trimesh(self.NSPACESUBDIVION).femprocess('p1')
        self.Mp = fem.assemble_laplace(trial_mesh,
            fem.get_fem_data_struct_laplace(trial_mesh))[1]

        toc = time()
        self.TIME_MATRIX_ASSEMBLY = toc - tic

        # Desired states
        if self.FEMSPACE is 'p1_bubble':
            self.M_velocity_x \
                = self.M * fem.basis.bubble_interpolation(self.mesh,
                desired_velocity_x)
            self.M_velocity_y \
                = self.M * fem.basis.bubble_interpolation(self.mesh,
                desired_velocity_y)
            self.M_temperature \
                = self.M * fem.basis.bubble_interpolation(self.mesh,
                desired_temperature)
        else:
            self.M_velocity_x \
                = self.M * desired_velocity_x(self.mesh.all_node[:, 0],
                self.mesh.all_node[:, 1])
            self.M_velocity_y \
                = self.M * desired_velocity_y(self.mesh.all_node[:, 0],
                self.mesh.all_node[:, 1])
            self.M_temperature \
                = self.M * desired_temperature(self.mesh.all_node[:, 0],
                self.mesh.all_node[:, 1])

    def convection_assembly_function(self, u, v, theta):
        """
        Assembly of convection terms in the coupled state-adjoint system.
        """
        N = fem.convection(self.mesh, u, v, self.femstruct)
        N1, N2, N3, N4 \
            = fem.convection_dual_partition(self.mesh, u, v, self.femstruct)
        N5, N6 \
            = fem.heat_convection_dual(self.mesh, theta, self.femstruct)
        return N, N1, N2, N3, N4, N5, N6

    def convection_assembly_gradient(self, theta, phi, psi, zeta):
        """
        Assembly of convection terms of the Jacobian corresponding to
        the coupled state-adjoint system.
        """
        pN = fem.convection(self.mesh, phi, psi, self.femstruct)
        pN1, pN2, pN3, pN4 \
            = fem.convection_dual_partition(self.mesh, phi, psi,
            self.femstruct)
        pN5, pN6 \
            = fem.heat_convection_dual(self.mesh, zeta, self.femstruct)
        NTx, NTy \
            = fem.heat_convection(self.mesh, theta, self.femstruct)
        NZx, NZy \
            = fem.heat_convection(self.mesh, zeta, self.femstruct)
        return pN, pN1, pN2, pN3, pN4, pN5, pN6, NTx, NTy, NZx, NZy

    def BLOCK_MATRIX(self, u, v, theta, phi, psi, zeta,
        ACT_U, ACT_V, ACT_T):
        """
        Generates the block matrix for the coupled state-adjoint system.
        """
        self.TIME_CONVECTION_ASSEMBLY = 0.
        tic = time()
        N, N1, N2, N3, N4, N5, N6 = \
            self.convection_assembly_function(u, v, theta)
        pN, pN1, pN2, pN3, pN4, pN5, pN6, NTx, NTy, NZx, NZy = \
            self.convection_assembly_gradient(theta, phi, psi, zeta)
        toc = time()
        self.TIME_CONVECTION_ASSEMBLY = toc - tic

        self.TIME_BLOCK_MATRIX_ASSEMBLY_RHS = 0.
        tic = time()
        MAT_RHS = block_matrix_function(self.NU, self.KAPPA,
            self.K, self.M, self.Bx, self.By, self.Z,
            N, N1, N2, N3, N4, N5, N6, ACT_U, ACT_V, ACT_T)
        toc = time()
        self.TIME_BLOCK_MATRIX_ASSEMBLY_RHS = toc - tic

        self.TIME_BLOCK_MATRIX_ASSEMBLY_GRADIENT = 0.
        tic = time()
        MAT_SOLVE = block_matrix_gradient(self.NU, self.KAPPA,
            self.K, self.M, self.Bx, self.By, self.Z,
            N, N1, N2, N3, N4, N5, N6, ACT_U, ACT_V, ACT_T,
            pN, pN1, pN2, pN3, pN4, pN5, pN6, NTx, NTy, NZx, NZy,
            self.dirichlet_bc)
        toc = time()
        self.TIME_BLOCK_MATRIX_ASSEMBLY_GRADIENT = toc - tic
        return MAT_RHS, MAT_SOLVE

    def RHS_NEWTON(self, MAT_RHS, ACTIVE_SET,
        u, v, p, theta, phi, psi, pi, zeta, GAMMA):
        """
        Construct right hand side in the Newton solver.
        """
        F = MAT_RHS * np.hstack([u, v, p, theta, phi, psi, pi, zeta])
        F += np.hstack([
            - self.ALPHA*GAMMA*(ACTIVE_SET[0] - ACTIVE_SET[1]),
            - self.ALPHA*GAMMA*(ACTIVE_SET[2] - ACTIVE_SET[3]),
            np.zeros(self.NUMNODE),
            - self.BETA*GAMMA*(ACTIVE_SET[4] - ACTIVE_SET[5]),
            self.M_velocity_x, self.M_velocity_y, np.zeros(self.NUMNODE),
            self.M_temperature])
        F[self.dirichlet_bc] = 0.
        return F

    def RHS_SEMISMOOTH(self, MAT_RHS, u, v, p, theta, phi, psi, pi, zeta,
        mu_u, mu_v, mu_t):
        """
        Construct right hand side in the semismooth Newton method.
        """
        F = MAT_RHS * np.hstack([u, v, p, theta, phi, psi, pi, zeta])
        F += np.hstack([-mu_u, -mu_v, np.zeros(self.NUMNODE), -mu_t,
            self.M_velocity_x, self.M_velocity_y, np.zeros(self.NUMNODE),
            self.M_temperature])
        F[self.dirichlet_bc] = 0.
        return F

    def MAT_ACTIVE_SET(self, ACTIVE_SET, GAMMA):
        """
        Returns matrices associated with the active sets.
        """
        ACT_U = sp.spdiags(GAMMA*(ACTIVE_SET[0] + ACTIVE_SET[1]), 0,
            self.DOF, self.DOF)
        ACT_V = sp.spdiags(GAMMA*(ACTIVE_SET[2] + ACTIVE_SET[3]), 0,
            self.DOF, self.DOF)
        ACT_T = sp.spdiags(GAMMA*(ACTIVE_SET[4] + ACTIVE_SET[5]), 0,
            self.DOF, self.DOF)
        return ACT_U, ACT_V, ACT_T

    def newton_solver(self, MAT_SOLVE, RHS):
        """
        Newton solver for the nonlinear primal-dual systems.
        """
        index = 3*self.DOF + self.NUMNODE

        self.TIME_NEWTON_SOLVER = 0.
        tic = time()
        sol = spsolve(MAT_SOLVE, RHS)
        toc = time()
        self.TIME_NEWTON_SOLVER = toc - tic

        return (sol[:self.DOF], sol[self.DOF:2*self.DOF],
                sol[2*self.DOF:2*self.DOF + self.NUMNODE],
                sol[2*self.DOF + self.NUMNODE:index],
                sol[index:index + self.DOF],
                sol[index + self.DOF:index + 2*self.DOF],
                sol[index + 2*self.DOF:index + 2*self.DOF + self.NUMNODE],
                sol[index + 2*self.DOF + self.NUMNODE:])

    def update_solution(self, MAT_SOLVE, RHS, u, v, p, theta,
        phi, psi, pi, zeta):
        """
        Solution update for the Newton solver.
        """
        u_new, v_new, p_new, theta_new, phi_new, psi_new, pi_new, zeta_new \
            = self.newton_solver(MAT_SOLVE, RHS)
        return (u - u_new, v - v_new, p - p_new, theta - theta_new,
                phi - phi_new, psi - psi_new, pi - pi_new, zeta - zeta_new,
                self.check_tolerance(u_new, v_new, pi_new, theta_new, phi_new,
                psi_new, pi_new, zeta_new))

    def check_tolerance(self, u, v, p, theta, phi, psi, pi, zeta):
        """
        Stopping criterion for the Newton solver.
        """
        error = 0.
        for var in [u, v, theta, phi, psi, zeta]:
            norm = np.sqrt(np.dot(self.M * var, var))
            error = max(error, norm)
        error += max(error, np.sqrt(np.dot(self.Mp * p, p)))
        error += max(error, np.sqrt(np.dot(self.Mp * pi, pi)))
        return error

    def active_set_update(self, phi, psi, zeta):
        """
        Updates the active sets for the Newton solver.
        """
        return ((phi > self.ALPHA).astype(float),
                (phi < -self.ALPHA).astype(float),
                (psi > self.ALPHA).astype(float),
                (psi < -self.ALPHA).astype(float),
                (zeta > self.BETA).astype(float),
                (zeta < -self.BETA).astype(float))

    def active_set_change(self, ACTIVE_SET, ACTIVE_SET_OLD):
        """
        Calculate the number of changes for the active set.
        """
        change = np.zeros(self.DOF)
        for k in range(len(ACTIVE_SET)):
            change += ACTIVE_SET[k] - ACTIVE_SET_OLD[k]
        return len(change[change.nonzero()])

    def active_set_update_semismooth(self, ACTIVE_SET, mu_u, mu_v, mu_t,
        phi, psi, zeta):
        """
        Updates the active sets for the semi-smooth Newton method.
        """
        return ((- mu_u + phi > self.ALPHA).astype(float),
                (- mu_u + phi < -self.ALPHA).astype(float),
                (- mu_v + psi > self.ALPHA).astype(float),
                (- mu_v + psi < -self.ALPHA).astype(float),
                (- mu_t + zeta > self.BETA).astype(float),
                (- mu_t + zeta < -self.BETA).astype(float))

    def control_update(self, ACTIVE_SET, mu_u, mu_v, mu_t,
        phi, psi, zeta):
        """
        Updates the measure controls.
        """
        ACT_U, ACT_V, ACT_T = self.MAT_ACTIVE_SET(ACTIVE_SET, 1.0)
        mu_u = ACT_U * (-phi + mu_u) \
            + self.ALPHA*(ACTIVE_SET[0] - ACTIVE_SET[1])
        mu_v = ACT_V * (-psi + mu_v) \
            + self.ALPHA*(ACTIVE_SET[2] - ACTIVE_SET[3])
        mu_t = ACT_T * (-zeta + mu_t) \
            + self.BETA*(ACTIVE_SET[4] - ACTIVE_SET[5])
        return mu_u, mu_v, mu_t


def SEMISMOOTH_NEWTON(cls):
    """
    Semismooth Newton algorithm.
    """
    tic = time()
    string = '\t>>> {} POINTS CHANGED IN ACTIVE SET\n'
    TOTAL_MATRIX_SIZE = 2 * (6*BMC.DOF + 2*BMC.NUMNODE, )
    print('-'*78 + '\n\nSEMISMOOTH NETWON METHOD --- '
        + "Size of Total Matrix: {}\n".format(TOTAL_MATRIX_SIZE))

    print('>>> CONTINUATION STRATEGY\n')
    # initialize and copy active sets
    ACTIVE_SET = initialize_active_sets(cls.DOF)
    ACTIVE_SET_OLD = initialize_active_sets(cls.DOF)

    u, v, p, theta = initialize_solution(cls.DOF, cls.NUMNODE)
    phi, psi, pi, zeta = initialize_solution(cls.DOF, cls.NUMNODE)

    for GAMMA in [10**k for k in range(8)]:
        print('>>> SOLVING FOR GAMMA = {:1.0e}\n'.format(GAMMA))

        for it in range(cls.MAXIT):
            print("    Iteration {}:".format(it+1))
            # Newton's method
            print("\tNEWTON'S NONLINEAR SOLVER")
            ACT_U, ACT_V, ACT_T = cls.MAT_ACTIVE_SET(ACTIVE_SET, GAMMA)

            for nit in range(cls.MAXIT):
                print("\t{}\tAssembling total matrix ...".format(nit+1))
                MAT_RHS, MAT_SOLVE \
                    = cls.BLOCK_MATRIX(u, v, theta, phi, psi, zeta,
                    ACT_U, ACT_V, ACT_T)
                print("\t\t\tElapsed time: {:.12e} seconds".format(
                    cls.TIME_BLOCK_MATRIX_ASSEMBLY_RHS
                    + cls.TIME_BLOCK_MATRIX_ASSEMBLY_GRADIENT))
                F = cls.RHS_NEWTON(MAT_RHS, ACTIVE_SET,
                    u, v, p, theta, phi, psi, pi, zeta, GAMMA)
                print("\t\tSolving sparse linear system ...")
                u, v, p, theta, phi, psi, pi, zeta, error \
                    = cls.update_solution(MAT_SOLVE, F,
                    u, v, p, theta, phi, psi, pi, zeta)
                print("\t\t\tElapsed time: {:.12e} seconds".format(
                    cls.TIME_NEWTON_SOLVER))
                print("\t\tTol = {:.10e}".format(error))
                if error < 1e-10:
                    break

            ACTIVE_SET = cls.active_set_update(phi, psi, zeta)
            update = cls.active_set_change(ACTIVE_SET, ACTIVE_SET_OLD)
            print(string.format(update))
            if update == 0:
                break
            ACTIVE_SET_OLD = ACTIVE_SET
        if it == 0:
            break

    print('>>> PRIMAL-DUAL ACTIVE SET STRATEGY')
    mu_u = - GAMMA * (np.maximum(0, phi - cls.ALPHA)
        + np.minimum(0, phi + cls.ALPHA))
    mu_v = - GAMMA * (np.maximum(0, psi - cls.ALPHA)
        + np.minimum(0, psi + cls.ALPHA))
    mu_t = - GAMMA * (np.maximum(0, zeta - cls.BETA)
        + np.minimum(0, zeta + cls.BETA))
    ACTIVE_SET = cls.active_set_update_semismooth(ACTIVE_SET,
        mu_u, mu_v, mu_t, phi, psi, zeta)
    ACTIVE_SET_OLD = ACTIVE_SET

    for it in range(cls.MAXIT):
        print("    Iteration {}:".format(it+1))
        for nit in range(cls.MAXIT):
            # Newton's method
            print("\tNEWTON'S NONLINEAR SOLVER")
            print("\t{}\tAssembling total matrix ...".format(nit+1))
            MAT_RHS, MAT_SOLVE \
                = cls.BLOCK_MATRIX(u, v, theta, phi, psi, zeta,
                None, None, None)
            print("\t\t\tElapsed time: {:.12e} seconds".format(
                cls.TIME_BLOCK_MATRIX_ASSEMBLY_RHS
                + cls.TIME_BLOCK_MATRIX_ASSEMBLY_GRADIENT))
            F = cls.RHS_SEMISMOOTH(MAT_RHS, u, v, p, theta,
                phi, psi, pi, zeta, mu_u, mu_v, mu_t)
            print("\t\tSolving sparse linear system ...")
            u, v, p, theta, phi, psi, pi, zeta, error \
                = cls.update_solution(MAT_SOLVE, F,
                u, v, p, theta, phi, psi, pi, zeta)
            print("\t\t\tElapsed time: {:.12e} seconds".format(
                cls.TIME_NEWTON_SOLVER))
            print("\t\tTol = {:.10e}".format(error))
            if error < 1e-10:
                break

        mu_u, mu_v, mu_t = cls.control_update(ACTIVE_SET, mu_u, mu_v, mu_t,
            phi, psi, zeta)
        ACTIVE_SET = cls.active_set_update_semismooth(ACTIVE_SET,
            mu_u, mu_v, mu_t, phi, psi, zeta)
        update = cls.active_set_change(ACTIVE_SET, ACTIVE_SET_OLD)
        print(string.format(update))
        if update == 0:
            break
        ACTIVE_SET_OLD = ACTIVE_SET

    toc = time()
    print("Elapsed time for semismooth Newton method: {:.12e} seconds\n".format(
        toc - tic))
    return u, v, theta, p, mu_u, mu_v, mu_t, phi, psi, zeta, pi


def triplot(cls, u, v, theta, p, mu_u, mu_v, mu_t, phi, psi, zeta, pi):
    """
    Plotting function.
    """
    x, y = cls.mesh.node[:, 0], cls.mesh.node[:, 1]

    def plot(fig, window, data):
        ax = fig.add_subplot(window, projection = '3d')
        ax.plot_trisurf(x, y, cls.mesh.cell, data,
            linewidth=0.05, antialiased=True, shade=True,
            cmap=cm.RdBu_r, alpha=0.8)
        return ax

    fig1 = plt.figure(1)
    plot(fig1, 121, desired_velocity_x(x, y))
    plot(fig1, 122, u[:cls.NUMNODE])
    plt.suptitle('X-COMPONENT OF VELOCITY')

    fig2 = plt.figure(2)
    plot(fig2, 121, desired_velocity_y(x, y))
    plot(fig2, 122, v[:cls.NUMNODE])
    plt.suptitle('Y-COMPONENT OF VELOCITY')

    fig3 = plt.figure(3)
    plot(fig3, 121, desired_temperature(x, y))
    plot(fig3, 122, theta[:cls.NUMNODE])
    plt.suptitle('TEMPERATURE')

    fig4 = plt.figure(4)
    plot(fig4, 111, p[:cls.NUMNODE])
    plt.title('PRESSURE')

    fig5 = plt.figure(5)
    plot(fig5, 111, mu_u[:cls.NUMNODE])
    plt.title('X-COMPONENT OF VELOCITY CONTROL')

    fig6 = plt.figure(6)
    plot(fig6, 111, mu_v[:cls.NUMNODE])
    plt.title('Y-COMPONENT OF VELOCITY CONTROL')

    fig7 = plt.figure(7)
    plot(fig7, 111, mu_t[:cls.NUMNODE])
    plt.title('THERMAL CONTROL')

    fig8 = plt.figure(8)
    plot(fig8, 111, phi[:cls.NUMNODE])
    plt.title('X-COMPONENT OF ADJOINT VELOCITY')

    fig9 = plt.figure(9)
    plot(fig9, 111, psi[:cls.NUMNODE])
    plt.title('Y-COMPONENT OF ADJOINT VELOCITY')

    fig10 = plt.figure(10)
    plot(fig10, 111, zeta[:cls.NUMNODE])
    plt.title('ADJOINT TEMPERATURE')

    fig11 = plt.figure(11)
    plot(fig11, 111, pi[:cls.NUMNODE])
    plt.title('ADJOINT PRESSURE')

    plt.axis('tight')
    plt.show()


if __name__ == '__main__':
    print_start()
    dict_names = {1: 'p1_bubble', 2: 'taylor_hood'}
    FEMNUM = int(input('>>> Input finite element type '
        + '(1: P1BUBBLE, 2: TAYLORHOOD): '))
    SUBDIV = int(input('>>> Input number of subdivision: '))
    BMC = BoussinesqMeasureControl(N=SUBDIV, FEMSPACE=dict_names[FEMNUM])
    print(BMC)
    print(">>> Mesh generation elapsed time: "
        + "{:.12e} seconds".format(BMC.TIME_MESH_GENERATION))
    print("    " + str(BMC.mesh) + "\n")
    print(">>> FEM data struct elapsed time: "
        + "{:.12e} seconds \n".format(BMC.TIME_FEMDATASTRUCT))
    print(">>> Matrix assembly elapsed time: "
        + "{:.12e} seconds\n ".format(BMC.TIME_MATRIX_ASSEMBLY))

    u, v, theta, p, mu_u, mu_v, mu_t, phi, psi, zeta, pi \
        = SEMISMOOTH_NEWTON(BMC)

    if BMC.FEMSPACE is 'p1_bubble':
        FileName = os.getcwd() + '/npyfiles/boussinesqbubble.npy'
    else:
        FileName = os.getcwd() + '/npyfiles/boussinesqtaylorhood.npy'
    np.save(FileName, {'velocity_x': u, 'velocity_y': v,
        'pressure': p, 'temperature': theta, 'fluid_control_x': mu_u,
        'fluid_control_y': mu_v, 'heat_control': mu_t,
        'dual_velocity_x': phi, 'dual_velocity_y': psi,
        'dual_pressure': pi, 'dual_temperature': zeta})

    triplot(BMC, u, v, theta, p, mu_u, mu_v, mu_t, phi, psi, zeta, pi)
    print_end()
