# -*- coding: utf-8 -*-

from __future__ import division
from scipy import sparse as sp
from .basis import p1basis
from .basis import p2basis
from .basis import p1bubblebasis
from .quadrature import quad_gauss_tri
from .transform import affine_transform
import numpy as np

class FEMDataStruct():
    """
    Class for finite element data structure.
    """

    def __init__(self, name, quad, vbasis, dxvbasis, dyvbasis, pbasis,
        transform, vbasis_localdof, pbasis_localdof):
        """
        Class initialization.

        Parameters
        ----------
            name : str
                name of the finite element
            quad : Quadrature class
                numerical quadrature data structure
            vbasis : array
                velocity basis functions at quadrature nodes
            dxvbasis : array
                derivative with respect to x velocity basis functions
                at quadrature nodes
            dyvbasis : array
                derivative with respect to y velocity basis functions
                at quadrature nodes
            pbasis : array
                pressure basis functions at quadrature nodes
            transform : Transform class
                affine transformations data structure
            vbasis_localdof : int
                number of local degrees of freedom for velocity
            pbasis_localdof : int
                number of local degrees of freedom for pressure
        """

        self.name = name
        self.quad = quad
        self.vbasis = vbasis
        self.dxvbasis = dxvbasis
        self.dyvbasis = dyvbasis
        self.pbasis = pbasis
        self.transform = transform
        self.vbasis_localdof = vbasis_localdof
        self.pbasis_localdof = pbasis_localdof

    def __str__(self):
        return "{} FEM".format(self.name.upper())

def get_fem_data_struct(mesh, quad_order=6, name='taylor_hood'):
    """
    Get the finite element data structure for matrix assembly.

    Parameters
    ----------
        mesh : Mesh class
            triangulation of the domain
        quad_order : int
            order of numerical quadrature
        name : str
            either 'taylor_hood' or 'p1_bubble'
    """
    quad = quad_gauss_tri(quad_order)

    if name is 'taylor_hood':
        vbasis = p2basis(quad.node)
    elif name is 'p1_bubble':
        vbasis = p1bubblebasis(quad.node)
    else:
        raise UserWarning('Invalid name!')
    pbasis = p1basis(quad.node)

    transform = affine_transform(mesh)
    gradvbasis = np.zeros((mesh.num_cell, vbasis.dim, vbasis.dof,
        len(quad.node))).astype(float)

    for gpt in range(len(quad.node)):
        gradvbasis_temp = vbasis.grad[:, gpt, :]
        gradvbasis[:, :, :, gpt] = \
            np.dot(transform.invmatT.reshape(vbasis.dim*mesh.num_cell, vbasis.dim),
            gradvbasis_temp.T).reshape(mesh.num_cell, vbasis.dim,
            vbasis.dof)

    return FEMDataStruct(name, quad, vbasis.val, gradvbasis[:, 0, :, :],
        gradvbasis[:, 1, :, :], pbasis.val, transform, vbasis.dof,
        pbasis.dof)


def get_fem_data_struct_laplace(mesh, quad_order=6, name='p1'):
    """
    Get the finite element data structure for matrix assembly.

    Parameters
    ----------
        mesh : MeshTri class
            triangulation of the domain
        quad_order : int
            order of numerical quadrature
        name : str
            either 'p2' or 'p1'
    """
    quad = quad_gauss_tri(quad_order)

    if name is 'p2':
        vbasis = p2basis(quad.node)
    elif name is 'p1':
        vbasis = p1basis(quad.node)
    else:
        raise UserWarning('Invalid name!')
    pbasis = None

    transform = affine_transform(mesh)
    gradvbasis = np.zeros((mesh.num_cell, vbasis.dim, vbasis.dof,
        len(quad.node))).astype(float)

    for gpt in range(len(quad.node)):
        gradvbasis_temp = vbasis.grad[:, gpt, :]
        gradvbasis[:, :, :, gpt] = \
            np.dot(transform.invmatT.reshape(vbasis.dim*mesh.num_cell, vbasis.dim),
            gradvbasis_temp.T).reshape(mesh.num_cell, vbasis.dim,
            vbasis.dof)

    return FEMDataStruct(name, quad, vbasis.val, gradvbasis[:, 0, :, :],
        gradvbasis[:, 1, :, :], None, transform, vbasis.dof,
        None)


def local_to_global(mesh, j, name='taylor_hood'):
    """
    Returns local to global dof mapping.
    """
    if j < 3:
        return mesh.cell[:, j]
    else:
        if name is 'taylor_hood':
            return mesh.num_node + mesh.cell_to_edge[:, j-3]
        elif name is 'p2':
            return mesh.num_node + mesh.cell_to_edge[:, j-3]
        elif name is 'p1_bubble':
            return mesh.num_node + np.array(range(mesh.num_cell))
        elif name is 'p1':
            pass
        else:
            raise UserWarning('Invalid name!')

def assemble(mesh, femdatastruct):
    """
    Matrix assembly.

    Parameters
    ----------
        mesh : Mesh class
            triangulation of the domain
        femdatastruct : FEMDataStruct class
            finite element data structure

    Returns
    -------
        A, M, Bx, By : tuple of scipy.sparse.csr_matrix
            stifness matrix A, mass matrix M, and components [Bx, By]
            of the discrete divergence matrix
    """
    Ae = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Me = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Bxe = np.zeros((mesh.num_cell, femdatastruct.pbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Bye = np.zeros((mesh.num_cell, femdatastruct.pbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)

    for gpt in range(len(femdatastruct.quad.weight)):
        wgpt = femdatastruct.quad.weight[gpt]
        phi = femdatastruct.vbasis[:, gpt]
        dphidx = femdatastruct.dxvbasis[:, :, gpt]
        dphidy = femdatastruct.dyvbasis[:, :, gpt]
        psi = femdatastruct.pbasis[:, gpt]
        for j in range(femdatastruct.vbasis_localdof):
            for k in range(femdatastruct.vbasis_localdof):
                Me[:, j, k] += wgpt * femdatastruct.transform.det \
                    * phi[j] * phi[k]
                Ae[:, j, k] += wgpt * femdatastruct.transform.det \
                    * dphidx[:, j] * dphidx[:, k]
                Ae[:, j, k] += wgpt * femdatastruct.transform.det \
                    * dphidy[:, j] * dphidy[:, k]
            for k in range(femdatastruct.pbasis_localdof):
                Bxe[:, k, j] += wgpt * femdatastruct.transform.det \
                    * dphidx[:, j] * psi[k]
                Bye[:, k, j] += wgpt * femdatastruct.transform.det \
                    * dphidy[:, j] * psi[k]

    # sparse assembly
    M = sp.csr_matrix((mesh.dof, mesh.dof))
    A = sp.csr_matrix((mesh.dof, mesh.dof))
    Bx = sp.csr_matrix((mesh.num_node, mesh.dof))
    By = sp.csr_matrix((mesh.num_node, mesh.dof))
    for j in range(femdatastruct.vbasis_localdof):
        row = local_to_global(mesh, j, femdatastruct.name)
        for k in range(femdatastruct.vbasis_localdof):
            col = local_to_global(mesh, k, femdatastruct.name)
            M += sp.coo_matrix((Me[:, j, k], (row, col)),
                    shape=(mesh.dof, mesh.dof)).tocsr()
            A += sp.coo_matrix((Ae[:, j, k], (row, col)),
                    shape=(mesh.dof, mesh.dof)).tocsr()
        for k in range(femdatastruct.pbasis_localdof):
            col = local_to_global(mesh, k, femdatastruct.name)
            Bx += sp.coo_matrix((Bxe[:, k, j], (col, row)),
                    shape=(mesh.num_node, mesh.dof)).tocsr()
            By += sp.coo_matrix((Bye[:, k, j], (col, row)),
                    shape=(mesh.num_node, mesh.dof)).tocsr()
    return A, M, Bx, By

def assemble_laplace(mesh, femdatastruct):
    """
    Matrix assembly.

    Parameters
    ----------
        mesh : Mesh class
            triangulation of the domain
        femdatastruct : FEMDataStruct class
            finite element data structure

    Returns
    -------
        A, M : tuple of scipy.sparse.csr_matrix
            stifness matrix A and mass matrix M
    """
    # local matrices
    Ae = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Me = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)

    for gpt in range(len(femdatastruct.quad.weight)):
        wgpt = femdatastruct.quad.weight[gpt]
        phi = femdatastruct.vbasis[:, gpt]
        dphidx = femdatastruct.dxvbasis[:, :, gpt]
        dphidy = femdatastruct.dyvbasis[:, :, gpt]
        for j in range(femdatastruct.vbasis_localdof):
            for k in range(femdatastruct.vbasis_localdof):
                Me[:, j, k] += wgpt * femdatastruct.transform.det \
                    * phi[j] * phi[k]
                Ae[:, j, k] += wgpt * femdatastruct.transform.det \
                    * dphidx[:, j] * dphidx[:, k]
                Ae[:, j, k] += wgpt * femdatastruct.transform.det \
                    * dphidy[:, j] * dphidy[:, k]

    # sparse assembly
    M = sp.csr_matrix((mesh.dof, mesh.dof)).astype(float)
    A = sp.csr_matrix((mesh.dof, mesh.dof)).astype(float)
    for j in range(femdatastruct.vbasis_localdof):
        row = local_to_global(mesh, j, femdatastruct.name)
        for k in range(femdatastruct.vbasis_localdof):
            col = local_to_global(mesh, k, femdatastruct.name)
            M += sp.coo_matrix((Me[:, j, k], (row, col)),
                    shape=(mesh.dof, mesh.dof)).tocsr()
            A += sp.coo_matrix((Ae[:, j, k], (row, col)),
                    shape=(mesh.dof, mesh.dof)).tocsr()
    return A, M

def convection(mesh, u, v, femdatastruct):
    """
    Convection term assembly for ((V.Grad)w, z), where V = (u, v).

    Parameters
    ----------
        mesh : Mesh class
            triangulation of the domain
        u, v : arrays
            component of velocity vector V = (u, v)
        femdatastruct : FEMDataStruct class
            finite element data structure

    Returns
    -------
        scipy.sparse.csr_matrix
    """
    Ne = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)

    for gpt in range(len(femdatastruct.quad.weight)):
        wgpt = femdatastruct.quad.weight[gpt]
        phi = femdatastruct.vbasis[:, gpt]
        dphidx = femdatastruct.dxvbasis[:, :, gpt]
        dphidy = femdatastruct.dyvbasis[:, :, gpt]
        u_temp = np.zeros(mesh.num_cell).astype(float)
        v_temp = np.zeros(mesh.num_cell).astype(float)
        for vrtx in range(femdatastruct.vbasis_localdof):
            u_temp += u[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * phi[vrtx]
            v_temp += v[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * phi[vrtx]
        for j in range(femdatastruct.vbasis_localdof):
            for k in range(femdatastruct.vbasis_localdof):
                Ne[:, j, k] += wgpt * femdatastruct.transform.det \
                    * u_temp * dphidx[:, k] * phi[j]
                Ne[:, j, k] += wgpt * femdatastruct.transform.det \
                    * v_temp * dphidy[:, k] * phi[j]

    N = sp.csr_matrix((mesh.dof, mesh.dof))
    for j in range(femdatastruct.vbasis_localdof):
        row = local_to_global(mesh, j, femdatastruct.name)
        for k in range(femdatastruct.vbasis_localdof):
            col = local_to_global(mesh, k, femdatastruct.name)
            N += sp.coo_matrix((Ne[:, j, k], (row, col)),
                shape=(mesh.dof, mesh.dof)).tocsr()
    return N

def convection_dual(mesh, u, v, femdatastruct):
    """
    Convection term assembly for ((Grad V).T w, z), where V = (u, v).

    Parameters
    ----------
        mesh : Mesh class
            triangulation of the domain
        u, v : arrays
            component of velocity vector V = (u, v)
        femdatastruct : FEMDataStruct class
            finite element data structure

    Returns
    -------
        scipy.sparse.csr_matrix
    """
    Ne1x = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Ne2x = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Ne1y = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Ne2y = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)

    for gpt in range(len(femdatastruct.quad.weight)):
        wgpt = femdatastruct.quad.weight[gpt]
        phi = femdatastruct.vbasis[:, gpt]
        dphidx = femdatastruct.dxvbasis[:, :, gpt]
        dphidy = femdatastruct.dyvbasis[:, :, gpt]
        u_x_temp = np.zeros(mesh.num_cell).astype(float)
        u_y_temp = np.zeros(mesh.num_cell).astype(float)
        v_x_temp = np.zeros(mesh.num_cell).astype(float)
        v_y_temp = np.zeros(mesh.num_cell).astype(float)
        for vrtx in range(femdatastruct.vbasis_localdof):
            u_x_temp += u[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * dphidx[:, vrtx]
            u_y_temp += u[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * dphidy[:, vrtx]
            v_x_temp += v[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * dphidx[:, vrtx]
            v_y_temp += v[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * dphidy[:, vrtx]
        for j in range(femdatastruct.vbasis_localdof):
            for k in range(femdatastruct.vbasis_localdof):
                Ne1x[:, j, k] += wgpt * femdatastruct.transform.det \
                    * u_x_temp * phi[k] * phi[j]
                Ne2x[:, j, k] += wgpt * femdatastruct.transform.det \
                    * u_y_temp * phi[k] * phi[j]
                Ne1y[:, j, k] += wgpt * femdatastruct.transform.det \
                    * v_x_temp * phi[k] * phi[j]
                Ne2y[:, j, k] += wgpt * femdatastruct.transform.det \
                    * v_y_temp * phi[k] * phi[j]

    N = sp.csr_matrix((2*mesh.dof, 2*mesh.dof))
    for j in range(femdatastruct.vbasis_localdof):
        row = local_to_global(mesh, j, femdatastruct.name)
        for k in range(femdatastruct.vbasis_localdof):
            col = local_to_global(mesh, k, femdatastruct.name)
            N += sp.coo_matrix((Ne1x[:, j, k], (row, col)),
                shape=(2*mesh.dof, 2*mesh.dof)).tocsr()
            N += sp.coo_matrix((Ne2x[:, j, k], (row, mesh.dof+col)),
                shape=(2*mesh.dof, 2*mesh.dof)).tocsr()
            N += sp.coo_matrix((Ne1y[:, j, k], (mesh.dof+row, col)),
                shape=(2*mesh.dof, 2*mesh.dof)).tocsr()
            N += sp.coo_matrix((Ne2y[:, j, k], (mesh.dof+row, mesh.dof+col)),
                shape=(2*mesh.dof, 2*mesh.dof)).tocsr()

    return N

def apply_noslip_bc(A, index):
    """
    Apply no-slip boundary conditions.
    """
    A = A.tolil()
    if type(index) is not list:
        index = list(index)
    A[index, :] = 0.0
    A[:, index] = 0.0
    for k in list(index):
        A[k, k] = 1.0
    return A.tocsc()

def convection_dual_partition(mesh, u, v, femdatastruct):
    """
    Convection term assembly for ((Grad V).T w, z), where V = (u, v).

    Parameters
    ----------
        mesh : Mesh class
            triangulation of the domain
        u, v : arrays
            component of velocity vector V = (u, v)
        femdatastruct : FEMDataStruct class
            finite element data structure

    Returns
    -------
        scipy.sparse.csr_matrix
    """
    Ne1x = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Ne2x = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Ne1y = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Ne2y = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)

    for gpt in range(len(femdatastruct.quad.weight)):
        wgpt = femdatastruct.quad.weight[gpt]
        phi = femdatastruct.vbasis[:, gpt]
        dphidx = femdatastruct.dxvbasis[:, :, gpt]
        dphidy = femdatastruct.dyvbasis[:, :, gpt]
        u_x_temp = np.zeros(mesh.num_cell).astype(float)
        u_y_temp = np.zeros(mesh.num_cell).astype(float)
        v_x_temp = np.zeros(mesh.num_cell).astype(float)
        v_y_temp = np.zeros(mesh.num_cell).astype(float)
        for vrtx in range(femdatastruct.vbasis_localdof):
            u_x_temp += u[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * dphidx[:, vrtx]
            u_y_temp += u[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * dphidy[:, vrtx]
            v_x_temp += v[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * dphidx[:, vrtx]
            v_y_temp += v[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * dphidy[:, vrtx]
        for j in range(femdatastruct.vbasis_localdof):
            for k in range(femdatastruct.vbasis_localdof):
                Ne1x[:, j, k] += wgpt * femdatastruct.transform.det \
                    * u_x_temp * phi[k] * phi[j]
                Ne2x[:, j, k] += wgpt * femdatastruct.transform.det \
                    * u_y_temp * phi[k] * phi[j]
                Ne1y[:, j, k] += wgpt * femdatastruct.transform.det \
                    * v_x_temp * phi[k] * phi[j]
                Ne2y[:, j, k] += wgpt * femdatastruct.transform.det \
                    * v_y_temp * phi[k] * phi[j]

    N1 = sp.csr_matrix((mesh.dof, mesh.dof))
    N2 = sp.csr_matrix((mesh.dof, mesh.dof))
    N3 = sp.csr_matrix((mesh.dof, mesh.dof))
    N4 = sp.csr_matrix((mesh.dof, mesh.dof))
    for j in range(femdatastruct.vbasis_localdof):
        row = local_to_global(mesh, j, femdatastruct.name)
        for k in range(femdatastruct.vbasis_localdof):
            col = local_to_global(mesh, k, femdatastruct.name)
            N1 += sp.coo_matrix((Ne1x[:, j, k], (row, col)),
                shape=(mesh.dof, mesh.dof)).tocsr()
            N2 += sp.coo_matrix((Ne2x[:, j, k], (row, col)),
                shape=(mesh.dof, mesh.dof)).tocsr()
            N3 += sp.coo_matrix((Ne1y[:, j, k], (row, col)),
                shape=(mesh.dof, mesh.dof)).tocsr()
            N4 += sp.coo_matrix((Ne2y[:, j, k], (row, col)),
                shape=(mesh.dof, mesh.dof)).tocsr()

    return N1, N2, N3, N4

def heat_convection(mesh, v, femdatastruct):
    """
    Convection term assembly for (U.Grad v, z), where U = (u, v).

    Parameters
    ----------
        mesh : Mesh class
            triangulation of the domain
        u, v : arrays
            component of velocity vector U = (u, v)
        femdatastruct : FEMDataStruct class
            finite element data structure

    Returns
    -------
        scipy.sparse.csr_matrix
    """
    Nxe = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Nye = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)

    for gpt in range(len(femdatastruct.quad.weight)):
        wgpt = femdatastruct.quad.weight[gpt]
        phi = femdatastruct.vbasis[:, gpt]
        dphidx = femdatastruct.dxvbasis[:, :, gpt]
        dphidy = femdatastruct.dyvbasis[:, :, gpt]
        v_x_temp = np.zeros(mesh.num_cell).astype(float)
        v_y_temp = np.zeros(mesh.num_cell).astype(float)
        for vrtx in range(femdatastruct.vbasis_localdof):
            v_x_temp += v[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * dphidx[:, vrtx]
            v_y_temp += v[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * dphidy[:, vrtx]
        for j in range(femdatastruct.vbasis_localdof):
            for k in range(femdatastruct.vbasis_localdof):
                Nxe[:, j, k] += wgpt * femdatastruct.transform.det \
                    * v_x_temp * phi[k] * phi[j]
                Nye[:, j, k] += wgpt * femdatastruct.transform.det \
                    * v_y_temp * phi[k] * phi[j]

    Nx = sp.csr_matrix((mesh.dof, mesh.dof))
    Ny = sp.csr_matrix((mesh.dof, mesh.dof))
    for j in range(femdatastruct.vbasis_localdof):
        row = local_to_global(mesh, j, femdatastruct.name)
        for k in range(femdatastruct.vbasis_localdof):
            col = local_to_global(mesh, k, femdatastruct.name)
            Nx += sp.coo_matrix((Nxe[:, j, k], (row, col)),
                shape=(mesh.dof, mesh.dof)).tocsr()
            Ny += sp.coo_matrix((Nye[:, j, k], (row, col)),
                shape=(mesh.dof, mesh.dof)).tocsr()
    return Nx, Ny

def heat_convection_dual(mesh, v, femdatastruct):
    """
    Convection term assembly for (v Grad u, z).

    Returns
    -------
        scipy.sparse.csr_matrix
    """
    Nxe = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Nye = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)

    for gpt in range(len(femdatastruct.quad.weight)):
        wgpt = femdatastruct.quad.weight[gpt]
        phi = femdatastruct.vbasis[:, gpt]
        dphidx = femdatastruct.dxvbasis[:, :, gpt]
        dphidy = femdatastruct.dyvbasis[:, :, gpt]
        v_temp = np.zeros(mesh.num_cell).astype(float)
        for vrtx in range(femdatastruct.vbasis_localdof):
            v_temp += v[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * phi[vrtx]
        for j in range(femdatastruct.vbasis_localdof):
            for k in range(femdatastruct.vbasis_localdof):
                Nxe[:, j, k] += wgpt * femdatastruct.transform.det \
                    * v_temp * dphidx[:, k] * phi[j]
                Nye[:, j, k] += wgpt * femdatastruct.transform.det \
                    * v_temp * dphidy[:, k] * phi[j]

    Nx = sp.csr_matrix((mesh.dof, mesh.dof))
    Ny = sp.csr_matrix((mesh.dof, mesh.dof))
    for j in range(femdatastruct.vbasis_localdof):
        row = local_to_global(mesh, j, femdatastruct.name)
        for k in range(femdatastruct.vbasis_localdof):
            col = local_to_global(mesh, k, femdatastruct.name)
            Nx += sp.coo_matrix((Nxe[:, j, k], (row, col)),
                shape=(mesh.dof, mesh.dof)).tocsr()
            Ny += sp.coo_matrix((Nye[:, j, k], (row, col)),
                shape=(mesh.dof, mesh.dof)).tocsr()
    return Nx, Ny
