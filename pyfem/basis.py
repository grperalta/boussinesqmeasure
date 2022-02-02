# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

class Basis():
    """
    Class for finite element basis.
    """

    def __init__(self, val, grad, dof, dim):
        """
        Class initialization.

        Attributes
        ----------
            val : array
                function values at quadrature node of basis functions with
                shape = (no. of basis elements) x (no. quadtrature points)
            grad : array
                gradient values at quadrature node of basis functions with
                shape = (no. of basis elements) x (no. quadrature points)
                x (dim)
            dof : int
                number of local nodal degrees of freedom
            dim : int
                dimension of element

        """
        self.val = val
        self.grad = grad
        self.dof = dof
        self.dim = dim


def p1basis(p):
    """
    P1 Lagrange finite element bases at the points p.

    Returns
    -------
        Basis class
    """
    dof, dim = 3, 2
    x, y = p[:, 0], p[:, 1]
    numnode = p.shape[0]

    val = np.zeros((3, numnode)).astype(float)
    grad = np.zeros((3, numnode, 2)).astype(float)
    one = np.ones(numnode).astype(float)
    zero = np.zeros(numnode).astype(float)

    val[0, :] = 1 - x - y
    val[1, :] = x
    val[2, :] = y

    grad[0, :, :] = np.array([-one, -one]).T
    grad[1, :, :] = np.array([ one, zero]).T
    grad[2, :, :] = np.array([zero,  one]).T

    return Basis(val, grad, dof, dim)


def p1bubblebasis(p):
    """
    P1 Bubble Lagrange finite element bases at the points p.

    Returns
    -------
        Basis class
    """
    dof, dim = 4, 2
    x, y = p[:, 0], p[:, 1]
    numnode = p.shape[0]

    val = np.zeros((4, numnode)).astype(float)
    grad = np.zeros((4, numnode, 2)).astype(float)
    one = np.ones(numnode).astype(float)
    zero = np.zeros(numnode).astype(float)

    val[0, :] = 1 - x - y
    val[1, :] = x
    val[2, :] = y
    val[3, :] = 27 * val[0, :] * val[1, :] * val[2, :]

    grad[0, :, :] = np.array([-one, -one]).T
    grad[1, :, :] = np.array([ one, zero]).T
    grad[2, :, :] = np.array([zero,  one]).T
    for k in [0, 1]:
        grad[3, :, k] = 27 * (val[0, :] * val[1, :] * grad[2, :, k]
                + val[1, :] * val[2, :] * grad[0, :, k]
                + val[2, :] * val[0, :] * grad[1, :, k] )

    return Basis(val, grad, dof, dim)


def p2basis(p):
    """
    P2 Lagrange finite element bases at the points p.

    Returns
    -------
        Basis class
    """
    dof, dim = 6, 2
    x, y = p[:, 0], p[:, 1]
    numnode = p.shape[0]

    val = np.zeros((6, numnode)).astype(float)
    grad = np.zeros((6, numnode, 2)).astype(float)
    zero = np.zeros(numnode).astype(float)

    val[0, :] = (1 - x - y) * (1 - 2*x - 2*y)
    val[1, :] = x * (2*x - 1)
    val[2, :] = y * (2*y - 1)
    val[3, :] = 4 * x * y
    val[4, :] = 4 * y * (1 - x - y)
    val[5, :] = 4 * x * (1 - x - y)

    grad[0, :, :] = np.array([4*x + 4*y - 3, 4*x + 4*y - 3]).T
    grad[1, :, :] = np.array([4*x - 1, zero]).T
    grad[2, :, :] = np.array([zero, 4*y - 1]).T
    grad[3, :, :] = np.array([4*y, 4*x]).T
    grad[4, :, :] = np.array([-4*y, 4*(1 - x - 2*y)]).T
    grad[5, :, :] = np.array([4*(1 - 2*x - y), -4*x]).T

    return Basis(val, grad, dof, dim)


def get_bubble_coeffs(mesh, fun):
    """
    Calculate the coefficients for the bubble basis functions.

    Parameters
    ----------
        mesh : Mesh class
            triangulation of the domain
        fun : callable
            function
    """
    f_bub = np.zeros(mesh.num_cell).astype(float)

    center = mesh.cell_center()
    x_bar, y_bar = center[:, 0], center[:, 1]

    for i in range(mesh.num_cell):
        x_tri = mesh.node[mesh.cell[i, :], 0]
        y_tri = mesh.node[mesh.cell[i, :], 1]
        Acoeff = np.array([[x_tri[0], y_tri[0], 1],
            [x_tri[1], y_tri[1], 1], [x_tri[2], y_tri[2], 1]])
        coef = np.linalg.solve(Acoeff, np.identity(3))
        coef = np.dot(np.array([x_bar[i], y_bar[i], 1]), coef)
        f_bub[i] = fun(x_bar[i], y_bar[i]) - np.dot(coef, fun(x_tri, y_tri))

    return f_bub


def bubble_interpolation(mesh, fun):
    """
    Interpolation with the P1-bubble basis.

    Parameters
    ----------
        mesh : Mesh class
            triangulation of the domain
        fun : callable
            function
    """
    return np.append(fun(mesh.node[:, 0], mesh.node[:, 1]),
        get_bubble_coeffs(mesh, fun), axis=0)
