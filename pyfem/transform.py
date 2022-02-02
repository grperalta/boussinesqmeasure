# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

class Transform():
    """
    Class for transformations from the reference element to the
    physical element.
    """

    def __init__(self):
        """
        Class initialization.
        """
        pass


def affine_transform(mesh):
    """
    Generates the affine transformations Tx = Ax + b from the
    reference triangle with vertices at (0, 0), (0, 1) and (1, 0)
    to each element of the mesh.

    Parameter
    ---------
        mesh : Mesh class
            the domain triangulation

    Return
    ------
        Transform class with the following attributes:
            invmatT : array
                inverse transpose of the matrices A with
                shape = (no. of cell) x 2 x 2
            det : array
                absolute value of determinants of A with
                length = (no. of cell)
    """
    transform = Transform()
    transform.invmatT = np.zeros((mesh.num_cell, 2, 2)).astype(float)

    # coordinates of the triangles with local indices 0, 1, 2
    A = mesh.node[mesh.cell[:, 0], :]
    B = mesh.node[mesh.cell[:, 1], :]
    C = mesh.node[mesh.cell[:, 2], :]

    a = B - A
    b = C - A

    transform.invmatT[:, 0, :] = a
    transform.invmatT[:, 1, :] = b
    transform.det = np.abs(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0])
    transform.invmatT = np.linalg.inv(transform.invmatT)

    return transform
