# -*- coding: utf-8 -*-

from __future__ import division
from matplotlib import pyplot as plt
from scipy import sparse as sp
import numpy as np

class MeshTri():
    """
    Triangular mesh class for domain subdivision.
    """

    def __init__(self, node, cell):
        """
        Class initialization.

        Attributes
        ----------
            node : array
                list of coordinates of the node
            cell : array
                list of cell/triangle connectivity
            num_node : int
                number of nodes
            num_cell : int
                number of cells
        """
        self.node = node
        self.cell = cell
        self.num_node = node.shape[0]
        self.num_cell = cell.shape[0]

    def __str__(self):
        string = "Triangular mesh with {} nodes and {} cells."
        return string.format(self.num_node, self.num_cell)

    def __repr__(self):
        return self.__str__()

    def cell_center(self, marked_cell=None):
        """
        Calculate cell barycenters.

        *TODO: Better code
        """
        if marked_cell is None:
            return (self.node[self.cell[:, 0], :]
                + self.node[self.cell[:, 1], :]
                + self.node[self.cell[:, 2], :]) / 3
        else:
            return (self.node[self.cell[marked_cell, 0], :]
                + self.node[self.cell[marked_cell, 1], :]
                + self.node[self.cell[marked_cell, 2], :]) / 3

    def _get_max_edge_lengths(mesh):
        """
        Get the maximum edge length in the mesh.
        """
        edge = mesh.node[mesh.edge[:, 0], :] - mesh.node[mesh.edge[:, 1], :]
        return max(np.linalg.norm(edge, axis=1))

    def _get_edge_center(mesh):
        """
        Get the center of edges in the mesh.
        """
        return 0.5 * (mesh.node[mesh.edge[:, 0], :]
            + mesh.node[mesh.edge[:, 1], :])

    def _refine(mesh):
        """
        Main function for the mesh refinement via bisection.
        """
        # update node
        new_node = np.append(mesh.node, mesh.edge_center(), axis=0)

        # update triangle list
        new_cell = np.zeros((4*mesh.num_cell, 3), dtype=np.int)
        new_cell[:mesh.num_cell, :] \
            = np.vstack([mesh.cell[:, 0],
            mesh.num_node + mesh.cell_to_edge[:, 2],
            mesh.num_node + mesh.cell_to_edge[:, 1]]).T
        new_cell[mesh.num_cell:2*mesh.num_cell, :] \
            = np.vstack([mesh.cell[:, 1],
            mesh.num_node + mesh.cell_to_edge[:, 0],
            mesh.num_node + mesh.cell_to_edge[:, 2]]).T
        new_cell[2*mesh.num_cell:3*mesh.num_cell, :] \
            = np.vstack([mesh.cell[:, 2],
            mesh.num_node + mesh.cell_to_edge[:, 1],
            mesh.num_node + mesh.cell_to_edge[:, 0]]).T
        new_cell[3*mesh.num_cell:, :] \
            = np.vstack([mesh.num_node + mesh.cell_to_edge[:, 0],
            mesh.num_node + mesh.cell_to_edge[:, 1],
            mesh.num_node + mesh.cell_to_edge[:, 2]]).T

        return MeshTri(new_node, new_cell)

    def build_data_struct(self):
        """
        Build the following additional attributes of the mesh:
            edge : array
                node connectivity defining the edge
            num_edge : int
                number of edge
            bdy_edge : array
                indices of edge on the boundary
            bdy_node : array
                indices of node on the boundary
            cell_to_edge : array
                mapping of each cell to the indices of the edge
                containing the cell
        """
        # edge, number of edge, cell_to_edge
        edge_temp = np.hstack([self.cell[:, [1, 2]], self.cell[:, [2, 0]],
            self.cell[:, [0, 1]]]).reshape(3*self.num_cell, 2)
        edge_temp = np.sort(edge_temp)
        self.edge, index, inv_index = np.unique(edge_temp, axis=0,
            return_index=True, return_inverse=True)
        self.num_edge = self.edge.shape[0]
        self.cell_to_edge = inv_index.reshape((self.num_cell, 3))

        # edge_to_cell
        edge_temp = np.hstack([self.cell_to_edge[:, 0],
            self.cell_to_edge[:, 1], self.cell_to_edge[:, 2]])
        cell_temp = np.tile(np.arange(self.num_cell), (1, 3))[0]
        edge_frst, index_edge_frst = np.unique(edge_temp,
            return_index=True)
        edge_last, index_edge_last = np.unique(edge_temp[::-1],
            return_index=True)
        index_edge_last = edge_temp.shape[0] - index_edge_last - 1
        self.edge_to_cell = np.zeros((self.num_edge, 2), dtype=np.int64)
        self.edge_to_cell[edge_frst, 0] = cell_temp[index_edge_frst]
        self.edge_to_cell[edge_last, 1] = cell_temp[index_edge_last]
        self.edge_to_cell[np.nonzero(self.edge_to_cell[:, 0]
            == self.edge_to_cell[:, 1])[0], 1] = -1

        # bdy_edge and bdy_node
        self.bdy_edge = np.nonzero(self.edge_to_cell[:, 1] == -1)[0]
        self.bdy_node = np.unique(self.edge[self.bdy_edge])

        del self.edge_to_cell

    def node_to_cell(self):
        dat = np.hstack([self.cell[:, [1, 2]], self.cell[:, [2, 0]],
            self.cell[:, [0, 1]], self.cell[:, [2, 1]], self.cell[:, [0, 2]],
            self.cell[:, [1, 0]]]).reshape(6*self.num_cell, 2)
        arr, index, inv_index = np.unique(dat, axis=0,
            return_index=True, return_inverse=True)
        ent = np.kron(range(self.num_cell), [1, 1, 1, 1, 1, 1]) + 1
        ent = ent[index]

        # node to cell data stucture
        A = sp.coo_matrix((ent, (arr[:, 0], arr[:, 1])),
            shape=(self.num_node, self.num_node), dtype=int).tocsr()
        return A

    def edge_center(self):
        """
        Calculate edge midpoints.
        """
        try:
            return self._get_edge_center()
        except AttributeError:
            self.build_data_struct()
            return self._get_edge_center()

    def size(self):
        """
        Returns the length of the largest edge length in the mesh.
        """
        try:
            return self._get_max_edge_lengths()
        except AttributeError:
            self.build_data_struct()
            return self._get_max_edge_lengths()

    def int_node(self):
        """
        List of interior nodes.
        """
        try:
            in_node \
            = set(range(self.num_node)).difference(set(self.bdy_node))
        except AttributeError:
            self.build_data_struct()
            in_node \
            = set(range(self.num_node)).difference(set(self.bdy_node))
        return np.asarray(list(in_node))

    def int_edge(self):
        """
        List of interior edges.
        """
        try:
            in_edge \
            = set(range(self.num_edge)).difference(set(self.bdy_edge))
        except AttributeError:
            self.build_data_struct()
            in_edge \
            = set(range(self.num_edge)).difference(set(self.bdy_edge))
        return np.asarray(list(in_edge))

    def refine(self, level=1):
        """
        Refines the triangulation by bisection method.
        """
        ref_mesh = self
        try:
            for it in range(level):
                ref_mesh = ref_mesh._refine()
        except AttributeError:
            self.build_data_struct()
            for it in range(level):
                ref_mesh = ref_mesh._refine()
        return ref_mesh

    def plot(self, fignum=1, show=True, node_numbering=False,
        cell_numbering=False, edge_numbering=False, **kwargs):
        """
        Returns the plot of the mesh.
        """
        fig = plt.figure(fignum)
        ax = fig.add_subplot(111)
        ax.triplot(self.node[:, 0], self.node[:, 1],
            self.cell, lw=0.5, color='black', **kwargs)
        if node_numbering:
            for itr in range(self.num_node):
                ax.text(self.node[itr, 0], self.node[itr, 1], str(itr),
                bbox=dict(facecolor='blue', alpha=0.25))
        if edge_numbering:
            edge_center = self.edge_center()
            for itr in range(self.num_edge):
                ax.text(edge_center[itr, 0], edge_center[itr, 1], str(itr),
                bbox=dict(facecolor='green', alpha=0.25))
        if cell_numbering:
            cell_center = self.cell_center()
            for itr in range(self.num_cell):
                ax.text(cell_center[itr, 0], cell_center[itr, 1], str(itr),
                bbox=dict(facecolor='red', alpha=0.25))
        plt.gca().set_aspect('equal')
        plt.box(False)
        plt.axis('off')
        plt.title(self, fontsize=11)
        if show:
            plt.show()
        return ax

    def min_angle(self):
        """
	Returns the smallest interior angle (in degrees) of the triangles.
	"""
        theta = np.pi
        edge1 = self.node[self.cell[:, 0], :] - self.node[self.cell[:, 1], :]
        edge2 = self.node[self.cell[:, 1], :] - self.node[self.cell[:, 2], :]
        edge3 = self.node[self.cell[:, 2], :] - self.node[self.cell[:, 0], :]
        e1_norm = np.linalg.norm(edge1, axis=1)
        e2_norm = np.linalg.norm(edge2, axis=1)
        e3_norm = np.linalg.norm(edge3, axis=1)
        theta1 = np.arccos(- np.sum(edge1 * edge2, axis=1) / (e1_norm * e2_norm))
        theta2 = np.arccos(- np.sum(edge2 * edge3, axis=1) / (e2_norm * e3_norm))
        theta3 = np.arccos(- np.sum(edge3 * edge1, axis=1) / (e3_norm * e1_norm))
        theta = min(np.array([theta1, theta2, theta3]).flatten())

        return theta * 180. / np.pi

    def adaptive_refine(self, marked_cell=None):
        """
        Refine the set of provided elements.
        Credits: skfem
        """

        def sort_mesh(node, cell):
            """
            Make (0, 2) the longest edge in cell.
            """
            l01 = np.sqrt(np.sum((node[cell[:, 0], :] - node[cell[:, 1], :])**2,
                axis=1))
            l12 = np.sqrt(np.sum((node[cell[:, 1], :] - node[cell[:, 2], :])**2,
                axis=1))
            l02 = np.sqrt(np.sum((node[cell[:, 0], :] - node[cell[:, 2], :])**2,
                axis=1))

            ix01 = (l01 > l02) * (l01 > l12)
            ix12 = (l12 > l01) * (l12 > l02)

            # row swaps
            tmp = cell[ix01, 2]
            cell[ix01, 2] = cell[ix01, 1]
            cell[ix01, 1] = tmp

            tmp = cell[ix12, 0]
            cell[ix12, 0] = cell[ix12, 1]
            cell[ix12, 1] = tmp

            return cell

        def find_edges(mesh, marked_cells):
            """
            Find the edges to split.
            """
            try:
                edges = np.zeros(mesh.num_edge, dtype=np.int64)
            except AttributeError:
                mesh.build_data_struct()

                # TO FIX
                mesh.cell_to_edge = mesh.cell_to_edge[:, [2, 0, 1]]

                edges = np.zeros(mesh.num_edge, dtype=np.int64)
            edges[mesh.cell_to_edge[marked_cells, :].flatten('F')] = 1
            prev_nnz = -1e10

            while np.count_nonzero(edges) - prev_nnz > 0:
                prev_nnz = np.count_nonzero(edges)
                cell_to_edges = edges[mesh.cell_to_edge]
                cell_to_edges[cell_to_edges[:, 0] + cell_to_edges[:, 1] > 0, 2] = 1
                edges[mesh.cell_to_edge[cell_to_edges == 1]] = 1

            return edges

        def split_cells(mesh, edges):
            """Define new elements."""
            ix = (-1) * np.ones(mesh.num_edge, dtype=np.int64)
            ix[edges == 1] = np.arange(np.count_nonzero(edges)) + mesh.num_node
            ix = ix[mesh.cell_to_edge] # (0, 1) (1, 2) (0, 2)

            red =   (ix[:, 0] >= 0) * (ix[:, 1] >= 0) * (ix[:, 2] >= 0)
            blue1 = (ix[:, 0] ==-1) * (ix[:, 1] >= 0) * (ix[:, 2] >= 0)
            blue2 = (ix[:, 0] >= 0) * (ix[:, 1] ==-1) * (ix[:, 2] >= 0)
            green = (ix[:, 0] ==-1) * (ix[:, 1] ==-1) * (ix[:, 2] >= 0)
            rest =  (ix[:, 0] ==-1) * (ix[:, 1] ==-1) * (ix[:, 2] ==-1)

            # new red elements
            cell_red = np.hstack([
                np.vstack([mesh.cell[red, 0], ix[red, 0], ix[red, 2]]),
                np.vstack([mesh.cell[red, 1], ix[red, 0], ix[red, 1]]),
                np.vstack([mesh.cell[red, 2], ix[red, 1], ix[red, 2]]),
                np.vstack([       ix[red, 1], ix[red, 2], ix[red, 0]]),
                ]).T

            # new blue elements
            cell_blue1 = np.hstack([
                np.vstack([mesh.cell[blue1, 1], mesh.cell[blue1, 0], ix[blue1, 2]]),
                np.vstack([mesh.cell[blue1, 1],        ix[blue1, 1], ix[blue1, 2]]),
                np.vstack([mesh.cell[blue1, 2],        ix[blue1, 2], ix[blue1, 1]]),
                ]).T

            cell_blue2 = np.hstack([
                np.vstack([mesh.cell[blue2, 0], ix[blue2, 0],        ix[blue2, 2]]),
                np.vstack([       ix[blue2, 2], ix[blue2, 0], mesh.cell[blue2, 1]]),
                np.vstack([mesh.cell[blue2, 2], ix[blue2, 2], mesh.cell[blue2, 1]]),
                ]).T

            # new green elements
            cell_green = np.hstack([
                np.vstack([mesh.cell[green, 1], ix[green, 2], mesh.cell[green, 0]]),
                np.vstack([mesh.cell[green, 2], ix[green, 2], mesh.cell[green, 1]]),
                ]).T

            refined_edges = mesh.edge[edges == 1, :]

            #print(edges == 1)

            # new nodes
            node = 0.5 * (mesh.node[refined_edges[:, 0], :]
                  + mesh.node[refined_edges[:, 1], :])

            return np.vstack([mesh.node, node]),\
                   np.vstack([mesh.cell[rest, :], cell_red, cell_blue1,
                        cell_blue2, cell_green]), refined_edges

        sorted_mesh = MeshTri(self.node, sort_mesh(self.node, self.cell))
        edges = find_edges(sorted_mesh, marked_cell)
        node, cell, self.refined_edges = split_cells(sorted_mesh, edges)

        return MeshTri(node, cell)

    def regularize(self):
        """
        Barycentric regularization of the triangulation.

        Ref: Quarteroni, Numerical Mathematics, Springer.
        """
        Mat = np.zeros((self.num_node, self.num_node), dtype=np.float)
        for k in range(self.num_cell):
            Mat[self.cell[k, 1], self.cell[k, 2]] = - 1.
            Mat[self.cell[k, 2], self.cell[k, 0]] = - 1.
            Mat[self.cell[k, 0], self.cell[k, 1]] = - 1.
        self.build_data_struct()
        int_node = list(self.int_node())
        for k in int_node:
            Mat[k, k] = len(Mat[k, :].nonzero()[0])
        A = Mat[int_node, :][:, int_node]
        Mat = Mat[int_node, :][:, list(self.bdy_node)]
        new_node = self.node.copy()
        x = self.node[:, 0]
        new_node[int_node, 0] \
            = np.linalg.solve(A, - np.dot(Mat, x[self.bdy_node]))
        y = self.node[:, 1]
        new_node[int_node, 1] \
            = np.linalg.solve(A, - np.dot(Mat, y[self.bdy_node]))

        return MeshTri(new_node, self.cell)

    def _all_node(self):
        """
        Returns all node for interpolation in the Taylor-Hood method.
        """
        return np.append(self.node, self.edge_center(), axis=0)

    def _all_bdy_node(self):
        """
        Returns all boundary node for the Taylor-Hood method.
        """
        return np.append(self.bdy_node, self.num_node + self.bdy_edge)

    def get_num_global_dof(self, name='taylor_hood'):
        """
        Returns the number of global degrees of freedom.
        """
        if name is 'taylor_hood':
            return self.num_node + self.num_edge
        elif name is 'p1_bubble':
            return self.num_node + self.num_cell
        else:
            raise UserWarning('Invalid name!')

    def femprocess(self, name='taylor_hood'):
        """
        Processing mesh for finite element.
        """
        if name is 'taylor_hood':
            self.build_data_struct()
            self.all_node = self._all_node()
            self.all_bdy_node = self._all_bdy_node()
            self.dof = self.get_num_global_dof()
        elif name is 'p1_bubble':
            self.build_data_struct()
            self.dof = self.get_num_global_dof('p1_bubble')
        elif name is 'p2':
            self.build_data_struct()
            self.all_node = self._all_node()
            self.all_bdy_node = self._all_bdy_node()
            self.dof = self.get_num_global_dof()
            self.dof = self.num_node + self.num_edge
        elif name is 'p1':
            self.build_data_struct()
            self.dof = self.num_node
        else:
            raise UserWarning('Invalid name!')
        return self


def square_uni_trimesh(n):
    """
    Generates a uniform triangulation of the unit square with
    vertices at (0, 0), (1, 0), (0, 1) and (1, 1).
    """
    # number of elements
    numelem = 2*(n-1)**2

    # pre-allocation of node array
    node = np.zeros((n**2, 2)).astype(float)

    # generation of node list
    for i in range(1, n+1):
        for j in range(1, n+1):
            # node index
            index = (i-1)*n + j - 1
            # x-coordinates of a node
            node[index, 0] = (j-1) / (n-1)
            # y-coordinate of a node
            node[index, 1] = (i-1) / (n-1)

    # pre-allocation of node connectivity
    cell = np.zeros((numelem, 3)).astype(int)
    ctr = 0

    for i in range(n-1):
        for j in range(n-1):
            # lower right node of the square determined by two intersecting
            # triangles
            lr_node = i*n + j + 1
            # lower left triangle
            cell[ctr, :] = [lr_node, lr_node+n, lr_node-1]
            #upper right triangle
            cell[ctr+1, :] = [lr_node+n-1, lr_node-1, lr_node+n]
            ctr += 2

    return MeshTri(node, cell)

def samples(num=1):
    """
    Examples of coarse triangulations for the unit square.
    """
    if num == 0:
        node = [[0., 0.],
                [1., 0.],
                [1., 1.],
                [0., 1.]]
        cell = [[0, 1, 3],
                [1, 2, 3]]
    if num == 1:
        node = [[0., 0.],
                [1., 0.],
                [0.5, 0.5],
                [0., 1.],
                [1., 1.]]
        cell = [[0, 1, 2],
                [0, 2, 3],
                [2, 4, 3],
                [1, 4, 2]]
    if num == 2:
        node = [[0., 0.],
                [0.5, 0.],
                [1., 0.],
                [0, 0.5],
                [0.5, 0.5],
                [1., 0.5],
                [0., 1.],
                [0.5, 1.],
                [1., 1.]]
        cell = [[0, 1, 4],
                [1, 2, 4],
                [2, 4, 5],
                [0, 3, 4],
                [3, 4, 6],
                [4, 6, 7],
                [4, 7, 8],
                [4, 5, 8]]
    if num == 3:
        node = [[0., 0.],
                [1., 0.],
                [0.25, 0.5],
                [0., 1.],
                [1., 1.]]
        cell = [[0, 1, 2],
                [0, 2, 3],
                [2, 4, 3],
                [1, 4, 2]]
    if num == 4:
        node = [[0.,  0. ],
                [0.5, 0. ],
                [1.,  0. ],
                [0.,  0.5],
                [0.6, 0.4],
                [1.,  0.5],
                [0.,  1. ],
                [0.5, 1. ],
                [1.,  1. ]]
        cell = [[1, 3, 0],
                [3, 1, 4],
                [2, 5, 1],
                [4, 1, 5],
                [4, 6, 3],
                [6, 4, 7],
                [5, 8, 4],
                [7, 4, 8]]
    if num == 5:
        node = [[0., 0.],
                [1., 0.],
                [0.2, 0.4],
                [0.6, 0.4],
                [0., 1.],
                [1., 1.]]
        cell = [[0, 3, 2],
                [0, 1, 3],
                [0, 2, 4],
                [3, 5, 2],
                [1, 5, 3],
                [5, 4, 2]]
    return MeshTri(np.array(node), np.array(cell))
