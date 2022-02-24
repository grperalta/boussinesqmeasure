# -*- coding: utf-8 -*-

"""
Simple python package for the finite element method of the
Stokes equation on a 2 dimensional mesh. Implementation is
based on the two simplest finite elements for the Stokes
equation, the Taylor-Hood and P1Bubble/P1 on trianglesself.

Gilbert Peralta
Department of Mathematics and Computer Science
University of the Philippines Baguio
Governor Pack Road, Baguio, Philippines 2600
Email: grperalta@up.edu.ph
"""

from .mesh import square_uni_trimesh
from .mesh import samples
from .basis import p2basis
from .basis import p1basis
from .basis import p1bubblebasis
from .basis import get_bubble_coeffs
from .basis import bubble_interpolation
from .assembly import assemble
from .assembly import assemble_laplace
from .assembly import get_fem_data_struct
from .assembly import get_fem_data_struct_laplace
from .assembly import apply_noslip_bc
from .assembly import convection
from .assembly import convection_dual
from .assembly import convection_dual_partition
from .assembly import heat_convection
from .assembly import heat_convection_dual
from .transform import affine_transform
from .quadrature import quad_gauss_tri

__author__ = "Gilbert Peralta"
__copyright__ = "Copyright 2019, Gilbert Peralta"
__version__ = "1.0"
__maintainer__ = "The author"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "19 August 2019"
