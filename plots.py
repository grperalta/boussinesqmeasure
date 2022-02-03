from __future__ import division
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from boussinesq import BoussinesqMeasureControl
import numpy as np
import pyfem as fem
import os

_SUBDIV = 101
_FEMSPACE = 'p1_bubble'
_PWD = os.getcwd()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def plot(fig, window, mesh, data):
    ax = fig.add_subplot(window, projection = '3d')
    ax.plot_trisurf(mesh.node[:, 0], mesh.node[:, 1], mesh.cell, data,
        linewidth=0.05, antialiased=True, shade=True, cmap=cm.RdBu_r,
        alpha=0.8)
    return ax


def triplots(data):
    """
    Plotting function.
    """
    mesh = fem.square_uni_trimesh(_SUBDIV).femprocess(_FEMSPACE)
    NUMNODE = mesh.num_node

    fig1 = plt.figure(1, figsize=(12,4))
    ax11 = plot(fig1, 131, mesh, data['velocity_x'][:NUMNODE])
    plt.title(r'$u_{h1}^*$', fontsize=16)
    ax12 = plot(fig1, 132, mesh, data['velocity_y'][:NUMNODE])
    plt.title(r'$u_{h2}^*$', fontsize=16)
    ax13 = plot(fig1, 133, mesh, data['temperature'][:NUMNODE])
    plt.title(r'$\theta_h^*$', fontsize=16)
    for ax in [ax11, ax12, ax13]:
        ax.set_zlim3d(-2, 2)
    plt.savefig(_PWD + '/figfiles/optim_states.pdf', bbox_inches = 'tight',
        pad_inches = 0.25)

    fig2 = plt.figure(2, figsize=(12,4))
    ax21 = plot(fig2, 131, mesh, data['fluid_control_x'][:NUMNODE])
    plt.title(r'$\mu_{h1}^*$', fontsize=16)
    ax22 = plot(fig2, 132, mesh, data['fluid_control_y'][:NUMNODE])
    plt.title(r'$\mu_{h2}^*$', fontsize=16)
    ax23 = plot(fig2, 133, mesh, data['heat_control'][:NUMNODE])
    plt.title(r'$\vartheta_h^*$', fontsize=16)
    plt.savefig(_PWD + '/figfiles/optim_control.pdf', bbox_inches = 'tight',
        pad_inches = 0.25)

    fig3 = plt.figure(3, figsize=(12,4))
    ax31 = plot(fig3, 131, mesh, data['dual_velocity_x'][:NUMNODE])
    plt.title(r'$\varphi_{h1}^*$', fontsize=16)
    ax32 = plot(fig3, 132, mesh, data['dual_velocity_y'][:NUMNODE])
    plt.title(r'$\varphi_{h2}^*$', fontsize=16)
    ax33 = plot(fig3, 133, mesh, data['dual_temperature'][:NUMNODE])
    plt.title(r'$\zeta_h^*$', fontsize=16)
    for ax in [ax31, ax32, ax33]:
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
    plt.savefig(_PWD + '/figfiles/optim_adjoint.pdf', bbox_inches = 'tight',
        pad_inches = 0.25)

    fig4 = plt.figure(4, figsize=(12,4))
    ax41 = plot(fig4, 131, mesh, data['pressure'][:NUMNODE])
    plt.title(r'$p_h^*$', fontsize=16)
    ax42 = plot(fig4, 132, mesh, data['dual_pressure'][:NUMNODE])
    plt.title(r'$\pi_h^*$', fontsize=16)
    for ax in [ax41, ax42]:
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
    plt.savefig(_PWD + '/figfiles/optim_pressure.pdf', bbox_inches = 'tight',
        pad_inches = 0.25)
    plt.axis('tight')
    

if __name__ == '__main__':
    FileName = _PWD + "/npyfiles/boussinesqbubble.npy"
    data = np.load(FileName, encoding='latin1', allow_pickle=True)[()]
    triplots(data)
