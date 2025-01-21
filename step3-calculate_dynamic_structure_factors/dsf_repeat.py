# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 19:50:55 2024

@author: Chunlin Xu
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from dynasor import compute_dynamic_structure_factors, Trajectory
from dynasor.post_processing import fourier_cos 
import copy

# set log level
from dynasor.logging_tools import set_logging_level
set_logging_level('DEBUG')

q_max=1.5
q_num=150
direction = 0
dt = 5 #fs
window_size = 4000
window_step = 5
repeat = 20
nve_pre = "mpiexec lmp_mpi -in nve_pre.in -log nve_pre.log"
lmprun = "mpiexec lmp_mpi -in nve.in -log nve.log"

wmax = 1000/dt
dw = 1000/dt/window_size
#q_points = get_spherical_qpoints(traj.cell, q_max=2, max_points=1000)
q_points = np.zeros((q_num,3))
q_norms = np.linspace(0.0,q_max,q_num)
q_points[:,direction] = q_norms
time = np.arange(0, dt*(window_size+1), dt)
omega = np.arange(0, wmax+dw, dw)
data = {
        'Clqw': np.zeros((q_num, window_size+1)),
        'Ctqw': np.zeros((q_num, window_size+1)),
        'Clqt': np.zeros((q_num, window_size+1)),
        'Ctqt': np.zeros((q_num, window_size+1)),
        'Sqw': np.zeros((q_num, window_size+1)),
        'q_points': q_points,
        'q_norms': q_norms,
        'omega': omega,
        'time': time
}

def calc_dsf():
    trajectory_filename = 'dump.lammpstrj'
    traj = Trajectory(
        trajectory_filename,
        trajectory_format='lammps_internal')
    
    sample_raw = compute_dynamic_structure_factors(
        traj, q_points, dt=dt, window_size=window_size,
        window_step=window_step, calculate_currents=True)
    
    data['Clqw'] += sample_raw.Clqw
    data['Ctqw'] += sample_raw.Ctqw
    data['Clqt'] += sample_raw.Clqt
    data['Ctqt'] += sample_raw.Ctqt
    data['Sqw'] += sample_raw.Sqw

def save_fig(data=data, xmax = 1.5,ymax = 20,vmax = None,fname = 'current_corr.png'):
    Clqw = data['Clqw']
    Ctqw = data['Ctqw']
    q_norms = data['q_norms']
    omega = data['omega']

    cmap = ['PuRd','BuPu']
    fig, axes = plt.subplots(figsize=(3.4, 3.8), nrows=2, dpi=300,
                              sharex=True, sharey=True)
    ax = axes[0]
    ax.pcolormesh(q_norms, omega,
                  Clqw.T, cmap=cmap[0], vmin=0, vmax=vmax)
    ax.text(0.05, 0.85, '$C_\mathrm{L}(|\mathbf{q}|, \omega)$', transform=ax.transAxes, color='k')
    ax = axes[1]
    ax.pcolormesh(q_norms, omega,
                  Ctqw.T, cmap=cmap[1], vmin=0, vmax=vmax)
    ax.text(0.05, 0.85, '$C_\mathrm{T}(|\mathbf{q}|, \omega)$', transform=ax.transAxes, color='k')
    ax.set_xlabel('$|\mathbf{q}|$ ($\mathrm{\AA}^{-1}$)')
    ax.set_ylabel('Frequency (THz)', y=1)
    ax.set_ylim([0, ymax])
    ax.set_xlim([0, xmax])
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(fname,dpi=300)

def main():
    os.system(nve_pre)
    for n in range(repeat):
        os.system(lmprun)
        calc_dsf()
        tmp_data = copy.deepcopy(data)
        tmp_data['Clqw'] /= (n+1)
        tmp_data['Ctqw'] /= (n+1)
        tmp_data['Clqt'] /= (n+1)
        tmp_data['Ctqt'] /= (n+1)
        tmp_data['Sqw'] /= (n+1)
        
        save_fig(data=tmp_data, fname=f'step_{n+1}.png', vmax=5e-3)
        with open(f'output_tmp_{n+1}.pickle','wb') as out:
            pickle.dump(tmp_data, out)
        if n>0:
            os.system(f'rm output_tmp_{n}.pickle')
            
    os.system(f'rm output_tmp_{n+1}.pickle')
    os.system('rm dump.lammpstrj')
    data['Clqw'] /= repeat
    data['Ctqw'] /= repeat
    data['Clqt'] /= repeat
    data['Ctqt'] /= repeat
    data['Sqw'] /= repeat
    with open('output.pickle','wb') as out:
        pickle.dump(data, out)
    save_fig(data=data, vmax=5e-3)
    
    #Clqw = np.array([fourier_cos(C, dt)[1] for C in data['Clqt']])
    #Ctqw = np.array([fourier_cos(C, dt)[1] for C in data['Ctqt']])
    #np.save('Clqw.npy', Clqw)
    #np.save('Ctqw.npy', Ctqw)

if __name__ == '__main__':
    main()


