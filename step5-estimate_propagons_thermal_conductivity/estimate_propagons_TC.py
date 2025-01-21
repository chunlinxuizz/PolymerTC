# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:45:14 2024

@author: Chunlin Xu
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import constants as C
from scipy.optimize import curve_fit
import pickle
import copy
plt.rc('font', size=16)
plt.rc('pgf', preamble=r'\\usepackage{amsmath}')
pgf_config = {
    "font.family":'serif',
    "pgf.rcfonts": False,
    "text.usetex": True,
    "pgf.preamble": r"\\usepackage{unicode-math}\
\\setmainfont{Times New Roman}",
}
rcParams.update(pgf_config)
plt.rc('font', family='Times New Roman')
kB = C.Boltzmann
hbar = C.hbar
T = 300
lengths = np.linspace(0.1,50,100)*1e-9

def _vg_spline(q,nu,s=3):
    nu = nu[np.argsort(q)]
    q = q[np.argsort(q)]
    
    spl = UnivariateSpline(q, nu, s=s)
    spl_slope = spl.derivative()
    slope = spl_slope(q)
    vg = 2*np.pi*slope*1000
    vg[vg<0] = 1e-10
    q_dense = np.linspace(q.min(),q.max(),200)

    return vg, q_dense, spl(q_dense)

def calc_vg_spline(nu_dense,q,nu,s=3,gap=[np.inf,np.inf]):
    if nu.min() < 0.6:
        q = np.insert(q,0,0)
        nu = np.insert(nu,0,0)
    if gap[0] == np.inf:
        _vg , _q_dense, _nu_spl= _vg_spline(q,nu,s=s)
        return np.interp(nu_dense,nu,_vg), np.sum(_vg)/np.sum(_vg>0),[_q_dense],[_nu_spl]
    
    npart = len(gap) 
    vg = []
    q_dense = []
    nu_spl = []
    for n in range(npart-1):
        mask = (nu >= gap[n]) & (nu < gap[n+1])
        _q = q[mask]
        _nu = nu[mask]
        _vg, _q_dense, _nu_spl = _vg_spline(_q,_nu,s=3)
        [ vg.append(x) for x in _vg ]
        q_dense.append(_q_dense)
        nu_spl.append(_nu_spl)

    vg = np.array(vg)
    vg_ave = np.average(vg)
    vg = np.interp(nu_dense, nu, vg)

    return vg, vg_ave, q_dense, nu_spl

def calc_dos_squre(nu_dense,q,nu):
    def quadratic(x,C0):
        return C0*x**2
    C0 = curve_fit(quadratic, q, nu, bounds=([0],[np.inf]))[0]
    dqdw = 1/2 / C0 / np.sqrt(nu_dense / C0) * 1e9 / 1e12 / 2 / np.pi
    return dqdw

def apply_nu_gaps(nu_dense, fun, nu_gaps=[[np.inf,np.inf]]):
    fun = copy.deepcopy(fun)
    for nu_gap in nu_gaps:
        gap = np.where((nu_dense>nu_gap[0]) & (nu_dense<nu_gap[1]))
        fun[gap] = np.average(fun)*1e-20
    return fun

def calc_vg_debye(q,nu):
    def quadratic(x,C0):
        return C0*x
    q = np.insert(q,0,0)
    nu = np.insert(nu,0,0)
    C0 = curve_fit(quadratic, q, nu, bounds=([0],[np.inf]))[0]
    slope = C0[0]
    vg_ave = 2*np.pi*slope*1000

    return vg_ave

def calc_tau(nu_dense, nu, tau, nu_gap = [np.inf, np.inf]):
    def tau_power(nu,n,ln_A):
        return np.exp(n*np.log(nu) + ln_A) 
    
    if nu_gap[1] < nu.max():
        nu_1 = nu[nu<nu_gap[0]]
        nu_2 = nu[nu>nu_gap[1]]
        tau_1 = tau[nu<nu_gap[0]]
        tau_2 = tau[nu>nu_gap[1]]
        n_1,ln_A_1 = np.polyfit(np.log(nu_1), np.log(tau_1), 1)
        n_2,ln_A_2 = np.polyfit(np.log(nu_2), np.log(tau_2), 1)
        
        tau_fit = np.zeros(len(nu_dense))
        tau_fit[nu_dense<=nu_gap[0]] = tau_power(nu_dense[nu_dense<=nu_gap[0]],n_1,ln_A_1)
        tau_fit[nu_dense>=nu_gap[1]] = tau_power(nu_dense[nu_dense>=nu_gap[1]],n_2,ln_A_2)
        gap_map = (nu_dense>nu_gap[0]) & (nu_dense<nu_gap[1])
        tau_fit[gap_map] = np.linspace(tau_power(nu_gap[0],n_1,ln_A_1),
                                       tau_power(nu_gap[1],n_2,ln_A_2),
                                       np.sum(gap_map))
        print('\tn1\tA1\tn2\tA2')
        print('\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(n_1,np.exp(ln_A_1),n_2, np.exp(ln_A_2)))
    
    else:
        n,ln_A = np.polyfit(np.log(nu), np.log(tau), 1)
        tau_fit = tau_power(nu_dense,n,ln_A)
        print('\tn1\tA1')
        print('\t{:.3f}\t{:.3f}'.format(n,np.exp(ln_A)))
    
    return tau_fit

def Kappa_L(L, w,dos, vg, tau, Boltzmann_factor=1):
    tau_boundary = tau * L / (L + abs(vg*tau))
    kappa_w = dos * kB * Boltzmann_factor * vg**2 * tau_boundary
    kappa = np.trapz(kappa_w, w)
    return kappa, kappa_w
    

def Debye_1D(fname, a, volume, nu_D=20, q_D=10, nu_min=0.1, dnu=0.001, q_vq=2,dim=1, s=3, c=2, tau_max=1000,
             nu_gaps=[[np.inf,np.inf]], nu_gap_tau = [np.inf,np.inf], nu_gap_vg=[np.inf,np.inf]):

    data = np.loadtxt(fname).T
    q = data[0]*10 # nm^-1
    nu = data[1]
    Gamma = data[2]
    mask = (nu<nu_D)&(q<q_D)
    q = q[mask]
    Gamma = Gamma[mask]
    nu = nu[mask]

    tau = 1/Gamma/np.pi # s-1
    w = 2*np.pi*np.arange(nu_min,nu.max(),dnu)*1e12 # angular freq. in s-1
    nu_dense = w/2/np.pi/1e12 # freq. in THz
    
    if nu_gap_vg[0] != np.inf:
        nu_gap_vg.insert(0,0)
        nu_gap_vg.append(np.inf)
    
    vg_data = calc_vg_spline(nu_dense, q, nu,s,gap=nu_gap_vg)
    vg_fit, v_ave = vg_data[0], vg_data[1]

    v_s = calc_vg_debye(q[q<q_vq], nu[q<q_vq])
    v_ave = np.sum(vg_fit)/np.sum(vg_fit>0)

    print(f'\n###[{fname}]###')
    print('\tv_ave')
    print('\t{:.3f}\n'.format(v_s))

    tau_fit = calc_tau(nu_dense, nu, tau, nu_gap_tau) 
    tau_fit[tau_fit > tau_max] = tau_max
    tau_fit = tau_fit / 1e12
    
    Boltzmann_factor = 1 # high temperature limit
    if dim == 1:
        # dos = 1/v_ave/np.pi * np.ones(len(w)) / (volume/a*1e9) # 1D Debye
        dos = 1/vg_fit/np.pi / (volume/a*1e9) # 1D derivate
    elif dim == 3:
        dos = w**2 / (2*np.pi**2 * v_s**3)  # 3D Debye
    elif dim == 2:
        dos = w / (2*np.pi * v_ave**2) / c * 1e9  # 2D Debye

    vg_fit = apply_nu_gaps(nu_dense, vg_fit, nu_gaps)
    dos = apply_nu_gaps(nu_dense, dos, nu_gaps=nu_gaps)

    
    print(f'Integration of the DOS: {np.trapz(dos, w)*volume:.3f}')
    print(f'Debye wavevactor: {q_D:.3f} nm^-1')
    wD = v_ave*q_D*1e9
    print(f'Debye frequency: {wD/1e12/2/np.pi:.3f} THz')
    
    kappa_L = []
    for L in lengths:
        kappa, kappa_w = Kappa_L(L, w,dos, vg_fit, tau_fit, Boltzmann_factor)
        kappa_L.append(kappa)

    kappa_inf,kappa_w = Kappa_L(1e-4, w,dos, vg_fit, tau_fit, Boltzmann_factor)
    print(f'Bulk thermal conductivity: {kappa_inf:.3f} W/m-K')
    # plt.figure()
    # plt.plot(nu_dense,kappa_w)
    results = dict(
            nu = nu,
            vg_data = vg_data,
            nu_dense = nu_dense,
            q = q,
            tau = tau,
            tau_fit = tau_fit,
            vg_fit = vg_fit,
            vg_ave = v_ave,
            kappa_w = kappa_w,
            L = lengths,
            kappa_L = kappa_L
        )
    # with open('results/'+fname.split('.')[0]+'.pickle', 'wb') as fout:
    #     pickle.dump(results,fout)
    
    return np.array(kappa_L), results

def plot_dispersion_fitting(data_list):
    colors = ['blue','green','orange','red']
    plt.figure(figsize=(5,4),dpi=300)
    for i,d in enumerate(data_list):
        q = d['q']
        nu = d['nu']
        vg_data = d['vg_data']
        plt.scatter(q,nu,c='none',edgecolor=colors[i],s=20,alpha=0.5)
        q_dense = vg_data[2]
        nu_spl = vg_data[3]
        for j in range(len(q_dense)):
            plt.plot(q_dense[j], nu_spl[j], c=colors[i],lw=2)
    plt.xlim(0,None)
    plt.ylim(0,None)
    plt.xlabel('$q$ (nm$^{{-1}}$)')
    plt.ylabel('$\omega/2\pi$ (THz)')
        
def plot_tau_fitting(data_list):
    colors = ['blue','green','orange','red']
    plt.figure(figsize=(5,4),dpi=200)
    for i,d in enumerate(data_list):
        nu = d['nu']
        tau = d['tau']
        nu_dense = d['nu_dense']
        tau_fit = d['tau_fit']*1e12
        plt.scatter(nu,tau,edgecolor=colors[i],c='none',s=20)
        plt.plot(nu_dense,tau_fit,c=colors[i],lw=1)
    # plt.plot(nu_dense,1/nu_dense,'--',lw=0.5,c='k')
    plt.xlim(0.1,None)
    plt.ylim(0.1,None)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$\omega/2\pi$ (THz)')
    plt.ylabel('$\\tau$ (ps)')

def plot_mfp_fitting(data_list):
    colors = ['blue','green','orange','red']
    plt.figure(figsize=(5,4),dpi=200)
    for i,d in enumerate(data_list):
        nu_dense = d['nu_dense']
        vg_fit = d['vg_fit']
        tau_fit = d['tau_fit']
        mfp = vg_fit*tau_fit*1e9
        plt.scatter(nu_dense[mfp>1e-5],mfp[mfp>1e-5],s=2,edgecolor=colors[i],c='none')
    plt.xlim(0,None)
    plt.ylim(0,None)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('$\omega/2\pi$ (THz)')
    plt.ylabel('$\Lambda$ (nm)')

def plot_vg_fitting(data_list):
    colors = ['blue','green','orange','red']
    plt.figure(figsize=(5,4),dpi=200)
    for i,d in enumerate(data_list):
        nu_dense = d['nu_dense']
        vg_fit = d['vg_fit']
        plt.scatter(nu_dense[vg_fit>1e-5],vg_fit[vg_fit>1e-5],s=2,edgecolor=colors[i],c='none')
    plt.xlim(0,None)
    plt.ylim(0,None)
    plt.xlabel('$\omega/2\pi$ (THz)')
    plt.ylabel('$v$ (m s$^{{-1}}$)')

def plot_mfp_spectra(data_list,dnu = 0.001,nbin = 100,mfp_range = [0.1,100]):
    def _mfp_spectra(MFP, K_w, bins, dw):
        k_l = np.zeros(len(bins))
        for i,l in enumerate(bins):
            k_l[i] = np.sum(K_w[MFP<l])
        k_l = k_l * dw
        return k_l
    plt.figure(figsize=(5,4),dpi=200)
    colors = ['blue','green','orange','red']

    mfp = np.linspace(mfp_range[0], mfp_range[1], nbin)
    dw = dnu*2*np.pi * 1E12
    k_l_low = np.zeros(len(mfp))
    for i,d in enumerate(data_list):
        vg_fit = d['vg_fit']
        tau_fit = d['tau_fit']
        k_w = d['kappa_w']
        _mfp = vg_fit*tau_fit*1e9
        _k_l = _mfp_spectra(_mfp, k_w, mfp, dw)
        k_l_high = k_l_low + _k_l
        # plt.plot(mfp,_k_l,c=colors[i],lw=1)
        plt.fill_between(mfp,k_l_low,k_l_high,color=colors[i],alpha=0.3)
        k_l_low = k_l_high
        
    plt.plot(mfp,k_l_high,c='gray',lw=2)
    plt.xlabel('$\Lambda$ (nm)')
    plt.ylabel('Cumulate $\kappa (\Lambda)$ (W m$^{{-1}}$ K$^{{-1}}$)')
    plt.xlim(0, mfp_range[1])
    plt.ylim(0,None)
    # plt.xscale('log')

def plot_cumulate_kappa(data_list,dnu = 0.001,nbin = 100,colors=['blue','green','orange','red']):
    def _cumulate_kappa(nu, K_w, bins, dw):
        k_l = np.zeros(len(bins))
        for i,l in enumerate(bins):
            k_l[i] = np.sum(K_w[nu<l])
        k_l = k_l * dw
        return k_l
    plt.figure(figsize=(5,4),dpi=200)
    bins = np.linspace(0, 15, nbin) 
    dw = dnu*2*np.pi * 1E12
    k_w_low = np.zeros(nbin)
    for i,d in enumerate(data_list):
        nu = d['nu_dense']
        k_w = d['kappa_w']
        _k_w = _cumulate_kappa(nu, k_w, bins, dw)
        k_w_high = k_w_low + _k_w
        # plt.plot(mfp,_k_l,c=colors[i],lw=1)
        plt.fill_between(bins,k_w_low,k_w_high,color=colors[i],alpha=0.3)
        k_w_low = k_w_high
        
    plt.plot(bins,k_w_high,c='gray',lw=2)
    plt.xlabel('$\omega/2\pi$ (THz)')
    plt.ylabel('Cumulate $\kappa (\omega)$ (W m$^{{-1}}$ K$^{{-1}}$)')
    plt.xlim(0, 15)
    plt.ylim(0,None)

def PBTTT():
    kappa_bcb,d1 = Debye_1D('pbttt_bcb_bcb.txt', a=1.33, volume = 1.05E-27, nu_D=15, q_D=7.2,dim=1, c=1.95,
                nu_gaps =[[0.5,1],[1.8,2.8],[4.5,5.2],[8.4,9.5],[9.6,11.7],[12.4,np.inf]],
                nu_gap_vg=[2,5,9,10])
    kappa_sid,d4 = Debye_1D('pbttt_bcb_sid.txt', a=1.33, volume = 1.05E-27, q_D=7.2,dim=3,c=1.95)
    kappa_T1,d2 = Debye_1D('pbttt_bcb_T1.txt', a=1.33, volume = 1.05E-27, nu_D=15, q_D=7.08,dim=1,c=1.95)
    kappa_T2,d3 = Debye_1D('pbttt_bcb_T2.txt', a=1.33, volume = 1.05E-27, nu_D=15, q_D=7.2,dim=1,c=1.95,
                        nu_gaps =[[1.8,3]],nu_gap_vg=[2,5])

    plot_dispersion_fitting([d1,d2,d3])
    plot_tau_fitting([d1,d2,d3])
    plot_vg_fitting([d1,d2,d3])
    plot_cumulate_kappa([d1,d2,d3])

PBTTT()
