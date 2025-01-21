# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:41:43 2024

@author: chunlinxu and GPT4
"""

import numpy as np
import pickle
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rc('font', size=14)
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


class MultiPeakFitter:
    def __init__(self, data, cqw='Clqw', conversion_factor=1, wmin=0,wmax=10, rmsd_threshold=1e-5,
                 stop_improvement=1e-6, min_attempts=2, ratio_threshold=20, 
                 w_dist_threshold=0.01, min_peak_num=1, max_peak_num=4,smear_sigma=1):
        """
        Initialize an instance of the MultiPeakFitter class.
        
        Parameters:
        data (dict): A dictionary containing 'omega' (frequencies), 'q_points' 
                     (q vectors), and 'Clqw' data.
        conversion_factor (float): A factor to convert frequencies to desired units. 
                                   Default is 1 (no conversion).
        wmax (float): The maximum frequency value to consider for fitting. 
                      Default is 10.
        rmsd_threshold (float): The RMSD threshold below which the fit is considered 
                                acceptable. Default is 1e-5.
        stop_improvement (float): The minimum improvement in RMSD required to continue 
                                  adding peaks. Default is 1e-6.
        min_attempts (int): The minimum number of peak fitting attempts before 
                            considering stopping. Default is 2.
        ratio_threshold (float): The ratio threshold for the highest-to-lowest peak 
                                 height, used to discard poor fits. Default is 20.
        w_dist_threshold (float): The minimum frequency difference between peaks to 
                                  consider them as distinct. Default is 0.01.
        min_peak_num (int): The minimum number of peaks to consider for fitting. 
                            Default is 1.
        max_peak_num (int): The maximum number of peaks to consider for fitting. 
                            Default is 4.
        smear_sigma (int): The width of filter function to obtain initial gausses of
                           fitting parameters, raise this value for large omega.
                           Default is 1.
        """
        
        self.omega = data['omega'] * conversion_factor
        if data['q_norms'] is not None:
            self.q_norms = data['q_norms']
        else:
            self.q_norms = np.linalg.norm(data['q_points'], axis=1)
        self.Cqw = data[cqw]
        self.wmin = wmin
        self.wmax = wmax
        self.rmsd_threshold = rmsd_threshold
        self.stop_improvement = stop_improvement
        self.min_attempts = min_attempts
        self.ratio_threshold = ratio_threshold
        self.w_dist_threshold = w_dist_threshold
        self.min_peak_num = min_peak_num
        self.max_peak_num = max_peak_num
        self.smear_sigma = smear_sigma

    @staticmethod
    def function_Ckw(w, w0, gamma, A):
        return A * w**2 * 2 * gamma * w0**2 / ((w**2 - w0**2)**2 + gamma**2 * w**2)

    @staticmethod
    def function_Ckw_overdumped(w, w0, gamma, A):
        return A * w**2 * 2 * gamma * w0**2 / ((w**2 - w0**2)**2 + gamma**2 * w**2)

    def multi_peak_function(self, w, *params):
        num_peaks = len(params) // 3
        result = np.zeros_like(w)
        for i in range(num_peaks):
            w0, gamma, A = params[i*3:(i+1)*3]
            result += self.function_Ckw(w, w0, gamma, A)
        return result

    def estimate_initial_params(self, w_data, Ckw_data, n_peaks):
        if self.smear_sigma < 1:
            peaks, _ = find_peaks(Ckw_data, height=0)
            sorted_peaks = peaks[np.argsort(Ckw_data[peaks])[::-1]]
        else:
            Ckw_smoothed = gaussian_filter1d(Ckw_data, sigma=self.smear_sigma)
            peaks, _ = find_peaks(Ckw_smoothed, height=0)
            sorted_peaks = peaks[np.argsort(Ckw_smoothed[peaks])[::-1]]
        
        params = []
        for i in range(min(n_peaks, len(sorted_peaks))):
            w0_init = w_data[sorted_peaks[i]]
            gamma_init = w0_init / 2
            A_init = Ckw_data[sorted_peaks[i]] / self.function_Ckw(w0_init, w0_init, gamma_init, 1.0)
            params.extend([w0_init, gamma_init, A_init])
        
        while len(params) < n_peaks * 3:
            params.extend(params[-3:])
        # plt.plot(w_data, Ckw_data, 'gray', label='Original Data', lw=3, alpha=0.5)
        # plt.plot(w_data, self.multi_peak_function(w_data, *params), 'b', lw=1, label='Best Fit')
        return params

    @staticmethod
    def calculate_rmsd(y_true, y_pred, ratio_threshold):
        y_min = y_true.max() / ratio_threshold
        mask = np.where(y_true > y_min)
        return np.sqrt(np.mean(((y_true[mask] - y_pred[mask])/y_true[mask])**2))

    def are_peaks_too_close(self, params):
        w0_values = params[0::3]
        for i in range(len(w0_values)):
            for j in range(i + 1, len(w0_values)):
                if abs(w0_values[i] - w0_values[j]) < self.w_dist_threshold:
                    return True
        return False

    @staticmethod
    def are_peaks_negative(params):
        A_values = params[2::3]
        return any(A < 0 for A in A_values)

    def is_peak_ratio_too_large(self, w_data, params):
        peak_heights = [
            abs(self.function_Ckw(w0, w0, gamma, A))
            for w0, gamma, A in zip(params[0::3], params[1::3], params[2::3])
        ]
        return max(peak_heights) / min(peak_heights) > self.ratio_threshold

    def plot_peaks(self,qpts):
        q = np.array(qpts)
        for q in qpts:
            w_data = self.omega[np.where(self.omega < self.wmax)]
            Ckw_data = self.Cqw[q][np.where(self.omega < self.wmax)]
            plt.figure(figsize=(4, 3), dpi=100)
            plt.plot(w_data, Ckw_data, 'k', label='Original Data', lw=1, alpha=1)
            plt.scatter(w_data, Ckw_data,s=4)
            plt.xlabel(r'$\omega$')
            plt.ylabel(r'C($q$,$\omega$)')
            plt.xlim(0, self.wmax)
            # plt.ylim(0, None)
            plt.title(f'\#q: {q}')
            # plt.tight_layout()

    def fit_and_plot(self, q, figname):
        omega_mask = (self.omega < self.wmax) & (self.omega > self.wmin)
        w_data = self.omega[np.where(omega_mask)]
        Ckw_data = self.Cqw[q][np.where(omega_mask)]

        best_rmsd = float('inf')
        best_num_peaks = 0
        best_params = None
        all_results = []

        for num_peaks in range(self.min_peak_num, self.max_peak_num + 1):
            p0_estimated = self.estimate_initial_params(w_data, Ckw_data, num_peaks)
            try:
                popt, _ = curve_fit(self.multi_peak_function, w_data, Ckw_data, p0=p0_estimated)
                fitted_curve = self.multi_peak_function(w_data, *popt)
                rmsd = self.calculate_rmsd(Ckw_data, fitted_curve, self.ratio_threshold)
                all_results.append((num_peaks, rmsd, popt))
                
                if self.are_peaks_too_close(popt) or \
                    self.is_peak_ratio_too_large(w_data, popt) or \
                    self.are_peaks_negative(popt):
                    continue

                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_num_peaks = num_peaks
                    best_params = popt

                if num_peaks >= self.min_attempts:
                    if rmsd < self.rmsd_threshold or (best_rmsd - rmsd) < self.stop_improvement:
                        break

            except RuntimeError:
                continue

        plt.figure(figsize=(4, 3), dpi=100)
        plt.plot(w_data, Ckw_data, 'gray', label='Original Data', lw=3, alpha=0.5)
        if best_num_peaks > 0:
            plt.plot(w_data, self.multi_peak_function(w_data, *best_params), 'b', lw=1, label='Best Fit')
            best_params = best_params.reshape(best_num_peaks, 3)
            best_params = best_params[np.argsort(best_params[:, 0])]
            for params in best_params:
                plt.plot(w_data, self.function_Ckw(w_data, *params), ls=':', lw=1)
                print(q, self.q_norms[q], abs(params[0]), params[1], params[2], end=' ')
            print('')
                
        plt.xlabel(r'$\omega$')
        plt.ylabel(r'C($q$,$\omega$)')
        plt.xlim(self.wmin, self.wmax)
        # plt.ylim(0, None)
        plt.title(f'$q$ = {self.q_norms[q]:.2f} nm$^{{-1}}$')
        plt.legend(frameon=0)
        plt.tight_layout()
        plt.savefig(figname)

if __name__ == '__main__':
    with open('pbttt_backbone.pickle', 'rb') as fin:
        data = pickle.load(fin)
    
    fitter_cfg = dict(data = data, cqw = 'Ctqw',
                      wmin=0,wmax=4,
                      min_attempts=2, 
                      ratio_threshold=50, 
                      w_dist_threshold=0.01, 
                      min_peak_num=1, max_peak_num=2,
                      smear_sigma=2)
    
    
    fitter = MultiPeakFitter(**fitter_cfg)
    for q in range(26,27):
        figname=f'expected_outputs/Cqw_{q}.png'
        fitter.fit_and_plot(q=q, figname=figname)
    
